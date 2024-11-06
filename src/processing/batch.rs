use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, warn};

use crate::{
    error::{EngineError, Result},
    model::ModelRuntime,
    metrics::MetricsCollector,
    types::ProcessingOutput,
};

use super::common::{ProcessingStats, ProcessingResult};
use super::Processor;

/// Dynamic batch size management
#[derive(Debug)]
struct BatchSizeManager {
    current_size: usize,
    min_size: usize,
    max_size: usize,
    growth_factor: f32,
    reduction_factor: f32,
    success_streak: usize,
    failure_streak: usize,
}

impl BatchSizeManager {
    fn new(min_size: usize, max_size: usize) -> Self {
        Self {
            current_size: min_size,
            min_size,
            max_size,
            growth_factor: 1.2,
            reduction_factor: 0.5,
            success_streak: 0,
            failure_streak: 0,
        }
    }

    fn handle_success(&mut self) {
        self.success_streak += 1;
        self.failure_streak = 0;

        // Grow batch size after consistent successes
        if self.success_streak >= 3 {
            let new_size = (self.current_size as f32 * self.growth_factor) as usize;
            self.current_size = new_size.min(self.max_size);
            self.success_streak = 0;
        }
    }

    fn handle_failure(&mut self) {
        self.failure_streak += 1;
        self.success_streak = 0;

        // Reduce batch size on failure
        self.current_size = (self.current_size as f32 * self.reduction_factor) as usize;
        self.current_size = self.current_size.max(self.min_size);
    }

    fn get_size(&self) -> usize {
        self.current_size
    }
}

/// Manages batch processing
pub struct BatchProcessor {
    runtime: Arc<ModelRuntime>,
    metrics: Arc<Mutex<MetricsCollector>>,
    batch_manager: Arc<RwLock<BatchSizeManager>>,
    processing_stats: Arc<RwLock<BatchProcessingStats>>,
}

#[derive(Debug, Default)]
struct BatchProcessingStats {
    total_batches: usize,
    total_sequences: usize,
    total_tokens: usize,
    processing_time: Duration,
    errors: usize,
    last_batch_time: Option<Duration>,
    throughput_history: Vec<f32>, // tokens per second
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(
        runtime: Arc<ModelRuntime>,
        metrics: Arc<Mutex<MetricsCollector>>,
        min_batch_size: usize,
        max_batch_size: usize,
    ) -> Self {
        Self {
            runtime,
            metrics,
            batch_manager: Arc::new(RwLock::new(BatchSizeManager::new(
                min_batch_size,
                max_batch_size,
            ))),
            processing_stats: Arc::new(RwLock::new(BatchProcessingStats::default())),
        }
    }

    /// Process a batch of inputs
    pub async fn process_batch(&self, inputs: Vec<String>) -> Result<Vec<ProcessingOutput>> {
        let start_time = std::time::Instant::now();
        let batch_size = self.batch_manager.read().await.get_size();

        let mut results = Vec::with_capacity(inputs.len());
        
        // Process in chunks according to batch size
        for chunk in inputs.chunks(batch_size) {
            match self.process_chunk(chunk.to_vec()).await {
                Ok(chunk_results) => {
                    self.batch_manager.write().await.handle_success();
                    results.extend(chunk_results);
                }
                Err(e) => {
                    self.batch_manager.write().await.handle_failure();
                    
                    // If this was an OOM error, retry with smaller batch
                    if let EngineError::ResourceError { .. } = e {
                        debug!("Retrying chunk with reduced batch size due to resource error");
                        // Retry the chunk with the new smaller batch size
                        let new_batch_size = self.batch_manager.read().await.get_size();
                        for sub_chunk in chunk.chunks(new_batch_size) {
                            let sub_results = self.process_chunk(sub_chunk.to_vec()).await?;
                            results.extend(sub_results);
                        }
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        // Update processing statistics
        let processing_time = start_time.elapsed();
        let total_tokens: usize = results.iter()
            .map(|r| r.tokens.len())
            .sum();
            
        let throughput = total_tokens as f32 / processing_time.as_secs_f32();
        
        let mut stats = self.processing_stats.write().await;
        stats.total_batches += 1;
        stats.total_sequences += inputs.len();
        stats.total_tokens += total_tokens;
        stats.processing_time += processing_time;
        stats.last_batch_time = Some(processing_time);
        stats.throughput_history.push(throughput);
        
        // Keep throughput history bounded
        if stats.throughput_history.len() > 100 {
            stats.throughput_history.remove(0);
        }

        // Update metrics
        let mut metrics = self.metrics.lock().await;
        metrics.record_batch_processing(
            inputs.len(),
            total_tokens,
            processing_time,
        ).await;

        Ok(results)
    }

    /// Process a single chunk
    async fn process_chunk(&self, inputs: Vec<String>) -> Result<Vec<ProcessingOutput>> {
        let results = self.runtime.process_batch(inputs).await?;
        
        if results.is_empty() {
            return Err(EngineError::ProcessingError {
                message: "Empty results from model".to_string(),
                source: None,
            });
        }

        Ok(results)
    }

    /// Get current batch size
    pub async fn get_batch_size(&self) -> usize {
        self.batch_manager.read().await.get_size()
    }

    /// Get processing throughput statistics
    pub async fn get_throughput_stats(&self) -> ThroughputStats {
        let stats = self.processing_stats.read().await;
        
        let avg_throughput = if !stats.throughput_history.is_empty() {
            stats.throughput_history.iter().sum::<f32>() / stats.throughput_history.len() as f32
        } else {
            0.0
        };

        // Calculate percentile throughputs
        let mut sorted_throughput = stats.throughput_history.clone();
        sorted_throughput.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p50_idx = (sorted_throughput.len() as f32 * 0.5) as usize;
        let p95_idx = (sorted_throughput.len() as f32 * 0.95) as usize;
        
        ThroughputStats {
            average: avg_throughput,
            p50: sorted_throughput.get(p50_idx).copied().unwrap_or(0.0),
            p95: sorted_throughput.get(p95_idx).copied().unwrap_or(0.0),
            samples: stats.throughput_history.len(),
        }
    }
}

/// Throughput statistics
#[derive(Debug, Clone)]
pub struct ThroughputStats {
    pub average: f32,
    pub p50: f32,
    pub p95: f32,
    pub samples: usize,
}

#[async_trait::async_trait]
impl Processor for BatchProcessor {
    async fn shutdown(&self) -> Result<()> {
        Ok(()) // Nothing to clean up
    }

    async fn get_stats(&self) -> ProcessingStats {
        let stats = self.processing_stats.read().await;
        let batch_size = self.batch_manager.read().await.get_size();
        
        ProcessingStats {
            active_streams: 0, // Batch processor doesn't use streams
            total_processed_tokens: stats.total_tokens,
            uptime: stats.processing_time,
            current_batch_size: Some(batch_size),
            average_latency: if stats.total_batches > 0 {
                Some(stats.processing_time / stats.total_batches as u32)
            } else {
                None
            },
            throughput: if stats.processing_time.as_secs_f32() > 0.0 {
                Some(stats.total_tokens as f32 / stats.processing_time.as_secs_f32())
            } else {
                None
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct TestRuntime {
        processed: Arc<Mutex<usize>>,
    }

    impl TestRuntime {
        fn new() -> Self {
            Self {
                processed: Arc::new(Mutex::new(0)),
            }
        }
    }

    #[async_trait::async_trait]
    impl ModelRuntime for TestRuntime {
        async fn process_batch(&self, inputs: Vec<String>) -> Result<Vec<ProcessingOutput>> {
            let mut processed = self.processed.lock().await;
            *processed += inputs.len();

            Ok(inputs.into_iter()
                .map(|input| ProcessingOutput {
                    text: format!("Processed: {}", input),
                    tokens: vec!["test".to_string()],
                    processing_time: Duration::from_millis(100),
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn test_batch_processing() -> Result<()> {
        let runtime = Arc::new(TestRuntime::new());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(
            Arc::new(crate::config::EngineConfig::default())
        )));
        
        let processor = BatchProcessor::new(runtime.clone(), metrics, 2, 8);

        let inputs = vec![
            "test1".to_string(),
            "test2".to_string(),
            "test3".to_string(),
            "test4".to_string(),
            "test5".to_string(),
        ];

        let results = processor.process_batch(inputs).await?;
        assert_eq!(results.len(), 5);

        let processed = *runtime.processed.lock().await;
        assert_eq!(processed, 5);

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_size_adaptation() -> Result<()> {
        let runtime = Arc::new(TestRuntime::new());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(
            Arc::new(crate::config::EngineConfig::default())
        )));
        
        let processor = BatchProcessor::new(runtime, metrics, 2, 8);
        let initial_size = processor.get_batch_size().await;

        // Process several successful batches
        for _ in 0..5 {
            let inputs = vec!["test".to_string(); 4];
            processor.process_batch(inputs).await?;
        }

        // Batch size should have increased
        let new_size = processor.get_batch_size().await;
        assert!(new_size > initial_size);

        Ok(())
    }

    #[tokio::test]
    async fn test_throughput_stats() -> Result<()> {
        let runtime = Arc::new(TestRuntime::new());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(
            Arc::new(crate::config::EngineConfig::default())
        )));
        
        let processor = BatchProcessor::new(runtime, metrics, 2, 8);

        // Process some batches
        for _ in 0..10 {
            let inputs = vec!["test".to_string(); 4];
            processor.process_batch(inputs).await?;
        }

        let stats = processor.get_throughput_stats().await;
        assert!(stats.average > 0.0);
        assert!(stats.p50 > 0.0);
        assert!(stats.p95 > 0.0);
        assert_eq!(stats.samples, 10);

        Ok(())
    }
}