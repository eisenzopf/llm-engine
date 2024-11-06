// Location: src/processing/batch.rs

use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, warn};
use std::time::Duration;

use crate::{
    config::EngineConfig,
    error::{EngineError, Result},
    model::{ModelRuntime, LlamaTokenizer},
    metrics::MetricsCollector,
    types::ProcessingOutput,
};

/// Manages batch processing
pub struct BatchProcessor {
    runtime: Arc<ModelRuntime>,
    tokenizer: Arc<LlamaTokenizer>,
    metrics: Arc<Mutex<MetricsCollector>>,
    batch_manager: Arc<RwLock<BatchSizeManager>>,
    processing_stats: Arc<RwLock<BatchProcessingStats>>,
    config: Arc<EngineConfig>,
}

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
    last_gpu_utilization: f32,
}

#[derive(Debug, Default)]
struct BatchProcessingStats {
    total_batches: usize,
    total_sequences: usize,
    total_tokens: usize,
    processing_time: Duration,
    errors: usize,
    last_batch_time: Option<Duration>,
    throughput_history: Vec<ThroughputRecord>,
}

#[derive(Debug)]
struct ThroughputRecord {
    batch_size: usize,
    tokens_per_second: f32,
    gpu_utilization: f32,
    timestamp: std::time::Instant,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(
        runtime: Arc<ModelRuntime>,
        tokenizer: Arc<LlamaTokenizer>,
        metrics: Arc<Mutex<MetricsCollector>>,
        config: Arc<EngineConfig>,
    ) -> Self {
        let batch_manager = BatchSizeManager {
            current_size: config.processing.batch_size.unwrap_or(1),
            min_size: 1,
            max_size: config.processing.batch_size.unwrap_or(32),
            growth_factor: 1.2,
            reduction_factor: 0.5,
            success_streak: 0,
            failure_streak: 0,
            last_gpu_utilization: 0.0,
        };

        Self {
            runtime,
            tokenizer,
            metrics,
            batch_manager: Arc::new(RwLock::new(batch_manager)),
            processing_stats: Arc::new(RwLock::new(BatchProcessingStats::default())),
            config,
        }
    }

    /// Process a batch of inputs
    pub async fn process_batch(&self, inputs: Vec<String>) -> Result<Vec<ProcessingOutput>> {
        let start_time = std::time::Instant::now();
        let batch_size = self.batch_manager.read().await.current_size;

        let mut results = Vec::with_capacity(inputs.len());
        let input_count = inputs.len();

        // Process in chunks according to batch size
        for chunk in inputs.chunks(batch_size) {
            match self.process_chunk(chunk).await {
                Ok(chunk_results) => {
                    self.handle_success(chunk_results.len()).await?;
                    results.extend(chunk_results);
                }
                Err(e) => {
                    self.handle_failure(&e).await?;
                    
                    // If this was an OOM error, retry with smaller batch
                    if matches!(e, EngineError::ResourceError { .. }) {
                        debug!("Retrying chunk with reduced batch size due to resource error");
                        let new_batch_size = self.batch_manager.read().await.current_size;
                        for sub_chunk in chunk.chunks(new_batch_size) {
                            let sub_results = self.process_chunk(sub_chunk).await?;
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
        let gpu_utilization = self.runtime.get_gpu_utilization().await?;
        
        let mut stats = self.processing_stats.write().await;
        stats.total_batches += 1;
        stats.total_sequences += input_count;
        stats.total_tokens += total_tokens;
        stats.processing_time += processing_time;
        stats.last_batch_time = Some(processing_time);
        
        stats.throughput_history.push(ThroughputRecord {
            batch_size: results.len(),
            tokens_per_second: throughput,
            gpu_utilization,
            timestamp: std::time::Instant::now(),
        });
        
        // Keep throughput history bounded
        if stats.throughput_history.len() > 100 {
            stats.throughput_history.remove(0);
        }

        // Update metrics
        let mut metrics = self.metrics.lock().await;
        metrics.record_batch_processing(
            input_count,
            total_tokens,
            processing_time,
            gpu_utilization,
        ).await;

        Ok(results)
    }

    /// Process a single chunk
    async fn process_chunk(&self, inputs: &[String]) -> Result<Vec<ProcessingOutput>> {
        // Add instruction template if configured
        let processed_inputs: Vec<String> = inputs.iter().map(|input| {
            if !input.starts_with("### Instruction:") {
                self.config.generation.instruction_template.replace("{}", input)
            } else {
                input.clone()
            }
        }).collect();

        // Tokenize inputs
        let encoding_result = self.tokenizer.encode_batch(
            &processed_inputs,
            true,  // Add special tokens
            true,  // Add padding
        ).await?;

        // Process through model
        self.runtime.process_batch_with_tokens(
            encoding_result.token_ids,
            encoding_result.attention_masks,
            encoding_result.sequence_lengths,
        ).await
    }

    async fn handle_success(&self, processed_count: usize) -> Result<()> {
        let mut manager = self.batch_manager.write().await;
        manager.success_streak += 1;
        manager.failure_streak = 0;

        // Consider increasing batch size after consistent successes
        if manager.success_streak >= 3 {
            let gpu_utilization = self.runtime.get_gpu_utilization().await?;
            
            if gpu_utilization < 0.8 && gpu_utilization > manager.last_gpu_utilization {
                let new_size = (manager.current_size as f32 * manager.growth_factor) as usize;
                manager.current_size = new_size.min(manager.max_size);
            }
            
            manager.last_gpu_utilization = gpu_utilization;
            manager.success_streak = 0;
        }

        Ok(())
    }

    async fn handle_failure(&self, error: &EngineError) -> Result<()> {
        let mut manager = self.batch_manager.write().await;
        manager.failure_streak += 1;
        manager.success_streak = 0;

        // Reduce batch size on failure
        manager.current_size = (manager.current_size as f32 * manager.reduction_factor) as usize;
        manager.current_size = manager.current_size.max(manager.min_size);

        // Update error statistics
        let mut stats = self.processing_stats.write().await;
        stats.errors += 1;

        Ok(())
    }

    /// Get current batch size
    pub async fn get_batch_size(&self) -> usize {
        self.batch_manager.read().await.current_size
    }

    /// Get processing throughput statistics
    pub async fn get_throughput_stats(&self) -> ThroughputStats {
        let stats = self.processing_stats.read().await;
        
        let records = &stats.throughput_history;
        let avg_throughput = if !records.is_empty() {
            records.iter().map(|r| r.tokens_per_second).sum::<f32>() / records.len() as f32
        } else {
            0.0
        };

        // Calculate percentile throughputs
        let mut sorted_throughput: Vec<_> = records.iter()
            .map(|r| r.tokens_per_second)
            .collect();
        sorted_throughput.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p50_idx = (sorted_throughput.len() as f32 * 0.5) as usize;
        let p95_idx = (sorted_throughput.len() as f32 * 0.95) as usize;
        
        let avg_gpu_util = if !records.is_empty() {
            records.iter().map(|r| r.gpu_utilization).sum::<f32>() / records.len() as f32
        } else {
            0.0
        };

        ThroughputStats {
            average_throughput: avg_throughput,
            p50_throughput: sorted_throughput.get(p50_idx).copied().unwrap_or(0.0),
            p95_throughput: sorted_throughput.get(p95_idx).copied().unwrap_or(0.0),
            average_gpu_utilization: avg_gpu_util,
            samples: records.len(),
        }
    }

    /// Get current processing statistics
    pub async fn get_stats(&self) -> BatchStats {
        let stats = self.processing_stats.read().await;
        let manager = self.batch_manager.read().await;

        BatchStats {
            total_batches: stats.total_batches,
            total_sequences: stats.total_sequences,
            total_tokens: stats.total_tokens,
            total_errors: stats.errors,
            current_batch_size: manager.current_size,
            uptime: stats.processing_time,
            last_batch_time: stats.last_batch_time,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThroughputStats {
    pub average_throughput: f32,
    pub p50_throughput: f32,
    pub p95_throughput: f32,
    pub average_gpu_utilization: f32,
    pub samples: usize,
}

#[derive(Debug, Clone)]
pub struct BatchStats {
    pub total_batches: usize,
    pub total_sequences: usize,
    pub total_tokens: usize,
    pub total_errors: usize,
    pub current_batch_size: usize,
    pub uptime: Duration,
    pub last_batch_time: Option<Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_processor() -> Result<(BatchProcessor, Arc<TestRuntime>)> {
        let runtime = Arc::new(TestRuntime::default());
        let tokenizer = Arc::new(LlamaTokenizer::from_file("models/tokenizer.json").await?);
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(Arc::new(EngineConfig::default()))));
        let config = Arc::new(EngineConfig::default());
        
        let processor = BatchProcessor::new(
            runtime.clone(),
            tokenizer,
            metrics,
            config,
        );

        Ok((processor, runtime))
    }

    #[derive(Default)]
    struct TestRuntime {
        processed: Arc<Mutex<usize>>,
    }

    #[async_trait::async_trait]
    impl ModelRuntime for TestRuntime {
        async fn process_batch_with_tokens(
            &self,
            token_ids: Vec<Vec<u32>>,
            _attention_masks: Vec<Vec<u8>>,
            _sequence_lengths: Vec<usize>,
        ) -> Result<Vec<ProcessingOutput>> {
            let mut processed = self.processed.lock().await;
            *processed += token_ids.len();
            
            Ok(token_ids.iter().map(|tokens| ProcessingOutput {
                text: "Test output".to_string(),
                tokens: tokens.clone(),
                processing_time: Duration::from_millis(100),
            }).collect())
        }

        async fn get_gpu_utilization(&self) -> Result<f32> {
            Ok(0.5)
        }
    }

    #[tokio::test]
    async fn test_batch_processing() -> Result<()> {
        let (processor, runtime) = create_test_processor().await?;
        
        let inputs = vec![
            "Test 1".to_string(),
            "Test 2".to_string(),
            "Test 3".to_string(),
        ];
        
        let results = processor.process_batch(inputs).await?;
        assert_eq!(results.len(), 3);
        
        let processed = runtime.processed.lock().await;
        assert_eq!(*processed, 3);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_size_adaptation() -> Result<()> {
        let (processor, _) = create_test_processor().await?;
        let initial_size = processor.get_batch_size().await;
        
        // Process several successful batches
        for _ in 0..5 {
            let inputs = vec!["test".to_string(); 4];
            processor.process_batch(inputs).await?;
        }
        
        let new_size = processor.get_batch_size().await;
        assert!(new_size >= initial_size);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_throughput_stats() -> Result<()> {
        let (processor, _) = create_test_processor().await?;
        
        // Process some batches
        for _ in 0..10 {
            let inputs = vec!["test".to_string(); 4];
            processor.process_batch(inputs).await?;
        }
        
        let stats = processor.get_throughput_stats().await;
        assert!(stats.average_throughput > 0.0);
        assert!(stats.p50_throughput > 0.0);
        assert!(stats.average_gpu_utilization > 0.0);
        
        Ok(())
    }
}