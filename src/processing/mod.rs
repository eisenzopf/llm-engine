// Location: src/processing/mod.rs

use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::Result;

mod stream;
mod batch;
mod queue;
mod common;

pub use stream::{StreamProcessor, StreamHandle, StreamStats};
pub use batch::{BatchProcessor, BatchStats, ThroughputStats};
pub use queue::{QueueProcessor, QueueHandle, Priority, QueueStatsSnapshot};
pub use common::{ProcessingStats, MemoryStats, TokenSequence, BatchTensors};

use crate::{
    config::{EngineConfig, ProcessingMode},
    error::EngineError,
    model::{ModelRuntime, LlamaTokenizer},
    metrics::MetricsCollector,
    types::ProcessingOutput,
};

/// Main processor interface
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    /// Process a single input
    async fn process(&self, input: String) -> Result<ProcessingOutput>;
    
    /// Process multiple inputs
    async fn process_many(&self, inputs: Vec<String>) -> Result<Vec<ProcessingOutput>>;
    
    /// Get current processing statistics
    async fn get_stats(&self) -> ProcessingStats;
    
    /// Shutdown the processor
    async fn shutdown(&self) -> Result<()>;
}

/// Processing engine that manages different processing modes
pub struct ProcessingEngine {
    /// Current processing mode
    mode: ProcessingMode,
    
    /// Stream processor
    stream_processor: Option<StreamProcessor>,
    
    /// Batch processor
    batch_processor: Option<BatchProcessor>,
    
    /// Queue processor
    queue_processor: Option<QueueProcessor>,
    
    /// Shared configuration
    config: Arc<EngineConfig>,
    
    /// Shared metrics collector
    metrics: Arc<Mutex<MetricsCollector>>,
    
    /// Model runtime
    runtime: Arc<ModelRuntime>,
    
    /// Tokenizer
    tokenizer: Arc<LlamaTokenizer>,
}

impl ProcessingEngine {
    /// Create a new processing engine
    pub fn new(
        config: Arc<EngineConfig>,
        runtime: Arc<ModelRuntime>,
        tokenizer: Arc<LlamaTokenizer>,
        metrics: Arc<Mutex<MetricsCollector>>,
    ) -> Self {
        let mode = config.processing.mode;
        let mut engine = Self {
            mode,
            stream_processor: None,
            batch_processor: None,
            queue_processor: None,
            config,
            metrics,
            runtime,
            tokenizer,
            
        };
        
        // Initialize appropriate processor
        engine.init_processor();
        engine
    }

    /// Initialize processor based on mode
    fn init_processor(&mut self) {
        match self.mode {
            ProcessingMode::Streaming => {
                self.stream_processor = Some(StreamProcessor::new(
                    Arc::clone(&self.runtime),
                    Arc::clone(&self.tokenizer),
                    Arc::clone(&self.metrics),
                    Arc::clone(&self.config),
                ));
            }
            ProcessingMode::Batch => {
                self.batch_processor = Some(BatchProcessor::new(
                    Arc::clone(&self.runtime),
                    Arc::clone(&self.tokenizer),
                    Arc::clone(&self.metrics),
                    Arc::clone(&self.config),
                ));
            }
            ProcessingMode::Queue => {
                self.queue_processor = Some(QueueProcessor::new(
                    Arc::clone(&self.runtime),
                    Arc::clone(&self.tokenizer),
                    Arc::clone(&self.metrics),
                    Arc::clone(&self.config),
                ));
            }
        }
    }

    /// Switch processing mode
    pub async fn switch_mode(&mut self, mode: ProcessingMode) -> Result<()> {
        if mode == self.mode {
            return Ok(());
        }

        // Shut down current processor
        self.shutdown().await?;

        // Switch mode and initialize new processor
        self.mode = mode;
        self.init_processor();

        Ok(())
    }

    /// Get current processing mode
    pub fn current_mode(&self) -> ProcessingMode {
        self.mode
    }

    /// Create a new stream handle
    pub async fn create_stream(&self) -> Result<StreamHandle> {
        self.stream_processor
            .as_ref()
            .ok_or_else(|| EngineError::InvalidMode {
                expected: ProcessingMode::Streaming,
                actual: self.mode,
            })?
            .create_handle()
            .await
    }

    /// Process a batch of inputs
    pub async fn process_batch(&self, inputs: Vec<String>) -> Result<Vec<ProcessingOutput>> {
        self.batch_processor
            .as_ref()
            .ok_or_else(|| EngineError::InvalidMode {
                expected: ProcessingMode::Batch,
                actual: self.mode,
            })?
            .process_batch(inputs)
            .await
    }

    /// Enqueue an input for processing
    pub async fn enqueue(
        &self,
        input: String,
        priority: Priority,
    ) -> Result<QueueHandle> {
        self.queue_processor
            .as_ref()
            .ok_or_else(|| EngineError::InvalidMode {
                expected: ProcessingMode::Queue,
                actual: self.mode,
            })?
            .enqueue_with_priority(input, priority)
            .await
    }

    /// Get comprehensive processing statistics
    pub async fn get_stats(&self) -> ProcessingStats {
        match self.mode {
            ProcessingMode::Streaming => {
                if let Some(processor) = &self.stream_processor {
                    processor.get_stats().await
                } else {
                    ProcessingStats::default()
                }
            }
            ProcessingMode::Batch => {
                if let Some(processor) = &self.batch_processor {
                    processor.get_stats().await
                } else {
                    ProcessingStats::default()
                }
            }
            ProcessingMode::Queue => {
                if let Some(processor) = &self.queue_processor {
                    processor.get_stats().await
                } else {
                    ProcessingStats::default()
                }
            }
        }
    }

    /// Get GPU memory usage statistics
    pub async fn get_memory_stats(&self) -> Result<MemoryStats> {
        let gpu_stats = self.runtime.get_gpu_stats().await?;
        
        Ok(MemoryStats {
            gpu_memory_used: gpu_stats.memory_used,
            peak_gpu_memory: gpu_stats.peak_memory,
            token_cache_size: self.tokenizer.get_cache_size().await,
        })
    }

    /// Shutdown processing engine
    pub async fn shutdown(&mut self) -> Result<()> {
        match self.mode {
            ProcessingMode::Streaming => {
                if let Some(processor) = &self.stream_processor {
                    processor.shutdown().await?;
                }
            }
            ProcessingMode::Batch => {
                if let Some(processor) = &self.batch_processor {
                    processor.shutdown().await?;
                }
            }
            ProcessingMode::Queue => {
                if let Some(processor) = &self.queue_processor {
                    processor.shutdown().await?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn create_test_engine() -> Result<ProcessingEngine> {
        let config = Arc::new(EngineConfig::default());
        let runtime = Arc::new(TestRuntime::default());
        let tokenizer = Arc::new(LlamaTokenizer::from_file("models/tokenizer.json").await?);
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(config.clone())));
        
        Ok(ProcessingEngine::new(
            config,
            runtime,
            tokenizer,
            metrics,
        ))
    }

    #[derive(Default)]
    struct TestRuntime {
        processed: Arc<Mutex<usize>>,
    }

    #[async_trait::async_trait]
    impl ModelRuntime for TestRuntime {
        async fn process(&self, _input: String) -> Result<ProcessingOutput> {
            let mut processed = self.processed.lock().await;
            *processed += 1;
            
            Ok(ProcessingOutput {
                text: "Test output".to_string(),
                tokens: vec![1, 2, 3],
                processing_time: Duration::from_millis(100),
            })
        }

        async fn get_gpu_stats(&self) -> Result<GpuStats> {
            Ok(GpuStats {
                memory_used: 1000,
                peak_memory: 2000,
                utilization: 0.5,
            })
        }
    }

    #[tokio::test]
    async fn test_mode_switching() -> Result<()> {
        let mut engine = create_test_engine().await?;
        assert_eq!(engine.current_mode(), ProcessingMode::Streaming);
        
        engine.switch_mode(ProcessingMode::Batch).await?;
        assert_eq!(engine.current_mode(), ProcessingMode::Batch);
        
        engine.switch_mode(ProcessingMode::Queue).await?;
        assert_eq!(engine.current_mode(), ProcessingMode::Queue);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_streaming_mode() -> Result<()> {
        let engine = create_test_engine().await?;
        let mut handle = engine.create_stream().await?;
        
        let result = handle.process("Test input".to_string()).await?;
        assert!(!result.text.is_empty());
        
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_mode() -> Result<()> {
        let mut engine = create_test_engine().await?;
        engine.switch_mode(ProcessingMode::Batch).await?;
        
        let inputs = vec!["Test 1".to_string(), "Test 2".to_string()];
        let results = engine.process_batch(inputs).await?;
        
        assert_eq!(results.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_queue_mode() -> Result<()> {
        let mut engine = create_test_engine().await?;
        engine.switch_mode(ProcessingMode::Queue).await?;
        
        let handle = engine.enqueue(
            "Test input".to_string(),
            Priority::High
        ).await?;
        
        let result = handle.await?;
        assert!(!result.text.is_empty());
        
        Ok(())
    }

    #[tokio::test]
    async fn test_statistics() -> Result<()> {
        let engine = create_test_engine().await?;
        
        let stats = engine.get_stats().await;
        let memory_stats = engine.get_memory_stats().await?;
        
        assert!(memory_stats.gpu_memory_used > 0);
        assert!(memory_stats.peak_gpu_memory > 0);
        
        Ok(())
    }
}