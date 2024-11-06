use std::sync::Arc;
use tokio::sync::Mutex;
use crate::{
    config::{EngineConfig, ProcessingMode},
    error::{EngineError, Result},
    gpu::GpuManager,
    model::ModelManager,
    metrics::{MetricsCollector, MetricsSnapshot}
    types::{ProcessingOutput, StreamHandle, QueueHandle},
};

/// Main entry point for the LLM Engine library
pub struct LLMEngine {
    config: Arc<EngineConfig>,
    gpu_manager: Arc<GpuManager>,
    model_manager: Arc<ModelManager>,
    metrics: Arc<Mutex<MetricsCollector>>,
}

/// Builder for constructing an LLMEngine instance
pub struct EngineBuilder {
    config: EngineConfig,
}

impl EngineBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: EngineConfig::default(),
        }
    }

    /// Set the engine configuration
    pub fn with_config(mut self, config: EngineConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the processing mode
    pub fn with_mode(mut self, mode: ProcessingMode) -> Self {
        self.config.processing.mode = mode;
        self
    }

    /// Build the engine instance
    pub async fn build(self) -> Result<LLMEngine> {
        // Validate configuration
        self.config.validate()?;

        // Initialize components
        let config = Arc::new(self.config);
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(config.clone())));
        
        // Initialize GPU manager
        let gpu_manager = GpuManager::new(config.clone(), metrics.clone()).await
            .map_err(|e| EngineError::InitializationError {
                message: "Failed to initialize GPU manager".to_string(),
                source: Some(Box::new(e)),
            })?;
        let gpu_manager = Arc::new(gpu_manager);

        // Initialize model manager
        let model_manager = ModelManager::new(config.clone(), gpu_manager.clone()).await
            .map_err(|e| EngineError::InitializationError {
                message: "Failed to initialize model manager".to_string(),
                source: Some(Box::new(e)),
            })?;
        let model_manager = Arc::new(model_manager);

        Ok(LLMEngine {
            config,
            gpu_manager,
            model_manager,
            metrics,
        })
    }
}

impl LLMEngine {
    /// Create a new engine builder
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    /// Get information about the engine's capabilities and state
    pub fn info(&self) -> EngineInfo {
        EngineInfo {
            available_gpus: self.gpu_manager.available_gpus(),
            processing_mode: self.config.processing.mode,
            model_path: self.config.model.model_path.clone(),
            max_sequence_length: self.config.model.max_sequence_length,
        }
    }

    /// Get handles for streaming mode processing
    pub async fn get_stream_handles(&self) -> Result<Vec<StreamHandle>> {
        if self.config.processing.mode != ProcessingMode::Streaming {
            return Err(EngineError::ConfigurationError {
                message: "Engine not configured for streaming mode".to_string(),
                parameter: "mode".to_string(),
            });
        }

        self.model_manager.create_stream_handles().await
    }

    /// Process a batch of inputs
    pub async fn process_batch(&self, inputs: Vec<String>) -> Result<Vec<ProcessingOutput>> {
        if self.config.processing.mode != ProcessingMode::Batch {
            return Err(EngineError::ConfigurationError {
                message: "Engine not configured for batch mode".to_string(),
                parameter: "mode".to_string(),
            });
        }

        self.model_manager.process_batch(inputs).await
    }

    /// Enqueue an input for processing
    pub async fn enqueue(&self, input: String) -> Result<QueueHandle> {
        if self.config.processing.mode != ProcessingMode::Queue {
            return Err(EngineError::ConfigurationError {
                message: "Engine not configured for queue mode".to_string(),
                parameter: "mode".to_string(),
            });
        }

        self.model_manager.enqueue(input).await
    }

    /// Get the current metrics
    pub async fn get_metrics(&self) -> Result<MetricsSnapshot> {
        let metrics = self.metrics.lock().await;
        metrics.snapshot()
    }

    /// Shutdown the engine and release resources
    pub async fn shutdown(self) -> Result<()> {
        self.model_manager.shutdown().await?;
        self.gpu_manager.shutdown().await?;
        Ok(())
    }
}

/// Information about the engine's capabilities and state
#[derive(Debug, Clone)]
pub struct EngineInfo {
    pub available_gpus: usize,
    pub processing_mode: ProcessingMode,
    pub model_path: std::path::PathBuf,
    pub max_sequence_length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    async fn create_test_engine() -> Result<LLMEngine> {
        let mut config = EngineConfig::default();
        config.model.model_path = PathBuf::from("/path/to/test/model");
        
        LLMEngine::builder()
            .with_config(config)
            .build()
            .await
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = create_test_engine().await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_processing_mode_validation() {
        let engine = create_test_engine().await.unwrap();
        
        // Test streaming mode
        assert!(engine.process_batch(vec!["test".to_string()]).await.is_err());
        
        // Test mode-specific operations
        match engine.config.processing.mode {
            ProcessingMode::Streaming => {
                assert!(engine.get_stream_handles().await.is_ok());
            }
            ProcessingMode::Batch => {
                assert!(engine.process_batch(vec!["test".to_string()]).await.is_ok());
            }
            ProcessingMode::Queue => {
                assert!(engine.enqueue("test".to_string()).await.is_ok());
            }
        }
    }

    #[tokio::test]
    async fn test_engine_shutdown() {
        let engine = create_test_engine().await.unwrap();
        assert!(engine.shutdown().await.is_ok());
    }
}