use std::sync::Arc;
use crate::{
    config::EngineConfig,
    error::{EngineError, Result},
    gpu::GpuManager,
    model::ModelManager,
    metrics::MetricsCollector,
};

use super::engine::LLMEngine;

/// Builder for constructing an LLMEngine instance
pub struct EngineBuilder {
    config: Option<EngineConfig>,
}

impl EngineBuilder {
    /// Create a new builder instance
    pub fn new() -> Self {
        Self {
            config: None,
        }
    }

    /// Set the engine configuration
    pub fn with_config(mut self, config: EngineConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the engine instance
    pub async fn build(self) -> Result<LLMEngine> {
        // Get configuration or use default
        let config = self.config.unwrap_or_default();
        
        // Validate configuration
        config.validate()?;

        // Create shared components
        let config = Arc::new(config);
        let metrics = Arc::new(tokio::sync::Mutex::new(
            MetricsCollector::new(config.clone())
        ));

        // Initialize GPU manager
        let gpu_manager = GpuManager::new(
            config.clone(),
            metrics.clone(),
        ).await?;
        let gpu_manager = Arc::new(gpu_manager);

        // Initialize model manager
        let model_manager = ModelManager::new(
            config.clone(),
            gpu_manager.clone(),
        ).await?;
        let model_manager = Arc::new(model_manager);

        // Create engine instance
        Ok(LLMEngine {
            config,
            gpu_manager,
            model_manager,
            metrics,
        })
    }
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_builder_default() {
        let builder = EngineBuilder::default();
        let config = EngineConfig::default();
        config.model.model_path = PathBuf::from("/path/to/test/model");
        let engine = builder.with_config(config).build().await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_builder_validation() {
        let builder = EngineBuilder::default();
        // Empty config should fail validation
        let result = builder.build().await;
        assert!(result.is_err());
    }
}