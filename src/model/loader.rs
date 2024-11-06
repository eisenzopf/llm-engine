use std::sync::Arc;
use anyhow::Result;
use crate::{
    config::EngineConfig,
    error::EngineError,
};

use super::runtime::ModelRuntime;

/// Handles model loading and initialization
pub struct ModelLoader {
    config: Arc<EngineConfig>,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(config: Arc<EngineConfig>) -> Self {
        Self { config }
    }

    /// Load a model for the specified device
    pub async fn load_model(&self, device_id: usize) -> Result<ModelRuntime> {
        let model_path = &self.config.model.model_path;
        
        // Ensure model files exist
        if !model_path.exists() {
            return Err(EngineError::InitializationError {
                message: format!("Model path does not exist: {}", model_path.display()),
                source: None,
            }.into());
        }

        // Load configuration
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            EngineError::InitializationError {
                message: format!("Failed to read config.json: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Parse model configuration
        let model_config: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            EngineError::InitializationError {
                message: format!("Failed to parse config.json: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Initialize candle device
        let device = candle_core::Device::cuda_if_available(device_id).map_err(|e| {
            EngineError::GPUError {
                device_id,
                message: format!("Failed to initialize CUDA device: {}", e),
                recoverable: false,
            }
        })?;

        // Load model weights
        let weights_path = model_path.join("model.safetensors");
        let weights = unsafe {
            candle_core::safetensors::load(&weights_path, &device).map_err(|e| {
                EngineError::InitializationError {
                    message: format!("Failed to load model weights: {}", e),
                    source: Some(Box::new(e)),
                }
            })?
        };

        // Create model runtime
        ModelRuntime::new(
            device_id,
            device,
            weights,
            model_config,
            self.config.clone(),
        )
    }

    /// Check if model files are available
    pub async fn check_model_files(&self) -> Result<bool> {
        let model_path = &self.config.model.model_path;
        
        if !model_path.exists() {
            return Ok(false);
        }

        let required_files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
        ];

        for file in required_files {
            if !model_path.join(file).exists() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Get model metadata
    pub async fn get_model_metadata(&self) -> Result<ModelMetadata> {
        let config_path = self.config.model.model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            EngineError::InitializationError {
                message: format!("Failed to read config.json: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        let