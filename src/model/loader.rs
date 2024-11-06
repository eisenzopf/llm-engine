use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, Llama, Cache};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;
use crate::error::EngineError;
use crate::config::EngineConfig;

pub struct ModelLoader {
    config: Arc<EngineConfig>,
    max_concurrent_loads: usize,
    semaphore: Arc<Semaphore>,
}

impl ModelLoader {
    pub fn new(config: Arc<EngineConfig>) -> Self {
        let max_concurrent_loads = config.gpu.device_ids.as_ref()
            .map(|ids| ids.len())
            .unwrap_or(1);
            
        Self {
            config,
            max_concurrent_loads,
            semaphore: Arc::new(Semaphore::new(max_concurrent_loads)),
        }
    }

    pub async fn load_models(
        &self,
        model_paths: Vec<Vec<PathBuf>>,
        devices: Vec<Device>,
    ) -> Result<Vec<Arc<Llama>>> {
        let mut handles = Vec::new();
        
        for (device_id, (paths, device)) in model_paths.iter()
            .zip(devices.iter())
            .enumerate() 
        {
            let paths = paths.clone();
            let device = device.clone();
            let semaphore = Arc::clone(&self.semaphore);
            let config = self.config.clone();

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                
                // Load the model configuration
                let llama_config = config.model.to_llama_config(
                    config.gpu.flash_attention
                );

                // Create the variable builder with proper dtype
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &paths,
                        DType::BF16,
                        &device
                    )?
                };

                // Initialize model
                let model = Llama::load(vb, &llama_config)
                    .map_err(|e| EngineError::ModelError {
                        message: format!("Failed to load model on device {}: {}", device_id, e),
                        source: Some(Box::new(e)),
                    })?;

                Ok::<Arc<Llama>, EngineError>(Arc::new(model))
            });

            handles.push(handle);
        }

        let mut models = Vec::new();
        for handle in handles {
            match handle.await? {
                Ok(model) => models.push(model),
                Err(e) => {
                    // Clean up already loaded models
                    models.clear();
                    return Err(e.into());
                }
            }
        }

        if models.is_empty() {
            return Err(EngineError::InitializationError {
                message: "No models were successfully loaded".to_string(),
                source: None,
            }.into());
        }

        Ok(models)
    }

    pub fn create_cache(&self, device: &Device) -> Result<Cache> {
        Cache::new(
            self.config.gpu.flash_attention,
            DType::BF16,
            &self.config.model.to_llama_config(self.config.gpu.flash_attention),
            device,
        ).map_err(|e| EngineError::InitializationError {
            message: format!("Failed to create cache: {}", e),
            source: Some(Box::new(e)),
        }.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[tokio::test]
    async fn test_model_loading() -> Result<()> {
        let config = Arc::new(EngineConfig::default());
        let loader = ModelLoader::new(config);

        let device = Device::Cpu;
        let model_path = Path::new("test_models/meta-llama/Llama-3.1-8B-Instruct");
        
        let paths = vec![
            model_path.join("model-00001-of-00003.safetensors"),
            model_path.join("model-00002-of-00003.safetensors"),
            model_path.join("model-00003-of-00003.safetensors"),
        ];

        let models = loader.load_models(
            vec![paths],
            vec![device]
        ).await?;

        assert_eq!(models.len(), 1);
        Ok(())
    }
}