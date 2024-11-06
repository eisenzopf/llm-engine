use std::path::PathBuf;
use std::time::Duration;
use serde::{Serialize, Deserialize};

/// Processing modes supported by the engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingMode {
    /// Stream tokens as they're generated
    Streaming,
    /// Process inputs in batches
    Batch,
    /// Queue inputs for processing
    Queue,
}

/// Complete configuration for the LLM Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    // Model settings
    pub model: ModelConfig,
    
    // Processing settings
    pub processing: ProcessingConfig,
    
    // GPU settings
    pub gpu: GpuConfig,
    
    // Performance monitoring
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the model files
    pub model_path: PathBuf,
    
    /// Maximum sequence length for input
    pub max_sequence_length: usize,
    
    /// Model type and parameters
    pub parameters: ModelParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Temperature for sampling
    pub temperature: f32,
    
    /// Top-p sampling parameter
    pub top_p: f32,
    
    /// Maximum tokens to generate
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Processing mode
    pub mode: ProcessingMode,
    
    /// Batch size (for batch mode)
    pub batch_size: Option<usize>,
    
    /// Queue size (for queue mode)
    pub queue_size: Option<usize>,
    
    /// Whether to automatically adjust batch size
    pub auto_adjust_batch: bool,
    
    /// Operation timeout
    pub timeout: Duration,
    
    /// Number of concurrent operations
    pub concurrency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Specific GPU devices to use
    pub device_ids: Option<Vec<usize>>,
    
    /// Memory threshold for batch size adjustment
    pub memory_threshold: f32,
    
    /// Minimum free memory to maintain (in GB)
    pub min_free_memory_gb: f32,
    
    /// Whether to use flash attention
    pub use_flash_attention: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Whether to collect metrics
    pub enable_metrics: bool,
    
    /// Metrics collection interval
    pub metrics_interval: Duration,
    
    /// Log level for the engine
    pub log_level: LogLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                model_path: PathBuf::new(),
                max_sequence_length: 2048,
                parameters: ModelParameters {
                    temperature: 0.7,
                    top_p: 0.95,
                    max_tokens: 100,
                },
            },
            processing: ProcessingConfig {
                mode: ProcessingMode::Streaming,
                batch_size: Some(32),
                queue_size: Some(1000),
                auto_adjust_batch: true,
                timeout: Duration::from_secs(30),
                concurrency: 1,
            },
            gpu: GpuConfig {
                device_ids: None,
                memory_threshold: 0.9,
                min_free_memory_gb: 2.0,
                use_flash_attention: true,
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                metrics_interval: Duration::from_secs(1),
                log_level: LogLevel::Info,
            },
        }
    }
}

impl EngineConfig {
    pub fn validate(&self) -> crate::Result<()> {
        // Validate model configuration
        if self.model.model_path.as_os_str().is_empty() {
            return Err(crate::EngineError::ConfigurationError {
                message: "Model path cannot be empty".to_string(),
                parameter: "model_path".to_string(),
            });
        }

        // Validate processing configuration
        match self.processing.mode {
            ProcessingMode::Batch => {
                if self.processing.batch_size.is_none() {
                    return Err(crate::EngineError::ConfigurationError {
                        message: "Batch size must be set for batch mode".to_string(),
                        parameter: "batch_size".to_string(),
                    });
                }
            }
            ProcessingMode::Queue => {
                if self.processing.queue_size.is_none() {
                    return Err(crate::EngineError::ConfigurationError {
                        message: "Queue size must be set for queue mode".to_string(),
                        parameter: "queue_size".to_string(),
                    });
                }
            }
            _ => {}
        }

        // Validate GPU configuration
        if let Some(ids) = &self.gpu.device_ids {
            if ids.is_empty() {
                return Err(crate::EngineError::ConfigurationError {
                    message: "Device IDs cannot be empty when specified".to_string(),
                    parameter: "device_ids".to_string(),
                });
            }
        }

        if !(0.0..=1.0).contains(&self.gpu.memory_threshold) {
            return Err(crate::EngineError::ConfigurationError {
                message: "Memory threshold must be between 0 and 1".to_string(),
                parameter: "memory_threshold".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EngineConfig::default();
        assert_eq!(config.processing.mode, ProcessingMode::Streaming);
        assert!(config.processing.auto_adjust_batch);
    }

    #[test]
    fn test_config_validation() {
        let mut config = EngineConfig::default();
        config.model.model_path = PathBuf::from("/path/to/model");
        assert!(config.validate().is_ok());

        // Test invalid memory threshold
        config.gpu.memory_threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_batch_mode_validation() {
        let mut config = EngineConfig::default();
        config.model.model_path = PathBuf::from("/path/to/model");
        config.processing.mode = ProcessingMode::Batch;
        config.processing.batch_size = None;
        assert!(config.validate().is_err());
    }
}