// Location: src/config.rs

use std::path::PathBuf;
use std::time::Duration;
use serde::{Serialize, Deserialize};
use candle_transformers::models::llama::Config as LlamaConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub model: ModelConfig,
    pub processing: ProcessingConfig,
    pub gpu: GpuConfig,
    pub generation: GenerationConfig,
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to model files
    pub model_path: PathBuf,
    
    /// Model type (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    pub model_type: String,
    
    /// Maximum sequence length
    pub max_sequence_length: usize,
    
    /// Model hidden size
    pub hidden_size: usize,
    
    /// Number of attention heads
    pub num_attention_heads: usize,
    
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    
    /// Intermediate size for feed forward layers
    pub intermediate_size: usize,
    
    /// RMS normalization epsilon
    pub rms_norm_eps: f32,
    
    /// Vocabulary size
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingMode {
    Streaming,
    Batch,
    Queue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Processing mode
    pub mode: ProcessingMode,
    
    /// Batch size for batch mode
    pub batch_size: Option<usize>,
    
    /// Queue size for queue mode
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
    pub flash_attention: bool,
    
    /// Maximum memory per GPU (in GB)
    pub max_memory_gb: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_output_tokens: usize,
    
    /// Temperature for sampling
    pub temperature: f64,
    
    /// Top-p sampling threshold
    pub top_p: f64,
    
    /// Top-k sampling
    pub top_k: Option<usize>,
    
    /// Repetition penalty
    pub repetition_penalty: f32,
    
    /// Context size for repetition penalty
    pub repetition_context_size: usize,
    
    /// Instruction template for model input
    pub instruction_template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Whether to collect metrics
    pub enable_metrics: bool,
    
    /// Metrics collection interval
    pub metrics_interval: Duration,
    
    /// Log level
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

impl ModelConfig {
    pub fn to_llama_config(&self, use_flash_attention: bool) -> LlamaConfig {
        LlamaConfig {
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            num_key_value_heads: self.num_attention_heads, // Usually same as attention heads
            rope_theta: 10000.0, // Standard value for Llama models
            use_flash_attn,
            max_position_embeddings: self.max_sequence_length,
            max_sequence_length: Some(self.max_sequence_length),
            eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(2)), // Standard EOS token for Llama
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                model_path: PathBuf::new(),
                model_type: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                max_sequence_length: 4096,
                hidden_size: 4096,
                num_attention_heads: 32,
                num_hidden_layers: 32,
                intermediate_size: 11008,
                rms_norm_eps: 1e-6,
                vocab_size: 32000,
            },
            processing: ProcessingConfig {
                mode: ProcessingMode::Streaming,
                batch_size: Some(1),
                queue_size: Some(1000),
                auto_adjust_batch: true,
                timeout: Duration::from_secs(30),
                concurrency: 1,
            },
            gpu: GpuConfig {
                device_ids: None,
                memory_threshold: 0.9,
                min_free_memory_gb: 2.0,
                flash_attention: true,
                max_memory_gb: None,
            },
            generation: GenerationConfig {
                max_output_tokens: 2048,
                temperature: 0.7,
                top_p: 0.95,
                top_k: Some(40),
                repetition_penalty: 1.1,
                repetition_context_size: 128,
                instruction_template: "### Instruction: {}\n### Response:".to_string(),
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
    pub fn validate(&self) -> crate::error::Result<()> {
        // Validate model configuration
        if self.model.model_path.as_os_str().is_empty() {
            return Err(crate::error::EngineError::ConfigurationError {
                message: "Model path cannot be empty".to_string(),
                parameter: "model_path".to_string(),
            });
        }

        // Validate processing configuration
        match self.processing.mode {
            ProcessingMode::Batch => {
                if self.processing.batch_size.is_none() {
                    return Err(crate::error::EngineError::ConfigurationError {
                        message: "Batch size must be set for batch mode".to_string(),
                        parameter: "batch_size".to_string(),
                    });
                }
            }
            ProcessingMode::Queue => {
                if self.processing.queue_size.is_none() {
                    return Err(crate::error::EngineError::ConfigurationError {
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
                return Err(crate::error::EngineError::ConfigurationError {
                    message: "Device IDs cannot be empty when specified".to_string(),
                    parameter: "device_ids".to_string(),
                });
            }
        }

        if !(0.0..=1.0).contains(&self.gpu.memory_threshold) {
            return Err(crate::error::EngineError::ConfigurationError {
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
        assert_eq!(config.model.model_type, "meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(config.model.max_sequence_length, 4096);
    }

    #[test]
    fn test_llama_config_conversion() {
        let model_config = ModelConfig {
            model_path: PathBuf::new(),
            model_type: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
            max_sequence_length: 4096,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_hidden_layers: 32,
            intermediate_size: 11008,
            rms_norm_eps: 1e-6,
            vocab_size: 32000,
        };

        let llama_config = model_config.to_llama_config(true);
        assert_eq!(llama_config.hidden_size, 4096);
        assert_eq!(llama_config.num_attention_heads, 32);
        assert!(llama_config.use_flash_attention);
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
}