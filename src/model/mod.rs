//! Model module providing core LLM model functionality and management

mod runtime;
mod loader;
mod tokenizer;

// Re-export core types and traits
pub use runtime::ModelRuntime;
pub use loader::ModelLoader;
pub use tokenizer::LlamaTokenizer;

// Re-export common types used throughout the module
pub use runtime::{
    RuntimeStats,
    GpuStats,
};

// Constants for model configuration
pub(crate) const DEFAULT_PROMPT_TEMPLATE: &str = "### Instruction:\n{}\n### Response:";
pub(crate) const DEFAULT_MAX_TOKENS: usize = 2048;
pub(crate) const DEFAULT_TEMPERATURE: f64 = 0.7;
pub(crate) const DEFAULT_TOP_P: f64 = 0.95;

/// Error type for model-related operations
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Failed to load model: {0}")]
    LoadError(String),
    
    #[error("Failed to process input: {0}")]
    ProcessingError(String),
    
    #[error("Invalid model configuration: {0}")]
    ConfigurationError(String),
    
    #[error("GPU error: {0}")]
    GpuError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constants() {
        assert!(DEFAULT_MAX_TOKENS > 0);
        assert!((0.0..=1.0).contains(&DEFAULT_TEMPERATURE));
        assert!((0.0..=1.0).contains(&DEFAULT_TOP_P));
    }
}