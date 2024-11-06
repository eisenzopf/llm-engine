//! LLM Engine - High performance GPU-accelerated language model processing
//! 
//! This crate provides a flexible interface for running large language models
//! across multiple GPUs with support for streaming, batch, and queue processing.

#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]

use std::fmt;
use anyhow::Result;
use candle_core::Device;

// Public modules
pub mod engine;
pub mod error;
pub mod config;
pub mod types;
pub mod metrics;

// Internal modules
mod gpu;
mod model;
mod processing;
mod utils;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const MIN_RUST_VERSION: &str = "1.70.0";

// Re-exports for public API
pub use engine::{LLMEngine, EngineBuilder};
pub use error::{EngineError, Result};
pub use config::{EngineConfig, ProcessingMode};
pub use types::{ProcessingOutput, StreamHandle, QueueHandle};
pub use metrics::MetricsCollector;

/// Feature detection for supported backends
pub struct Features {
    /// Whether CUDA support is enabled
    pub cuda: bool,
    /// Number of detected CUDA devices
    pub cuda_devices: usize,
}

impl Features {
    /// Detect available features at runtime
    pub fn detect() -> Self {
        #[cfg(feature = "cuda")]
        let (cuda, cuda_devices) = {
            // Use candle's CUDA detection instead of cuda_runtime_sys
            match Device::cuda_if_available(0) {
                Ok(_) => {
                    let count = (0..8)
                        .filter(|&i| Device::cuda_if_available(i).is_ok())
                        .count();
                    (true, count)
                }
                Err(_) => (false, 0),
            }
        };

        #[cfg(not(feature = "cuda"))]
        let (cuda, cuda_devices) = (false, 0);

        Self {
            cuda,
            cuda_devices,
        }
    }
}

impl fmt::Display for Features {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CUDA support: {}", if self.cuda { "yes" } else { "no" })?;
        if self.cuda {
            writeln!(f, "CUDA devices: {}", self.cuda_devices)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_detection() {
        let features = Features::detect();
        println!("Detected features:\n{}", features);
    }

    #[test]
    fn test_version_numbers() {
        assert!(!VERSION.is_empty());
        assert!(!MIN_RUST_VERSION.is_empty());
    }
}