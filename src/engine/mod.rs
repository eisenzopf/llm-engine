//! Engine module providing the main interface to the LLM processing engine

mod builder;
mod engine;

pub use builder::EngineBuilder;
pub use engine::{LLMEngine, EngineInfo};

use crate::{
    config::EngineConfig,
    error::Result,
    types::{ProcessingOutput, StreamHandle, QueueHandle},
};

/// Trait defining the core processing capabilities
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    /// Process a single input
    async fn process(&self, input: String) -> Result<ProcessingOutput>;
    
    /// Shutdown the processor and release resources
    async fn shutdown(&self) -> Result<()>;
}

/// Trait for streaming mode processing
#[async_trait::async_trait]
pub trait StreamProcessor: Processor {
    /// Get stream handles for processing
    async fn get_stream_handles(&self) -> Result<Vec<StreamHandle>>;
}

/// Trait for batch mode processing
#[async_trait::async_trait]
pub trait BatchProcessor: Processor {
    /// Process a batch of inputs
    async fn process_batch(&self, inputs: Vec<String>) -> Result<Vec<ProcessingOutput>>;
}

/// Trait for queue mode processing
#[async_trait::async_trait]
pub trait QueueProcessor: Processor {
    /// Enqueue an input for processing
    async fn enqueue(&self, input: String) -> Result<QueueHandle>;
    
    /// Get current queue size
    fn queue_size(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // Mock processor for testing
    struct MockProcessor;

    #[async_trait::async_trait]
    impl Processor for MockProcessor {
        async fn process(&self, input: String) -> Result<ProcessingOutput> {
            Ok(ProcessingOutput {
                text: input,
                tokens: vec![],
                processing_time: Duration::from_millis(100),
            })
        }

        async fn shutdown(&self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_mock_processor() {
        let processor = MockProcessor;
        let result = processor.process("test".to_string()).await;
        assert!(result.is_ok());
    }
}