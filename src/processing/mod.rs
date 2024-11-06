//! Processing module handling different processing modes

mod stream;
mod batch;
mod queue;
mod common;

pub use stream::{StreamProcessor, StreamHandle};
pub use batch::{BatchProcessor, BatchHandle};
pub use queue::{QueueProcessor, QueueHandle};
pub use common::{ProcessingStats, ProcessingResult};

use crate::error::Result;

/// Common trait for all processors
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    /// Shutdown the processor
    async fn shutdown(&self) -> Result<()>;
    
    /// Get current processing statistics
    async fn get_stats(&self) -> ProcessingStats;
}