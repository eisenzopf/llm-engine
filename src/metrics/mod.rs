//! Performance metrics collection and monitoring

mod collector;
mod types;

pub use collector::MetricsCollector;
pub use types::{
    MetricsSnapshot,
    ProcessingMetrics,
    GpuMetrics,
    ModelMetrics,
    QueueMetrics,
};

// Re-export utility functions
pub use collector::MetricsBuilder;

// Constants for metrics collection
pub(crate) const DEFAULT_METRICS_WINDOW: std::time::Duration = std::time::Duration::from_secs(60);
pub(crate) const MAX_METRICS_HISTORY: usize = 1000;