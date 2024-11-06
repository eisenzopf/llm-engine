//! Utility functions for logging, profiling, and performance monitoring

mod logging;
mod profiler;
mod memory;
mod tokenizer;

pub use logging::{setup_logging, LogConfig};
pub use profiler::{Profiler, ProfilerGuard, ProfileSpan};
pub use memory::{MemoryTracker, MemoryStats};
pub use tokenizer::{TokenCounter, TokenStats};

// Re-export commonly used utilities
pub mod prelude {
    pub use super::logging::{debug, info, warn, error};
    pub use super::profiler::profile_span;
    pub use super::memory::track_memory;
}