//! GPU management and memory allocation

mod manager;
mod device;
mod allocation;

pub use manager::GpuManager;
pub use device::{GpuDevice, DeviceMetricsSnapshot};
pub use allocation::GpuAllocation;

// Re-export commonly used types
pub use manager::{GpuConfig, GpuStats};