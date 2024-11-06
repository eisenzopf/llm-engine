use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::Instant;

use crate::error::{EngineError, Result};
use crate::metrics::MetricsCollector;

/// Represents a GPU memory allocation
pub struct GpuAllocation {
    device_id: usize,
    size: usize,
    allocated_at: Instant,
    metrics: Arc<Mutex<MetricsCollector>>,
}

impl GpuAllocation {
    /// Create a new GPU allocation
    pub(crate) fn new(
        device_id: usize,
        size: usize,
        metrics: Arc<Mutex<MetricsCollector>>,
    ) -> Self {
        let allocation = Self {
            device_id,
            size,
            allocated_at: Instant::now(),
            metrics.clone(),
        };

        // Record allocation in metrics
        tokio::spawn({
            let metrics = metrics.clone();
            async move {
                if let Ok(mut metrics) = metrics.try_lock() {
                    metrics.record_gpu_allocation(device_id, size);
                }
            }
        });

        allocation
    }

    /// Get the allocation size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the device ID this allocation is for
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Get how long this allocation has existed
    pub fn age(&self) -> std::time::Duration {
        self.allocated_at.elapsed()
    }
}

impl Drop for GpuAllocation {
    fn drop(&mut self) {
        // Record deallocation in metrics
        let metrics = self.metrics.clone();
        let device_id = self.device_id;
        let size = self.size;
        
        tokio::spawn(async move {
            if let Ok(mut metrics) = metrics.try_lock() {
                metrics.record_gpu_deallocation(device_id, size);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn create_test_allocation() -> GpuAllocation {
        let config = Arc::new(crate::config::EngineConfig::default());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(config)));
        
        GpuAllocation::new(0, 1024, metrics)
    }

    #[tokio::test]
    async fn test_allocation_basics() {
        let alloc = create_test_allocation().await;
        assert_eq!(alloc.size(), 1024);
        assert_eq!(alloc.device_id(), 0);
        assert!(alloc.age() >= Duration::from_nanos(0));
    }

    #[tokio::test]
    async fn test_allocation_metrics() {
        let config = Arc::new(crate::config::EngineConfig::default());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(config)));
        
        // Create and drop allocation
        {
            let _alloc = GpuAllocation::new(0, 1024, metrics.clone());
        }

        // Allow time for async operations
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let metrics = metrics.lock().await;
        let snapshot = metrics.snapshot().await;
        assert_eq!(snapshot.gpu_memory_used[0], 0); // Should be freed
    }
}