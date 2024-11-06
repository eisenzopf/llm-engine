use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use anyhow::Result;

use crate::error::EngineError;
use crate::metrics::MetricsCollector;

/// Represents a physical GPU device
pub struct GpuDevice {
    /// Device index
    index: usize,
    /// CUDA device handle
    device: candle_core::Device,
    /// Memory tracking
    memory: Arc<Mutex<DeviceMemory>>,
    /// Performance metrics
    metrics: Arc<Mutex<DeviceMetrics>>,
    /// Configuration
    config: Arc<crate::config::GpuConfig>,
}

/// Tracks GPU memory usage
#[derive(Debug)]
struct DeviceMemory {
    /// Total memory in bytes
    total_bytes: u64,
    /// Currently allocated memory in bytes
    allocated_bytes: u64,
    /// Peak memory usage in bytes
    peak_bytes: u64,
    /// Memory allocations
    allocations: Vec<MemoryAllocation>,
    /// Last garbage collection
    last_gc: Instant,
}

/// Represents a single memory allocation
#[derive(Debug)]
struct MemoryAllocation {
    /// Size in bytes
    size: u64,
    /// When the allocation was made
    timestamp: Instant,
    /// What the allocation is for
    purpose: String,
}

/// Device performance metrics
#[derive(Debug, Default)]
struct DeviceMetrics {
    /// Compute utilization percentage
    compute_utilization: f32,
    /// Memory bandwidth utilization
    memory_bandwidth: f32,
    /// Temperature in celsius
    temperature: f32,
    /// Power usage in watts
    power_usage: f32,
    /// Operation timings
    timings: Vec<OperationTiming>,
}

/// Timing for a GPU operation
#[derive(Debug)]
struct OperationTiming {
    /// Operation name
    operation: String,
    /// Duration
    duration: Duration,
    /// Memory used
    memory_used: u64,
    /// Timestamp
    timestamp: Instant,
}

impl GpuDevice {
    /// Create a new GPU device
    pub async fn new(
        index: usize,
        config: Arc<crate::config::GpuConfig>,
        metrics_collector: Arc<Mutex<MetricsCollector>>,
    ) -> Result<Self> {
        // Initialize CUDA device
        let device = candle_core::Device::cuda_if_available(index)
            .map_err(|e| EngineError::GPUError {
                device_id: index,
                message: format!("Failed to initialize CUDA device: {}", e),
                recoverable: false,
            })?;

        // Get device properties
        let (total_memory, _) = unsafe {
            cuda_runtime_sys::cudaMemGetInfo(
                &mut 0 as *mut usize,
                &mut 0 as *mut usize,
            )
        }.map_err(|e| EngineError::GPUError {
            device_id: index,
            message: format!("Failed to get device memory info: {}", e),
            recoverable: false,
        })?;

        let memory = DeviceMemory {
            total_bytes: total_memory as u64,
            allocated_bytes: 0,
            peak_bytes: 0,
            allocations: Vec::new(),
            last_gc: Instant::now(),
        };

        Ok(Self {
            index,
            device,
            memory: Arc::new(Mutex::new(memory)),
            metrics: Arc::new(Mutex::new(DeviceMetrics::default())),
            config,
        })
    }

    /// Allocate memory on the device
    pub async fn allocate_memory(
        &self,
        size: u64,
        purpose: &str,
    ) -> Result<DeviceAllocation> {
        let mut memory = self.memory.lock().await;
        
        // Check if we need to run garbage collection
        if memory.should_gc(&self.config) {
            self.force_gc().await?;
            memory = self.memory.lock().await;
        }

        // Check if we have enough memory
        let available = memory.total_bytes - memory.allocated_bytes;
        if size > available {
            return Err(EngineError::ResourceError {
                message: format!(
                    "Not enough GPU memory. Requested: {}GB, Available: {}GB",
                    size as f64 / 1024.0 / 1024.0 / 1024.0,
                    available as f64 / 1024.0 / 1024.0 / 1024.0
                ),
                resource_type: crate::error::ResourceType::Memory,
            }.into());
        }

        // Track allocation
        memory.allocated_bytes += size;
        memory.peak_bytes = memory.peak_bytes.max(memory.allocated_bytes);
        memory.allocations.push(MemoryAllocation {
            size,
            timestamp: Instant::now(),
            purpose: purpose.to_string(),
        });

        Ok(DeviceAllocation::new(self.clone(), size))
    }

    /// Force garbage collection
    pub async fn force_gc(&self) -> Result<()> {
        let mut memory = self.memory.lock().await;
        
        // Synchronize device
        unsafe {
            cuda_runtime_sys::cudaDeviceSynchronize();
        }

        // Clear CUDA cache
        unsafe {
            cuda_runtime_sys::cudaMemGetInfo(
                &mut 0 as *mut usize,
                &mut 0 as *mut usize,
            )?;
        }

        memory.last_gc = Instant::now();
        memory.allocated_bytes = 0;
        memory.allocations.clear();

        Ok(())
    }

    /// Record an operation timing
    pub async fn record_timing(
        &self,
        operation: &str,
        duration: Duration,
        memory_used: u64,
    ) {
        let mut metrics = self.metrics.lock().await;
        metrics.timings.push(OperationTiming {
            operation: operation.to_string(),
            duration,
            memory_used,
            timestamp: Instant::now(),
        });

        // Keep timing history bounded
        if metrics.timings.len() > 1000 {
            metrics.timings.remove(0);
        }
    }

    /// Update device metrics
    pub async fn update_metrics(&self) -> Result<()> {
        let mut metrics = self.metrics.lock().await;

        // Get compute utilization
        let mut utilization = 0u32;
        unsafe {
            cuda_runtime_sys::cudaDeviceGetAttribute(
                &mut utilization as *mut u32,
                cuda_runtime_sys::cudaDeviceAttr_t_cudaDevAttrComputeCapabilityMajor,
                self.index as i32,
            )?;
        }
        metrics.compute_utilization = utilization as f32;

        // Get temperature
        let mut temp = 0i32;
        unsafe {
            cuda_runtime_sys::cudaDeviceGetTemperature(
                &mut temp as *mut i32,
                self.index as i32,
            )?;
        }
        metrics.temperature = temp as f32;

        // Get power usage
        let mut power = 0f32;
        unsafe {
            cuda_runtime_sys::cudaDeviceGetPowerUsage(
                &mut power as *mut f32,
                self.index as i32,
            )?;
        }
        metrics.power_usage = power;

        Ok(())
    }

    /// Get current memory usage in bytes
    pub async fn memory_used(&self) -> u64 {
        self.memory.lock().await.allocated_bytes
    }

    /// Get peak memory usage in bytes
    pub async fn peak_memory(&self) -> u64 {
        self.memory.lock().await.peak_bytes
    }

    /// Get device metrics
    pub async fn get_metrics(&self) -> DeviceMetricsSnapshot {
        let metrics = self.metrics.lock().await;
        let memory = self.memory.lock().await;

        DeviceMetricsSnapshot {
            compute_utilization: metrics.compute_utilization,
            memory_bandwidth: metrics.memory_bandwidth,
            temperature: metrics.temperature,
            power_usage: metrics.power_usage,
            memory_used: memory.allocated_bytes,
            peak_memory: memory.peak_bytes,
            total_memory: memory.total_bytes,
        }
    }
}

impl DeviceMemory {
    /// Check if garbage collection is needed
    fn should_gc(&self, config: &crate::config::GpuConfig) -> bool {
        let memory_pressure = self.allocated_bytes as f64 / self.total_bytes as f64;
        let time_since_gc = self.last_gc.elapsed();

        memory_pressure > config.memory_cleanup_threshold ||
            time_since_gc > Duration::from_secs(60)
    }
}

/// Represents a memory allocation on the device
pub struct DeviceAllocation {
    device: GpuDevice,
    size: u64,
}

impl DeviceAllocation {
    fn new(device: GpuDevice, size: u64) -> Self {
        Self { device, size }
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        // Queue memory deallocation
        let device = self.device.clone();
        let size = self.size;
        tokio::spawn(async move {
            if let Ok(mut memory) = device.memory.try_lock() {
                memory.allocated_bytes = memory.allocated_bytes.saturating_sub(size);
            }
        });
    }
}

/// Snapshot of device metrics
#[derive(Debug, Clone)]
pub struct DeviceMetricsSnapshot {
    pub compute_utilization: f32,
    pub memory_bandwidth: f32,
    pub temperature: f32,
    pub power_usage: f32,
    pub memory_used: u64,
    pub peak_memory: u64,
    pub total_memory: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn create_test_device() -> Result<GpuDevice> {
        let config = Arc::new(crate::config::GpuConfig {
            memory_threshold: 0.9,
            memory_cleanup_threshold: 0.8,
            ..Default::default()
        });

        let metrics = Arc::new(Mutex::new(MetricsCollector::new(
            Arc::new(crate::config::EngineConfig::default())
        )));

        GpuDevice::new(0, config, metrics).await
    }

    #[tokio::test]
    async fn test_memory_allocation() -> Result<()> {
        let device = create_test_device().await?;
        
        // Allocate some memory
        let allocation = device.allocate_memory(1024 * 1024, "test").await?;
        assert_eq!(device.memory_used().await, 1024 * 1024);
        
        // Free memory
        drop(allocation);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(device.memory_used().await, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_garbage_collection() -> Result<()> {
        let device = create_test_device().await?;
        
        // Allocate memory
        let _allocation = device.allocate_memory(1024 * 1024, "test").await?;
        assert!(device.memory_used().await > 0);
        
        // Force GC
        device.force_gc().await?;
        assert_eq!(device.memory_used().await, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_metrics_recording() -> Result<()> {
        let device = create_test_device().await?;
        
        // Record some operations
        device.record_timing("test_op", Duration::from_millis(100), 1024).await;
        
        let metrics = device.get_metrics().await;
        assert!(metrics.memory_used >= 0);
        assert!(metrics.temperature >= 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_pressure() -> Result<()> {
        let device = create_test_device().await?;
        
        // Try to allocate more than available
        let total_memory = device.memory.lock().await.total_bytes;
        let result = device.allocate_memory(total_memory * 2, "too_large").await;
        
        assert!(result.is_err());
        match result {
            Err(e) => {
                match e.downcast_ref::<EngineError>() {
                    Some(EngineError::ResourceError { resource_type, .. }) => {
                        assert_eq!(*resource_type, crate::error::ResourceType::Memory);
                    }
                    _ => panic!("Expected ResourceError"),
                }
            }
            _ => panic!("Expected error"),
        }

        Ok(())
    }
}