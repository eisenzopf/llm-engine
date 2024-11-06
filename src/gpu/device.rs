// Location: src/gpu/device.rs

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use std::collections::HashMap;

use crate::error::EngineError;
use crate::metrics::MetricsCollector;

/// Represents a physical GPU device
pub struct GpuDevice {
    /// Device index
    index: usize,
    /// Candle device handle
    device: Device,
    /// Memory tracking
    memory: Arc<Mutex<DeviceMemory>>,
    /// Performance metrics
    metrics: Arc<Mutex<DeviceMetrics>>,
    /// Device capabilities
    capabilities: DeviceCapabilities,
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
    /// Memory pool for tensor recycling
    tensor_pool: TensorPool,
}

/// Device capabilities and limits
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Whether flash attention is supported
    pub supports_flash_attention: bool,
    /// Available compute capability
    pub compute_capability: (i32, i32),
    /// Memory bus width
    pub memory_bus_width: i32,
    /// Maximum shared memory per block
    pub max_shared_memory: usize,
}

/// Pool for recycling tensor allocations
#[derive(Debug, Default)]
struct TensorPool {
    /// Pooled tensors by shape
    tensors: HashMap<Vec<usize>, Vec<Tensor>>,
    /// Total memory used by pool
    total_bytes: u64,
    /// Maximum pool size
    max_pool_bytes: u64,
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
    /// Stack trace of allocation
    stack_trace: String,
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
        device: Device,
        metrics_collector: Arc<Mutex<MetricsCollector>>,
    ) -> Result<Self> {
        let capabilities = Self::detect_capabilities(&device)?;
        let total_memory = Self::get_total_memory(&device)?;
        
        let memory = DeviceMemory {
            total_bytes: total_memory,
            allocated_bytes: 0,
            peak_bytes: 0,
            allocations: Vec::new(),
            last_gc: Instant::now(),
            tensor_pool: TensorPool {
                tensors: HashMap::new(),
                total_bytes: 0,
                max_pool_bytes: total_memory / 4, // Use up to 25% for pool
            },
        };

        Ok(Self {
            index,
            device,
            memory: Arc::new(Mutex::new(memory)),
            metrics: Arc::new(Mutex::new(DeviceMetrics::default())),
            capabilities,
        })
    }

    fn get_total_memory(device: &Device) -> Option<usize> {
        match device {
            Device::Cuda(_) => unsafe {
                let mut free = 0;
                let mut total = 0;
                candle_core::cuda_backend::cuda::cuMemGetInfo(
                    &mut free,
                    &mut total,
                ).ok()?;
                Some(total)
            },
            _ => None
        }
    }

    pub async fn allocate_tensor<T: candle_core::WithDType>(
        &self,
        shape: &[usize],
        data: &[T],
        pool_key: Option<String>,
    ) -> Result<Tensor> {
        let size = data.len() * std::mem::size_of::<T>();
        
        // Try to get from pool first
        if let Some(ref key) = pool_key {
            let mut memory = self.memory.lock().await;
            if let Some(tensor) = memory.tensor_pool.get(shape) {
                return Ok(tensor);
            }
        }

        // Allocate new tensor
        let tensor = Tensor::new(data, &self.device)?;
        
        Ok(tensor)
    }

    /// Detect device capabilities
    fn detect_capabilities(device: &Device) -> Result<DeviceCapabilities> {
        match device {
            Device::Cuda(cuda_dev) => {
                let (major, minor) = unsafe {
                    let mut major = 0;
                    let mut minor = 0;
                    candle_core::cuda_backend::cuda::cuDeviceGetAttribute(
                        &mut major,
                        candle_core::cuda_backend::cuda::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        cuda_dev.ordinal() as i32,
                    )?;
                    candle_core::cuda_backend::cuda::cuDeviceGetAttribute(
                        &mut minor,
                        candle_core::cuda_backend::cuda::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        cuda_dev.ordinal() as i32,
                    )?;
                    (major, minor)
                };

                Ok(DeviceCapabilities {
                    max_batch_size: 32,
                    max_sequence_length: 32768,
                    supports_flash_attention: major >= 8,
                    compute_capability: (major, minor),
                    memory_bus_width: 384,  // Default for modern GPUs
                    max_shared_memory: 48 * 1024,  // 48KB default
                })
            }
            Device::Cpu => Ok(DeviceCapabilities {
                max_batch_size: 8,
                max_sequence_length: 32768,
                supports_flash_attention: false,
                compute_capability: (0, 0),
                memory_bus_width: 0,
                max_shared_memory: 0,
            }),
            _ => Err(EngineError::DeviceError {
                message: "Unsupported device type".to_string(),
            }.into()),
        }
    }

    /// Allocate a new tensor with optional pooling
    pub async fn allocate_tensor<T: candle_core::WithDType>(
        &self,
        shape: &[usize],
        data: &[T],
        pool_key: Option<String>,
    ) -> Result<Tensor> {
        let size = data.len() * std::mem::size_of::<T>();
        
        // Try to get from pool first
        if let Some(ref key) = pool_key {
            let mut memory = self.memory.lock().await;
            if let Some(tensor) = memory.tensor_pool.get(shape) {
                return Ok(tensor);
            }
        }

        // Allocate new tensor
        let tensor = Tensor::new(data, &self.device)?;
        
        // Track allocation
        let mut memory = self.memory.lock().await;
        memory.allocated_bytes += size as u64;
        memory.peak_bytes = memory.peak_bytes.max(memory.allocated_bytes);
        
        memory.allocations.push(MemoryAllocation {
            size: size as u64,
            timestamp: Instant::now(),
            purpose: pool_key.unwrap_or_else(|| "unknown".to_string()),
            stack_trace: std::backtrace::Backtrace::force_capture().to_string(),
        });

        Ok(tensor)
    }

    /// Release a tensor back to the pool
    pub async fn release_tensor(&self, tensor: Tensor, pool_key: Option<String>) -> Result<()> {
        if let Some(key) = pool_key {
            let mut memory = self.memory.lock().await;
            memory.tensor_pool.add(tensor, &key)?;
        }
        Ok(())
    }

    /// Force garbage collection
    pub async fn force_gc(&self) -> Result<()> {
        let mut memory = self.memory.lock().await;
        
        // Clear tensor pool
        memory.tensor_pool.clear();
        
        // Synchronize device
        if let Device::Cuda(_) = self.device {
            unsafe {
                candle_core::cuda_backend::cuda::cuStreamSynchronize(0)?;
                candle_core::cuda_backend::cuda::cuMemGetInfo(
                    &mut 0 as *mut usize,
                    &mut 0 as *mut usize,
                )?;
            }
        }

        memory.allocated_bytes = 0;
        memory.allocations.clear();
        memory.last_gc = Instant::now();

        Ok(())
    }

    /// Record operation timing
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

        if metrics.timings.len() > 1000 {
            metrics.timings.remove(0);
        }
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> DeviceMetricsSnapshot {
        let metrics = self.metrics.lock().await;
        let memory = self.memory.lock().await;

        DeviceMetricsSnapshot {
            compute_utilization: metrics.compute_utilization,
            memory_bandwidth: metrics.memory_bandwidth,
            temperature: metrics.temperature,
            power_usage: metrics.power_usage,
            allocated_memory: memory.allocated_bytes,
            peak_memory: memory.peak_bytes,
            pool_memory: memory.tensor_pool.total_bytes,
        }
    }
}

impl TensorPool {
    /// Get a tensor from the pool
    fn get(&mut self, shape: &[usize]) -> Option<Tensor> {
        self.tensors.get_mut(shape)?.pop()
    }

    /// Add a tensor to the pool
    fn add(&mut self, tensor: Tensor, key: &str) -> Result<()> {
        let shape = tensor.dims().to_vec();
        let size = tensor.elem_size()? * shape.iter().product::<usize>() as u64;
        
        // Check if adding would exceed pool limit
        if self.total_bytes + size > self.max_pool_bytes {
            // Remove oldest tensors until we have space
            for tensors in self.tensors.values_mut() {
                while !tensors.is_empty() && self.total_bytes + size > self.max_pool_bytes {
                    tensors.remove(0);
                    self.total_bytes -= size;
                }
            }
        }

        self.tensors.entry(shape).or_default().push(tensor);
        self.total_bytes += size;
        Ok(())
    }

    /// Clear the pool
    fn clear(&mut self) {
        self.tensors.clear();
        self.total_bytes = 0;
    }
}

#[derive(Debug, Clone)]
pub struct DeviceMetricsSnapshot {
    pub compute_utilization: f32,
    pub memory_bandwidth: f32,
    pub temperature: f32,
    pub power_usage: f32,
    pub allocated_memory: u64,
    pub peak_memory: u64,
    pub pool_memory: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_device() -> Result<GpuDevice> {
        let metrics = Arc::new(Mutex::new(MetricsCollector::default()));
        GpuDevice::new(0, Device::Cpu, metrics).await
    }

    #[tokio::test]
    async fn test_tensor_allocation() -> Result<()> {
        let device = create_test_device().await?;
        
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = device.allocate_tensor(&[2, 2], &data, Some("test".to_string())).await?;
        
        assert_eq!(tensor.dims(), &[2, 2]);
        
        let metrics = device.get_metrics().await;
        assert!(metrics.allocated_memory > 0);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_tensor_pool() -> Result<()> {
        let device = create_test_device().await?;
        
        // Allocate and release to pool
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = device.allocate_tensor(&[2, 2], &data, Some("test".to_string())).await?;
        device.release_tensor(tensor, Some("test".to_string())).await?;
        
        // Verify pool usage
        let metrics = device.get_metrics().await;
        assert!(metrics.pool_memory > 0);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_garbage_collection() -> Result<()> {
        let device = create_test_device().await?;
        
        // Allocate some tensors
        let data: Vec<f32> = vec![1.0; 1000];
        for _ in 0..10 {
            let _ = device.allocate_tensor(&[100], &data, None).await?;
        }
        
        // Force GC
        device.force_gc().await?;
        
        // Verify memory was freed
        let metrics = device.get_metrics().await;
        assert_eq!(metrics.allocated_memory, 0);
        assert_eq!(metrics.pool_memory, 0);
        
        Ok(())
    }
}