use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use crate::{
    config::EngineConfig,
    error::{EngineError, Result, ResourceType},
    metrics::MetricsCollector,
};

/// Manages GPU devices and memory allocation
pub struct GpuManager {
    config: Arc<EngineConfig>,
    metrics: Arc<Mutex<MetricsCollector>>,
    devices: Vec<GpuDevice>,
    allocation_state: Arc<RwLock<AllocationState>>,
}

/// Represents a single GPU device
#[derive(Debug)]
struct GpuDevice {
    device_id: usize,
    total_memory: usize,
    allocated_memory: Arc<Mutex<usize>>,
    device: candle_core::Device,
}

/// Tracks memory allocations across GPUs
#[derive(Debug, Default)]
struct AllocationState {
    device_allocations: Vec<DeviceAllocation>,
}

#[derive(Debug)]
struct DeviceAllocation {
    device_id: usize,
    allocated_bytes: usize,
    timestamp: std::time::Instant,
}

impl GpuManager {
    /// Create a new GPU manager
    pub async fn new(
        config: Arc<EngineConfig>,
        metrics: Arc<Mutex<MetricsCollector>>,
    ) -> Result<Self> {
        let devices = Self::initialize_devices(&config)?;
        
        Ok(Self {
            config,
            metrics,
            devices,
            allocation_state: Arc::new(RwLock::new(AllocationState::default())),
        })
    }

    /// Initialize available GPU devices
    fn initialize_devices(config: &EngineConfig) -> Result<Vec<GpuDevice>> {
        let mut devices = Vec::new();

        // Check for specific device IDs in config
        let device_ids = if let Some(ids) = &config.gpu.device_ids {
            ids.clone()
        } else {
            // Auto-detect available devices
            (0..8).filter(|&i| candle_core::Device::cuda_if_available(i).is_ok())
                .collect()
        };

        for device_id in device_ids {
            match candle_core::Device::cuda_if_available(device_id) {
                Ok(device) => {
                    // Get device properties
                    let total_memory = Self::get_device_memory(&device)
                        .unwrap_or(config.gpu.min_free_memory_gb as usize * 1024 * 1024 * 1024);
                    
                    devices.push(GpuDevice {
                        device_id,
                        total_memory,
                        allocated_memory: Arc::new(Mutex::new(0)),
                        device,
                    });
                }
                Err(e) => {
                    return Err(EngineError::GPUError {
                        device_id,
                        message: format!("Failed to initialize GPU: {}", e),
                        recoverable: false,
                    });
                }
            }
        }

        if devices.is_empty() {
            return Err(EngineError::ResourceError {
                message: "No GPU devices available".to_string(),
                resource_type: ResourceType::GPU,
            });
        }

        Ok(devices)
    }

    /// Get the device memory in bytes
    fn get_device_memory(device: &candle_core::Device) -> Option<usize> {
        #[cfg(feature = "cuda")]
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            if cuda_runtime_sys::cudaMemGetInfo(
                &mut free as *mut usize,
                &mut total as *mut usize,
            ) == 0 {
                Some(total)
            } else {
                None
            }
        }
        #[cfg(not(feature = "cuda"))]
        None
    }

    /// Get the number of available GPUs
    pub fn available_gpus(&self) -> usize {
        self.devices.len()
    }

    /// Request GPU memory allocation
    pub async fn allocate_memory(
        &self,
        bytes: usize,
        device_id: Option<usize>,
    ) -> Result<GpuAllocation> {
        let device = if let Some(id) = device_id {
            self.devices.iter()
                .find(|d| d.device_id == id)
                .ok_or_else(|| EngineError::GPUError {
                    device_id: id,
                    message: "Invalid device ID".to_string(),
                    recoverable: false,
                })?
        } else {
            // Find device with most free memory
            self.devices.iter()
                .max_by_key(|d| {
                    let allocated = *d.allocated_memory.try_lock()
                        .unwrap_or(&d.total_memory);
                    d.total_memory.saturating_sub(allocated)
                })
                .ok_or_else(|| EngineError::ResourceError {
                    message: "No GPU devices available".to_string(),
                    resource_type: ResourceType::GPU,
                })?
        };

        // Check memory availability
        let mut allocated = device.allocated_memory.lock().await;
        let available = device.total_memory.saturating_sub(*allocated);
        
        if bytes > available {
            return Err(EngineError::ResourceError {
                message: format!(
                    "Insufficient GPU memory. Requested: {}, Available: {}",
                    bytes, available
                ),
                resource_type: ResourceType::Memory,
            });
        }

        // Update allocation state
        *allocated += bytes;
        
        let allocation = GpuAllocation {
            device_id: device.device_id,
            bytes,
            device: device.device.clone(),
            manager: self.clone(),
        };

        // Update metrics
        let mut metrics = self.metrics.lock().await;
        metrics.record_gpu_allocation(device.device_id, bytes);

        Ok(allocation)
    }

    /// Release GPU memory
    async fn release_memory(&self, device_id: usize, bytes: usize) {
        if let Some(device) = self.devices.iter().find(|d| d.device_id == device_id) {
            let mut allocated = device.allocated_memory.lock().await;
            *allocated = allocated.saturating_sub(bytes);

            // Update metrics
            let mut metrics = self.metrics.lock().await;
            metrics.record_gpu_deallocation(device_id, bytes);
        }
    }

    /// Shutdown and cleanup
    pub async fn shutdown(&self) -> Result<()> {
        for device in &self.devices {
            // Ensure all memory is released
            let mut allocated = device.allocated_memory.lock().await;
            if *allocated > 0 {
                *allocated = 0;
            }

            // Force CUDA synchronization and reset
            #[cfg(feature = "cuda")]
            unsafe {
                if let Err(e) = cuda_runtime_sys::cudaDeviceSynchronize() {
                    return Err(EngineError::GPUError {
                        device_id: device.device_id,
                        message: format!("Failed to synchronize device: {}", e),
                        recoverable: false,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Represents a GPU memory allocation
pub struct GpuAllocation {
    device_id: usize,
    bytes: usize,
    device: candle_core::Device,
    manager: GpuManager,
}

impl GpuAllocation {
    /// Get the device for this allocation
    pub fn device(&self) -> &candle_core::Device {
        &self.device
    }

    /// Get the allocated size in bytes
    pub fn size(&self) -> usize {
        self.bytes
    }
}

impl Drop for GpuAllocation {
    fn drop(&mut self) {
        let device_id = self.device_id;
        let bytes = self.bytes;
        let manager = self.manager.clone();
        
        tokio::spawn(async move {
            manager.release_memory(device_id, bytes).await;
        });
    }
}

impl Clone for GpuManager {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            metrics: Arc::clone(&self.metrics),
            devices: self.devices.clone(),
            allocation_state: Arc::clone(&self.allocation_state),
        }
    }
}

impl Clone for GpuDevice {
    fn clone(&self) -> Self {
        Self {
            device_id: self.device_id,
            total_memory: self.total_memory,
            allocated_memory: Arc::clone(&self.allocated_memory),
            device: self.device.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn create_test_manager() -> Result<GpuManager> {
        let config = Arc::new(EngineConfig::default());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(config.clone())));
        GpuManager::new(config, metrics).await
    }

    #[tokio::test]
    async fn test_gpu_initialization() {
        let manager = create_test_manager().await;
        assert!(manager.is_ok());
        
        let manager = manager.unwrap();
        assert!(manager.available_gpus() > 0);
    }

    #[tokio::test]
    async fn test_memory_allocation() {
        let manager = create_test_manager().await.unwrap();
        
        // Test successful allocation
        let allocation = manager.allocate_memory(1024 * 1024, None).await;
        assert!(allocation.is_ok());
        
        // Test allocation tracking
        let alloc = allocation.unwrap();
        assert_eq!(alloc.size(), 1024 * 1024);
        
        // Memory should be automatically released when allocation is dropped
        drop(alloc);
        
        // Allow time for async cleanup
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Verify memory was released
        let device = &manager.devices[0];
        let allocated = *device.allocated_memory.lock().await;
        assert_eq!(allocated, 0);
    }

    #[tokio::test]
    async fn test_overallocation() {
        let manager = create_test_manager().await.unwrap();
        
        // Try to allocate more memory than available
        let huge_allocation = usize::MAX;
        let result = manager.allocate_memory(huge_allocation, None).await;
        assert!(result.is_err());
        
        match result {
            Err(EngineError::ResourceError { resource_type: ResourceType::Memory, .. }) => (),
            _ => panic!("Expected ResourceError with Memory type"),
        }
    }

    #[tokio::test]
    async fn test_multiple_allocations() {
        let manager = create_test_manager().await.unwrap();
        
        let mut allocations = Vec::new();
        for _ in 0..5 {
            let allocation = manager.allocate_memory(1024 * 1024, None).await;
            assert!(allocation.is_ok());
            allocations.push(allocation.unwrap());
        }
        
        // Verify total allocated memory
        let device = &manager.devices[0];
        let allocated = *device.allocated_memory.lock().await;
        assert_eq!(allocated, 5 * 1024 * 1024);
        
        // Release all allocations
        allocations.clear();
        
        // Allow time for async cleanup
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Verify all memory was released
        let allocated = *device.allocated_memory.lock().await;
        assert_eq!(allocated, 0);
    }

    #[tokio::test]
    async fn test_shutdown() {
        let manager = create_test_manager().await.unwrap();
        
        // Make some allocations
        let allocation = manager.allocate_memory(1024 * 1024, None).await.unwrap();
        
        // Shutdown should fail while allocations exist
        drop(allocation);
        
        // Allow time for async cleanup
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Now shutdown should succeed
        assert!(manager.shutdown().await.is_ok());
    }
}