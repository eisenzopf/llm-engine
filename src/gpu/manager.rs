// Location: src/gpu/manager.rs

use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use anyhow::Result;
use candle_core::{Device, DType};
use crate::gpu::GpuAllocation;
use crate::{
    config::EngineConfig,
    error::{EngineError, ResourceType},
    metrics::MetricsCollector,
};


/// Manages GPU devices and memory allocation
pub struct GpuManager {
    config: Arc<EngineConfig>,
    metrics: Arc<Mutex<MetricsCollector>>,
    devices: Vec<GpuDevice>,
    allocation_state: Arc<RwLock<AllocationState>>,
}

#[derive(Debug)]
struct GpuDevice {
    device_id: usize,
    device: Device,
    total_memory: usize,
    allocated_memory: Arc<Mutex<usize>>,
    peak_memory: Arc<Mutex<usize>>,
    last_garbage_collection: Arc<Mutex<std::time::Instant>>,
}

#[derive(Debug, Default)]
struct AllocationState {
    device_allocations: Vec<DeviceAllocation>,
    allocation_history: Vec<AllocationRecord>,
}

#[derive(Debug)]
struct DeviceAllocation {
    device_id: usize,
    size: usize,
    purpose: String,
    timestamp: std::time::Instant,
}

#[derive(Debug)]
struct AllocationRecord {
    device_id: usize,
    size: usize,
    success: bool,
    error: Option<String>,
    timestamp: std::time::Instant,
}

impl GpuManager {
    /// Create a new GPU manager
    pub async fn new(
        config: Arc<EngineConfig>,
        metrics: Arc<Mutex<MetricsCollector>>,
    ) -> Result<Self> {
        let mut devices = Vec::new();
        let device_ids = if let Some(ids) = &config.gpu.device_ids {
            ids.clone()
        } else {
            // Auto-detect available devices
            let mut available = Vec::new();
            for i in 0..8 {
                if Device::cuda_if_available(i).is_ok() {
                    available.push(i);
                }
            }
            if available.is_empty() {
                return Err(EngineError::ResourceError {
                    message: "No GPU devices available".to_string(),
                    resource_type: ResourceType::GPU,
                }.into());
            }
            available
        };

        for device_id in device_ids {
            let device = Device::cuda_if_available(device_id).map_err(|e| {
                EngineError::GPUError {
                    device_id,
                    message: format!("Failed to initialize CUDA device: {}", e),
                    recoverable: false,
                }
            })?;

            let total_memory = Self::get_device_memory(&device)
                .unwrap_or_else(|| {
                    (config.gpu.max_memory_gb.unwrap_or(8.0) * 1024.0 * 1024.0 * 1024.0) as usize
                });

            devices.push(GpuDevice {
                device_id,
                device,
                total_memory,
                allocated_memory: Arc::new(Mutex::new(0)),
                peak_memory: Arc::new(Mutex::new(0)),
                last_garbage_collection: Arc::new(Mutex::new(std::time::Instant::now())),
            });
        }

        Ok(Self {
            config,
            metrics,
            devices,
            allocation_state: Arc::new(RwLock::new(AllocationState::default())),
        })
    }

    /// Get the number of available GPUs
    pub fn available_gpus(&self) -> usize {
        self.devices.len()
    }

    /// Request GPU memory allocation
    pub async fn allocate_memory(
        &self,
        size: usize,
        purpose: &str,
        preferred_device: Option<usize>,
    ) -> Result<GpuAllocation> {
        // Try preferred device first if specified
        if let Some(device_id) = preferred_device {
            if let Ok(allocation) = self.try_allocate(device_id, size, purpose).await {
                return Ok(allocation);
            }
        }

        // Otherwise, find device with most free memory
        let mut best_device = None;
        let mut max_free = 0;

        for device in &self.devices {
            let allocated = *device.allocated_memory.lock().await;
            let free = device.total_memory.saturating_sub(allocated);
            if free > max_free {
                max_free = free;
                best_device = Some(device);
            }
        }

        let device = best_device.ok_or_else(|| EngineError::ResourceError {
            message: "No GPU with sufficient memory available".to_string(),
            resource_type: ResourceType::Memory,
        })?;

        self.try_allocate(device.device_id, size, purpose).await
    }

    async fn try_allocate(
        &self,
        device_id: usize,
        size: usize,
        purpose: &str,
    ) -> Result<GpuAllocation> {
        let device = self.devices.iter()
            .find(|d| d.device_id == device_id)
            .ok_or_else(|| EngineError::GPUError {
                device_id,
                message: "Invalid device ID".to_string(),
                recoverable: false,
            })?;

        // Check if garbage collection is needed
        let should_gc = {
            let last_gc = device.last_garbage_collection.lock().await;
            last_gc.elapsed() > std::time::Duration::from_secs(60)
        };

        if should_gc {
            self.force_gc(device_id).await?;
        }

        // Check memory availability
        let mut allocated = device.allocated_memory.lock().await;
        let available = device.total_memory.saturating_sub(*allocated);
        
        if size > available {
            // Record failed allocation
            let mut state = self.allocation_state.write().await;
            state.allocation_history.push(AllocationRecord {
                device_id,
                size,
                success: false,
                error: Some("Insufficient memory".to_string()),
                timestamp: std::time::Instant::now(),
            });

            return Err(EngineError::ResourceError {
                message: format!(
                    "Insufficient GPU memory on device {}. Requested: {:.2}GB, Available: {:.2}GB",
                    device_id,
                    size as f64 / 1024.0 / 1024.0 / 1024.0,
                    available as f64 / 1024.0 / 1024.0 / 1024.0
                ),
                resource_type: ResourceType::Memory,
            }.into());
        }

        // Update allocation state
        *allocated += size;
        let mut peak = device.peak_memory.lock().await;
        *peak = (*peak).max(*allocated);

        let mut state = self.allocation_state.write().await;
        state.device_allocations.push(DeviceAllocation {
            device_id,
            size,
            purpose: purpose.to_string(),
            timestamp: std::time::Instant::now(),
        });

        state.allocation_history.push(AllocationRecord {
            device_id,
            size,
            success: true,
            error: None,
            timestamp: std::time::Instant::now(),
        });

        // Update metrics
        let mut metrics = self.metrics.lock().await;
        metrics.record_gpu_allocation(device_id, size).await;

        Ok(GpuAllocation {
            device_id,
            size,
            device: device.device.clone(),
            manager: self.clone(),
        })
    }

    /// Force garbage collection on a device
    pub async fn force_gc(&self, device_id: usize) -> Result<()> {
        let device = self.devices.iter()
            .find(|d| d.device_id == device_id)
            .ok_or_else(|| EngineError::GPUError {
                device_id,
                message: "Invalid device ID".to_string(),
                recoverable: false,
            })?;

        if let Device::Cuda(_) = device.device {
            unsafe {
                candle_core::backend::cuda::cuStreamSynchronize(0)?;
                candle_core::backend::cuda::cuMemGetInfo(
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )?;
            }
        }

        Ok(())
    }

    /// Get device memory in bytes
    pub fn get_device_memory(device: &Device) -> Option<usize> {
        match device {
            Device::Cuda(_) => {
                let mut free = 0;
                let mut total = 0;
                unsafe {
                    if candle_core::backend::cuda::cuMemGetInfo(
                        &mut free as *mut usize,
                        &mut total as *mut usize,
                    ).is_ok() {
                        Some(total)
                    } else {
                        None
                    }
                }
            }
            _ => None
        }
    }

    /// Get current memory usage stats for all devices
    pub async fn get_memory_stats(&self) -> Vec<DeviceMemoryStats> {
        let mut stats = Vec::new();
        for device in &self.devices {
            let allocated = *device.allocated_memory.lock().await;
            let peak = *device.peak_memory.lock().await;
            
            stats.push(DeviceMemoryStats {
                device_id: device.device_id,
                total_memory: device.total_memory,
                allocated_memory: allocated,
                peak_memory: peak,
                free_memory: device.total_memory.saturating_sub(allocated),
            });
        }
        stats
    }

    /// Shutdown and cleanup
    pub async fn shutdown(&self) -> Result<()> {
        for device in &self.devices {
            self.force_gc(device.device_id).await?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DeviceMemoryStats {
    pub device_id: usize,
    pub total_memory: usize,
    pub allocated_memory: usize,
    pub peak_memory: usize,
    pub free_memory: usize,
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
            device: self.device.clone(),
            total_memory: self.total_memory,
            allocated_memory: Arc::clone(&self.allocated_memory),
            peak_memory: Arc::clone(&self.peak_memory),
            last_garbage_collection: Arc::clone(&self.last_garbage_collection),
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
    async fn test_memory_allocation() -> Result<()> {
        let manager = create_test_manager().await?;
        
        // Test successful allocation
        let allocation = manager.allocate_memory(
            1024 * 1024 * 1024, // 1GB
            "test allocation",
            None
        ).await?;
        
        assert_eq!(allocation.size, 1024 * 1024 * 1024);
        
        // Check memory stats
        let stats = manager.get_memory_stats().await;
        assert!(!stats.is_empty());
        assert!(stats[0].allocated_memory > 0);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_garbage_collection() -> Result<()> {
        let manager = create_test_manager().await?;
        
        // Allocate some memory
        let _allocation = manager.allocate_memory(
            1024 * 1024 * 1024,
            "test allocation",
            None
        ).await?;
        
        // Force GC
        manager.force_gc(0).await?;
        
        // Check memory was freed
        let stats = manager.get_memory_stats().await;
        assert_eq!(stats[0].allocated_memory, 0);
        
        Ok(())
    }
}