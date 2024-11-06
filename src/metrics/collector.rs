use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use crate::config::EngineConfig;
use anyhow::Result;

/// Collects and manages performance metrics
pub struct MetricsCollector {
    config: Arc<EngineConfig>,
    state: Arc<RwLock<MetricsState>>,
    start_time: Instant,
}

#[derive(Debug, Default)]
struct MetricsState {
    // GPU metrics
    gpu_memory_allocated: Vec<usize>,
    gpu_utilization: Vec<f32>,
    
    // Processing metrics
    total_processed: usize,
    total_tokens: usize,
    total_processing_time: Duration,
    
    // Batch metrics
    batch_sizes: Vec<usize>,
    batch_processing_times: Vec<Duration>,
    
    // Error metrics
    total_errors: usize,
    error_types: Vec<(String, usize)>,
    
    // Performance metrics
    tokens_per_second: Vec<f32>,
    latencies: Vec<Duration>,
}

/// A snapshot of current metrics
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub timestamp: Instant,
    pub uptime: Duration,
    
    // GPU metrics
    pub gpu_memory_used: Vec<usize>,
    pub gpu_utilization: Vec<f32>,
    
    // Processing metrics
    pub total_processed: usize,
    pub total_tokens: usize,
    pub average_tokens_per_second: f32,
    
    // Latency metrics
    pub average_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    
    // Error metrics
    pub error_rate: f32,
    pub error_types: Vec<(String, usize)>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(config: Arc<EngineConfig>) -> Self {
        let gpu_count = config.gpu.device_ids.as_ref()
            .map(|ids| ids.len())
            .unwrap_or(1);
            
        let state = MetricsState {
            gpu_memory_allocated: vec![0; gpu_count],
            gpu_utilization: vec![0.0; gpu_count],
            ..Default::default()
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
            start_time: Instant::now(),
        }
    }

    /// Record GPU memory allocation
    pub async fn record_gpu_allocation(
        &self,
        device_id: usize,
        bytes: usize,
    ) {
        let mut state = self.state.write().await;
        if device_id < state.gpu_memory_allocated.len() {
            state.gpu_memory_allocated[device_id] += bytes;
        }
    }

    /// Record GPU memory deallocation
    pub fn record_gpu_deallocation(&mut self, device_id: usize, bytes: usize) {
        if let Ok(mut state) = self.state.try_write() {
            if device_id < state.gpu_memory_allocated.len() {
                state.gpu_memory_allocated[device_id] = 
                    state.gpu_memory_allocated[device_id].saturating_sub(bytes);
            }
        }
    }

    pub fn record_allocation(&self, size: usize, device_id: usize) {
        if let Some(stats) = self.device_stats.get(device_id) {
            let mut stats = stats.write();
            stats.allocated_memory += size;
            stats.peak_memory = stats.peak_memory.max(stats.allocated_memory);
        }
    }

    pub fn record_deallocation(&self, size: usize, device_id: usize) {
        if let Some(stats) = self.device_stats.get(device_id) {
            let mut stats = stats.write();
            stats.allocated_memory = stats.allocated_memory.saturating_sub(size);
        }
    }

    /// Record stream processing
    pub async fn record_stream_processing(
        &self,
        stream_id: usize,
        tokens: usize,
        duration: Duration,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        state.total_tokens += tokens;
        state.total_processing_time += duration;
        Ok(())
    }

    /// Record queue batch processing
    pub async fn record_queue_batch(
        &self,
        batch_size: usize,
        tokens: usize,
        duration: Duration,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        state.total_processed += batch_size;
        state.total_tokens += tokens;
        state.batch_processing_times.push(duration);
        Ok(())
    }

    /// Record batch processing
    pub async fn record_batch_processing(
        &self,
        size: usize,
        tokens: usize,
        duration: Duration,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        
        state.total_processed += size;
        state.total_tokens += tokens;
        state.total_processing_time += duration;
        
        state.batch_sizes.push(size);
        state.batch_processing_times.push(duration);
        
        // Calculate tokens per second for this batch
        let tps = tokens as f32 / duration.as_secs_f32();
        state.tokens_per_second.push(tps);
        
        // Keep history bounded
        if state.batch_sizes.len() > 1000 {
            state.batch_sizes.remove(0);
            state.batch_processing_times.remove(0);
            state.tokens_per_second.remove(0);
        }

        Ok(())
    }

    /// Record an error
    pub async fn record_error(&self, error_type: String) {
        let mut state = self.state.write().await;
        state.total_errors += 1;
        
        // Update error type counts
        if let Some(entry) = state.error_types.iter_mut()
            .find(|(t, _)| t == &error_type) {
            entry.1 += 1;
        } else {
            state.error_types.push((error_type, 1));
        }
    }

    /// Get a snapshot of current metrics
    pub async fn snapshot(&self) -> MetricsSnapshot {
        let state = self.state.read().await;
        
        MetricsSnapshot {
            timestamp: Instant::now(),
            uptime: self.start_time.elapsed(),
            gpu_memory_used: state.gpu_memory_allocated.clone(),
            gpu_utilization: state.gpu_utilization.clone(),
            total_processed: state.total_processed,
            total_tokens: state.total_tokens,
            average_tokens_per_second: calculate_average(&state.tokens_per_second),
            average_latency: calculate_average_duration(&state.batch_processing_times),
            p95_latency: calculate_percentile(&state.batch_processing_times, 0.95),
            p99_latency: calculate_percentile(&state.batch_processing_times, 0.99),
            error_rate: if state.total_processed > 0 {
                state.total_errors as f32 / state.total_processed as f32
            } else {
                0.0
            },
            error_types: state.error_types.clone(),
        }
    }
}

// Add proper Default implementation
impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new(Arc::new(EngineConfig::default()))
    }
}

// Helper functions
fn calculate_average(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn calculate_average_duration(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        Duration::default()
    } else {
        Duration::from_secs_f64(
            durations.iter()
                .map(|d| d.as_secs_f64())
                .sum::<f64>() / durations.len() as f64
        )
    }
}

fn calculate_percentile(durations: &[Duration], percentile: f32) -> Duration {
    if durations.is_empty() {
        return Duration::default();
    }
    
    let mut sorted: Vec<_> = durations.to_vec();
    sorted.sort();
    
    let index = ((sorted.len() as f32 - 1.0) * percentile).round() as usize;
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn create_test_collector() -> MetricsCollector {
        let config = Arc::new(EngineConfig::default());
        MetricsCollector::new(config)
    }

    #[tokio::test]
    async fn test_gpu_memory_tracking() {
        let mut collector = create_test_collector().await;
        
        collector.record_gpu_allocation(0, 1024);
        let snapshot = collector.snapshot().await;
        assert_eq!(snapshot.gpu_memory_used[0], 1024);
        
        collector.record_gpu_deallocation(0, 512);
        let snapshot = collector.snapshot().await;
        assert_eq!(snapshot.gpu_memory_used[0], 512);
    }

    #[tokio::test]
    async fn test_batch_processing_metrics() {
        let collector = create_test_collector().await;
        
        collector.record_batch_processing(
            32,
            1000,
            Duration::from_millis(100)
        ).await;
        
        let snapshot = collector.snapshot().await;
        assert_eq!(snapshot.total_processed, 32);
        assert_eq!(snapshot.total_tokens, 1000);
        assert!(snapshot.average_tokens_per_second > 0.0);
    }

    #[tokio::test]
    async fn test_error_tracking() {
        let collector = create_test_collector().await;
        
        collector.record_error("OutOfMemory".to_string()).await;
        collector.record_error("OutOfMemory".to_string()).await;
        collector.record_error("Timeout".to_string()).await;
        
        let snapshot = collector.snapshot().await;
        assert_eq!(snapshot.error_types.len(), 2);
        
        let oom_count = snapshot.error_types.iter()
            .find(|(t, _)| t == "OutOfMemory")
            .map(|(_, c)| c)
            .unwrap();
        assert_eq!(*oom_count, 2);
    }

    #[tokio::test]
    async fn test_latency_percentiles() {
        let collector = create_test_collector().await;
        
        // Record various processing times
        for i in 0..100 {
            collector.record_batch_processing(
                1,
                100,
                Duration::from_millis(i)
            ).await;
        }
        
        let snapshot = collector.snapshot().await;
        assert!(snapshot.p95_latency >= snapshot.average_latency);
        assert!(snapshot.p99_latency >= snapshot.p95_latency);
    }
}