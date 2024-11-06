use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Snapshot of all metrics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// When these metrics were collected
    pub timestamp: std::time::SystemTime,
    
    /// Processing metrics
    pub processing: ProcessingMetrics,
    
    /// GPU metrics
    pub gpu: GpuMetrics,
    
    /// Model metrics
    pub model: ModelMetrics,
    
    /// Queue metrics (if queue mode is enabled)
    pub queue: Option<QueueMetrics>,
}

/// Processing performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Total requests processed
    pub total_requests: usize,
    
    /// Total tokens generated
    pub total_tokens: usize,
    
    /// Average tokens per second
    pub tokens_per_second: f32,
    
    /// Average latency
    pub average_latency: Duration,
    
    /// 95th percentile latency
    pub p95_latency: Duration,
    
    /// 99th percentile latency
    pub p99_latency: Duration,
    
    /// Number of errors
    pub error_count: usize,
    
    /// Error rate
    pub error_rate: f32,
}

/// GPU performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// Memory used per GPU
    pub memory_used: Vec<usize>,
    
    /// Memory utilization per GPU
    pub memory_utilization: Vec<f32>,
    
    /// Compute utilization per GPU
    pub compute_utilization: Vec<f32>,
    
    /// Power usage per GPU (watts)
    pub power_usage: Vec<f32>,
    
    /// Temperature per GPU (celsius)
    pub temperature: Vec<f32>,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Number of loaded models
    pub loaded_models: usize,
    
    /// Memory used by models
    pub model_memory: usize,
    
    /// Average inference time
    pub average_inference_time: Duration,
    
    /// Model throughput (inferences/sec)
    pub throughput: f32,
    
    /// Cache hit rate
    pub cache_hit_rate: f32,
}

/// Queue performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueMetrics {
    /// Current queue depth
    pub queue_depth: usize,
    
    /// Average wait time in queue
    pub average_wait_time: Duration,
    
    /// 95th percentile wait time
    pub p95_wait_time: Duration,
    
    /// Queue throughput (items/sec)
    pub throughput: f32,
    
    /// Queue statistics by priority
    pub priority_stats: HashMap<String, QueuePriorityStats>,
}

/// Statistics for a specific queue priority level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuePriorityStats {
    /// Number of items at this priority
    pub item_count: usize,
    
    /// Average wait time for this priority
    pub average_wait_time: Duration,
    
    /// Maximum wait time seen
    pub max_wait_time: Duration,
    
    /// Throughput for this priority level
    pub throughput: f32,
}

impl ProcessingMetrics {
    /// Create empty processing metrics
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            total_tokens: 0,
            tokens_per_second: 0.0,
            average_latency: Duration::default(),
            p95_latency: Duration::default(),
            p99_latency: Duration::default(),
            error_count: 0,
            error_rate: 0.0,
        }
    }

    /// Calculate statistics from raw measurements
    pub fn from_measurements(
        requests: usize,
        tokens: usize,
        latencies: &[Duration],
        errors: usize,
        window: Duration,
    ) -> Self {
        let mut sorted_latencies = latencies.to_vec();
        sorted_latencies.sort();

        let p95_idx = (sorted_latencies.len() as f32 * 0.95) as usize;
        let p99_idx = (sorted_latencies.len() as f32 * 0.99) as usize;

        Self {
            total_requests: requests,
            total_tokens: tokens,
            tokens_per_second: if !window.is_zero() {
                tokens as f32 / window.as_secs_f32()
            } else {
                0.0
            },
            average_latency: if !latencies.is_empty() {
                let total: Duration = latencies.iter().sum();
                total / latencies.len() as u32
            } else {
                Duration::default()
            },
            p95_latency: sorted_latencies.get(p95_idx).copied()
                .unwrap_or_default(),
            p99_latency: sorted_latencies.get(p99_idx).copied()
                .unwrap_or_default(),
            error_count: errors,
            error_rate: if requests > 0 {
                errors as f32 / requests as f32
            } else {
                0.0
            },
        }
    }
}

impl Default for ProcessingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processing_metrics_calculation() {
        let latencies = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];

        let metrics = ProcessingMetrics::from_measurements(
            100,  // requests
            1000, // tokens
            &latencies,
            5,    // errors
            Duration::from_secs(10),
        );

        assert_eq!(metrics.total_requests, 100);
        assert_eq!(metrics.total_tokens, 1000);
        assert_eq!(metrics.tokens_per_second, 100.0);
        assert_eq!(metrics.error_rate, 0.05);
        assert_eq!(metrics.p95_latency, Duration::from_millis(50));
    }

    #[test]
    fn test_empty_metrics() {
        let metrics = ProcessingMetrics::new();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.error_rate, 0.0);
        assert_eq!(metrics.average_latency, Duration::default());
    }

    #[test]
    fn test_queue_metrics() {
        let mut priority_stats = HashMap::new();
        priority_stats.insert("high".to_string(), QueuePriorityStats {
            item_count: 10,
            average_wait_time: Duration::from_millis(50),
            max_wait_time: Duration::from_millis(100),
            throughput: 20.0,
        });

        let metrics = QueueMetrics {
            queue_depth: 10,
            average_wait_time: Duration::from_millis(75),
            p95_wait_time: Duration::from_millis(150),
            throughput: 50.0,
            priority_stats,
        };

        assert_eq!(metrics.queue_depth, 10);
        assert!(metrics.priority_stats.contains_key("high"));
    }
}