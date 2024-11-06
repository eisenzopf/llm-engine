use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, mpsc, oneshot};
use tokio::time;
use tracing::{debug, warn, error};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::{
    error::{EngineError, Result},
    model::ModelRuntime,
    metrics::MetricsCollector,
    types::ProcessingOutput,
};

use super::common::{ProcessingStats, ProcessingResult};
use super::Processor;

/// Priority level for queue items
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Medium = 1,
    High = 2,
}

/// A job in the processing queue
#[derive(Debug)]
struct QueueJob {
    id: usize,
    input: String,
    priority: Priority,
    enqueued_at: Instant,
    response_sender: oneshot::Sender<Result<ProcessingOutput>>,
}

/// Compare jobs by priority and enqueue time
impl Ord for QueueJob {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
            .then_with(|| other.enqueued_at.cmp(&self.enqueued_at))
    }
}

impl PartialOrd for QueueJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for QueueJob {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for QueueJob {}

/// Manages the processing queue
pub struct QueueProcessor {
    runtime: Arc<ModelRuntime>,
    metrics: Arc<Mutex<MetricsCollector>>,
    queue: Arc<Mutex<BinaryHeap<QueueJob>>>,
    stats: Arc<RwLock<QueueStats>>,
    max_queue_size: usize,
    next_job_id: Arc<Mutex<usize>>,
    shutdown_signal: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

#[derive(Debug, Default)]
struct QueueStats {
    total_jobs: usize,
    completed_jobs: usize,
    failed_jobs: usize,
    total_processing_time: Duration,
    queue_wait_times: Vec<Duration>,
}

impl QueueProcessor {
    /// Create a new queue processor
    pub fn new(
        runtime: Arc<ModelRuntime>,
        metrics: Arc<Mutex<MetricsCollector>>,
        max_queue_size: usize,
    ) -> Self {
        let processor = Self {
            runtime,
            metrics,
            queue: Arc::new(Mutex::new(BinaryHeap::new())),
            stats: Arc::new(RwLock::new(QueueStats::default())),
            max_queue_size,
            next_job_id: Arc::new(Mutex::new(0)),
            shutdown_signal: Arc::new(Mutex::new(None)),
        };

        // Start the processing worker
        processor.spawn_worker();
        processor
    }

    /// Enqueue a job with specified priority
    pub async fn enqueue_with_priority(
        &self,
        input: String,
        priority: Priority,
    ) -> Result<QueueHandle> {
        // Check queue size limit
        {
            let queue = self.queue.lock().await;
            if queue.len() >= self.max_queue_size {
                return Err(EngineError::QueueError {
                    message: "Queue is full".to_string(),
                    queue_size: queue.len(),
                });
            }
        }

        let (tx, rx) = oneshot::channel();
        let job_id = {
            let mut id = self.next_job_id.lock().await;
            *id += 1;
            *id
        };

        let job = QueueJob {
            id: job_id,
            input,
            priority,
            enqueued_at: Instant::now(),
            response_sender: tx,
        };

        // Add to queue
        {
            let mut queue = self.queue.lock().await;
            queue.push(job);
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_jobs += 1;
        }

        Ok(QueueHandle::new(job_id, rx))
    }

    /// Get current queue size
    pub async fn queue_size(&self) -> usize {
        let queue = self.queue.lock().await;
        queue.len()
    }

    /// Spawn the queue processing worker
    fn spawn_worker(&self) {
        let runtime = self.runtime.clone();
        let queue = self.queue.clone();
        let metrics = self.metrics.clone();
        let stats = self.stats.clone();
        let shutdown_signal = self.shutdown_signal.clone();

        tokio::spawn(async move {
            let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
            *shutdown_signal.lock().await = Some(shutdown_tx);

            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => {
                        debug!("Queue worker received shutdown signal");
                        break;
                    }
                    _ = Self::process_next_job(
                        runtime.clone(),
                        queue.clone(),
                        metrics.clone(),
                        stats.clone(),
                    ) => {}
                }
            }
        });
    }

    /// Process the next job in queue
    async fn process_next_job(
        runtime: Arc<ModelRuntime>,
        queue: Arc<Mutex<BinaryHeap<QueueJob>>>,
        metrics: Arc<Mutex<MetricsCollector>>,
        stats: Arc<RwLock<QueueStats>>,
    ) {
        let job = {
            let mut queue = queue.lock().await;
            queue.pop()
        };

        if let Some(job) = job {
            let wait_time = job.enqueued_at.elapsed();
            let start_time = Instant::now();

            match runtime.process(job.input).await {
                Ok(output) => {
                    // Update stats
                    {
                        let mut stats = stats.write().await;
                        stats.completed_jobs += 1;
                        stats.total_processing_time += start_time.elapsed();
                        stats.queue_wait_times.push(wait_time);
                        
                        // Keep wait times history bounded
                        if stats.queue_wait_times.len() > 1000 {
                            stats.queue_wait_times.remove(0);
                        }
                    }

                    // Update metrics
                    let mut metrics = metrics.lock().await;
                    metrics.record_queue_processing(
                        job.priority as u8,
                        output.tokens.len(),
                        wait_time,
                        start_time.elapsed(),
                    ).await;

                    // Send result
                    if job.response_sender.send(Ok(output)).is_err() {
                        warn!("Failed to send result for job {}", job.id);
                    }
                }
                Err(e) => {
                    // Update failure stats
                    {
                        let mut stats = stats.write().await;
                        stats.failed_jobs += 1;
                    }

                    error!("Processing error for job {}: {:?}", job.id, e);
                    if job.response_sender.send(Err(e)).is_err() {
                        warn!("Failed to send error for job {}", job.id);
                    }
                }
            }
        } else {
            // No jobs in queue, wait a bit
            time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> QueueStatsSnapshot {
        let stats = self.stats.read().await;
        let queue_size = self.queue.lock().await.len();

        let wait_times = &stats.queue_wait_times;
        let avg_wait_time = if !wait_times.is_empty() {
            wait_times.iter().sum::<Duration>() / wait_times.len() as u32
        } else {
            Duration::default()
        };

        let mut sorted_wait_times = wait_times.clone();
        sorted_wait_times.sort();
        let p95_idx = (sorted_wait_times.len() as f32 * 0.95) as usize;
        let p95_wait_time = sorted_wait_times.get(p95_idx).copied()
            .unwrap_or_default();

        QueueStatsSnapshot {
            current_size: queue_size,
            total_jobs: stats.total_jobs,
            completed_jobs: stats.completed_jobs,
            failed_jobs: stats.failed_jobs,
            average_wait_time: avg_wait_time,
            p95_wait_time,
            average_processing_time: if stats.completed_jobs > 0 {
                stats.total_processing_time / stats.completed_jobs as u32
            } else {
                Duration::default()
            },
        }
    }
}

#[async_trait::async_trait]
impl Processor for QueueProcessor {
    async fn shutdown(&self) -> Result<()> {
        // Signal worker to shut down
        if let Some(tx) = self.shutdown_signal.lock().await.take() {
            let _ = tx.send(());
        }

        // Wait for remaining jobs to complete
        let mut retries = 0;
        while self.queue_size().await > 0 && retries < 10 {
            time::sleep(Duration::from_millis(100)).await;
            retries += 1;
        }

        Ok(())
    }

    async fn get_stats(&self) -> ProcessingStats {
        let queue_stats = self.get_queue_stats().await;
        ProcessingStats {
            active_streams: 0,
            total_processed_tokens: 0, // TODO: Track this in QueueStats
            uptime: Duration::from_secs(0), // TODO: Track this
            current_queue_size: Some(queue_stats.current_size),
            average_wait_time: Some(queue_stats.average_wait_time),
            average_processing_time: Some(queue_stats.average_processing_time),
        }
    }
}

/// Snapshot of queue statistics
#[derive(Debug, Clone)]
pub struct QueueStatsSnapshot {
    pub current_size: usize,
    pub total_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub average_wait_time: Duration,
    pub p95_wait_time: Duration,
    pub average_processing_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct TestRuntime {
        processed: Arc<Mutex<usize>>,
    }

    impl TestRuntime {
        fn new() -> Self {
            Self {
                processed: Arc::new(Mutex::new(0)),
            }
        }
    }

    #[async_trait::async_trait]
    impl ModelRuntime for TestRuntime {
        async fn process(&self, input: String) -> Result<ProcessingOutput> {
            let mut processed = self.processed.lock().await;
            *processed += 1;

            Ok(ProcessingOutput {
                text: format!("Processed: {}", input),
                tokens: vec!["test".to_string()],
                processing_time: Duration::from_millis(100),
            })
        }
    }

    #[tokio::test]
    async fn test_queue_processing() -> Result<()> {
        let runtime = Arc::new(TestRuntime::new());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(
            Arc::new(crate::config::EngineConfig::default())
        )));

        let processor = QueueProcessor::new(runtime.clone(), metrics, 100);

        // Enqueue some jobs with different priorities
        let handle1 = processor.enqueue_with_priority(
            "test1".to_string(),
            Priority::High
        ).await?;
        let handle2 = processor.enqueue_with_priority(
            "test2".to_string(),
            Priority::Low
        ).await?;

        // Wait for results
        let result1 = handle1.await?;
        let result2 = handle2.await?;

        assert!(result1.text.contains("test1"));
        assert!(result2.text.contains("test2"));

        // Verify processing order (high priority should complete first)
        assert!(result1.processing_time < result2.processing_time);

        Ok(())
    }

    #[tokio::test]
    async fn test_queue_size_limit() -> Result<()> {
        let runtime = Arc::new(TestRuntime::new());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(
            Arc::new(crate::config::EngineConfig::default())
        )));

        let processor = QueueProcessor::new(runtime, metrics, 2);

        // Fill queue
        processor.enqueue_with_priority("test1".to_string(), Priority::Medium).await?;
        processor.enqueue_with_priority("test2".to_string(), Priority::Medium).await?;

        // Should fail when queue is full
        let result = processor.enqueue_with_priority(
            "test3".to_string(),
            Priority::Medium
        ).await;
        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_priority_ordering() -> Result<()> {
        let runtime = Arc::new(TestRuntime::new());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(
            Arc::new(crate::config::EngineConfig::default())
        )));

        let processor = QueueProcessor::new(runtime, metrics, 100);

        // Enqueue low priority first
        let low = processor.enqueue_with_priority(
            "low".to_string(),
            Priority::Low
        ).await?;
        
        // Then high priority
        let high = processor.enqueue_with_priority(
            "high".to_string(),
            Priority::High
        ).await?;

        // High priority should complete first
        let high_result = high.await?;
        let low_result = low.await?;

        assert!(high_result.text.contains("high"));
        assert!(low_result.text.contains("low"));

        Ok(())
    }

    #[tokio::test]
    async fn test_queue_stats() -> Result<()> {
        let runtime = Arc::new(TestRuntime::new());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(
            Arc::new(crate::config::EngineConfig::default())
        )));

        let processor = QueueProcessor::new(runtime, metrics, 100);

        // Process some jobs
        for i in 0..5 {
            let handle = processor.enqueue_with_priority(
                format!("test{}", i),
                Priority::Medium
            ).await?;
            handle.await?;
        }

        let stats = processor.get_queue_stats().await;
        assert_eq!(stats.total_jobs, 5);
        assert_eq!(stats.completed_jobs, 5);
        assert_eq!(stats.failed_jobs, 0);
        assert!(stats.average_wait_time > Duration::from_nanos(0));
        assert!(stats.average_processing_time > Duration::from_nanos(0));
        assert!(stats.p95_wait_time >= stats.average_wait_time);

        Ok(())
    }

    #[tokio::test]
    async fn test_shutdown() -> Result<()> {
        let runtime = Arc::new(TestRuntime::new());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(
            Arc::new(crate::config::EngineConfig::default())
        )));

        let processor = QueueProcessor::new(runtime, metrics, 100);

        // Enqueue some jobs
        for i in 0..3 {
            processor.enqueue_with_priority(
                format!("test{}", i),
                Priority::Medium
            ).await?;
        }

        // Shutdown should wait for jobs to complete
        processor.shutdown().await?;

        let stats = processor.get_queue_stats().await;
        assert_eq!(stats.completed_jobs, 3);
        assert_eq!(stats.current_size, 0);

        Ok(())
    }
}

/// Handle for awaiting queue job completion
#[derive(Debug)]
pub struct QueueHandle {
    id: usize,
    receiver: oneshot::Receiver<Result<ProcessingOutput>>,
}

impl QueueHandle {
    /// Create a new queue handle
    pub(crate) fn new(
        id: usize,
        receiver: oneshot::Receiver<Result<ProcessingOutput>>,
    ) -> Self {
        Self { id, receiver }
    }

    /// Get the job ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Wait for the job to complete
    pub async fn await(self) -> Result<ProcessingOutput> {
        self.receiver.await
            .map_err(|_| EngineError::ProcessingError {
                message: "Job cancelled".to_string(),
                source: None,
            })?
    }
}