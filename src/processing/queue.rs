// Location: src/processing/queue.rs

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::time;
use tracing::{debug, warn, error};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::{
    config::EngineConfig,
    error::{EngineError, Result},
    model::{ModelRuntime, LlamaTokenizer},
    metrics::MetricsCollector,
    types::ProcessingOutput,
};

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
    tokens: Option<Vec<u32>>,  // Pre-tokenized input if available
    response_sender: tokio::sync::oneshot::Sender<Result<ProcessingOutput>>,
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
    tokenizer: Arc<LlamaTokenizer>,
    metrics: Arc<Mutex<MetricsCollector>>,
    queue: Arc<Mutex<BinaryHeap<QueueJob>>>,
    stats: Arc<RwLock<QueueStats>>,
    config: Arc<EngineConfig>,
    next_job_id: Arc<Mutex<usize>>,
    shutdown_signal: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}

#[derive(Debug, Default)]
struct QueueStats {
    total_jobs: usize,
    completed_jobs: usize,
    failed_jobs: usize,
    total_processing_time: Duration,
    queue_wait_times: Vec<QueueWaitRecord>,
    priority_stats: Vec<PriorityStats>,
}

#[derive(Debug)]
struct QueueWaitRecord {
    priority: Priority,
    wait_time: Duration,
    sequence_length: usize,
    timestamp: Instant,
}

#[derive(Debug, Clone)]
struct PriorityStats {
    priority: Priority,
    total_jobs: usize,
    completed_jobs: usize,
    total_tokens: usize,
    average_wait_time: Duration,
    max_wait_time: Duration,
}

impl Default for PriorityStats {
    fn default() -> Self {
        Self {
            priority: Priority::Medium, // Default priority
            total_jobs: 0,
            completed_jobs: 0,
            total_tokens: 0,
            average_wait_time: Duration::default(),
            max_wait_time: Duration::default(),
        }
    }
}

impl QueueProcessor {
    /// Create a new queue processor
    pub fn new(
        runtime: Arc<ModelRuntime>,
        tokenizer: Arc<LlamaTokenizer>,
        metrics: Arc<Mutex<MetricsCollector>>,
        config: Arc<EngineConfig>,
    ) -> Self {
        let processor = Self {
            runtime,
            tokenizer,
            metrics,
            queue: Arc::new(Mutex::new(BinaryHeap::new())),
            stats: Arc::new(RwLock::new(QueueStats::default())),
            config,
            next_job_id: Arc::new(Mutex::new(0)),
            shutdown_signal: Arc::new(Mutex::new(None)),
        };

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
            if queue.len() >= self.config.processing.queue_size.unwrap_or(1000) {
                return Err(EngineError::QueueError {
                    message: format!(
                        "Queue is full (max size: {})",
                        self.config.processing.queue_size.unwrap_or(1000)
                    ),
                    queue_size: queue.len(),
                });
            }
        }

        let (tx, rx) = tokio::sync::oneshot::channel();
        let job_id = {
            let mut id = self.next_job_id.lock().await;
            *id += 1;
            *id
        };

        // Pre-tokenize input if possible
        let tokens = match self.tokenizer.encode(&input, true).await {
            Ok(tokens) => Some(tokens),
            Err(e) => {
                warn!("Failed to pre-tokenize input: {:?}", e);
                None
            }
        };

        let job = QueueJob {
            id: job_id,
            input,
            priority,
            enqueued_at: Instant::now(),
            tokens,
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
            if let Some(priority_stats) = stats.priority_stats.iter_mut()
                .find(|s| s.priority == priority) {
                priority_stats.total_jobs += 1;
            } else {
                stats.priority_stats.push(PriorityStats {
                    priority,
                    total_jobs: 1,
                    ..Default::default()
                });
            }
        }

        Ok(QueueHandle::new(job_id, rx))
    }

    /// Spawn the queue processing worker
    fn spawn_worker(&self) {
        let runtime = self.runtime.clone();
        let tokenizer = self.tokenizer.clone();
        let queue = self.queue.clone();
        let metrics = self.metrics.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();
        let shutdown_signal = self.shutdown_signal.clone();

        tokio::spawn(async move {
            let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
            *shutdown_signal.lock().await = Some(shutdown_tx);

            let mut batch = Vec::new();
            let mut batch_timeout = time::interval(Duration::from_millis(100));

            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => {
                        debug!("Queue worker received shutdown signal");
                        break;
                    }
                    _ = batch_timeout.tick() => {
                        if !batch.is_empty() {
                            Self::process_batch(
                                batch.drain(..).collect(),
                                runtime.clone(),
                                tokenizer.clone(),
                                metrics.clone(),
                                stats.clone(),
                                config.clone(),
                            ).await;
                        }
                    }
                    Some(job) = Self::get_next_job(queue.clone()) => {
                        batch.push(job);
                        if batch.len() >= config.processing.batch_size.unwrap_or(32) {
                            Self::process_batch(
                                batch.drain(..).collect(),
                                runtime.clone(),
                                tokenizer.clone(),
                                metrics.clone(),
                                stats.clone(),
                                config.clone(),
                            ).await;
                        }
                    }
                }
            }
        });
    }

    /// Get the next job from the queue
    async fn get_next_job(queue: Arc<Mutex<BinaryHeap<QueueJob>>>) -> Option<QueueJob> {
        let mut queue = queue.lock().await;
        queue.pop()
    }

    /// Process a batch of jobs
    async fn process_batch(
        jobs: Vec<QueueJob>,
        runtime: Arc<ModelRuntime>,
        tokenizer: Arc<LlamaTokenizer>,
        metrics: Arc<Mutex<MetricsCollector>>,
        stats: Arc<RwLock<QueueStats>>,
        config: Arc<EngineConfig>,
    ) {
        let start_time = Instant::now();

        // Collect inputs and pre-tokenized tokens
        let mut inputs = Vec::with_capacity(jobs.len());
        let mut pre_tokenized = Vec::with_capacity(jobs.len());
        
        for job in &jobs {
            let input = if !job.input.starts_with("### Instruction:") {
                config.generation.instruction_template.replace("{}", &job.input)
            } else {
                job.input.clone()
            };
            
            inputs.push(input);
            if let Some(tokens) = &job.tokens {
                pre_tokenized.push(tokens.clone());
            }
        }

        // Tokenize remaining inputs if needed
        let token_ids = if pre_tokenized.len() == jobs.len() {
            pre_tokenized
        } else {
            match tokenizer.encode_batch(&inputs, true).await {
                Ok(encoding) => encoding.token_ids,
                Err(e) => {
                    error!("Batch tokenization failed: {:?}", e);
                    for job in jobs {
                        let _ = job.response_sender.send(Err(e.clone()));
                    }
                    return;
                }
            }
        };

        // Process through model
        match runtime.process_batch_with_tokens(&token_ids).await {
            Ok(outputs) => {
                // Update stats and send results
                let mut stats = stats.write().await;
                
                for ((job, output), tokens) in jobs.into_iter()
                    .zip(outputs)
                    .zip(token_ids) 
                {
                    let wait_time = job.enqueued_at.elapsed();
                    stats.queue_wait_times.push(QueueWaitRecord {
                        priority: job.priority,
                        wait_time,
                        sequence_length: tokens.len(),
                        timestamp: Instant::now(),
                    });

                    if let Some(priority_stats) = stats.priority_stats.iter_mut()
                        .find(|s| s.priority == job.priority) 
                    {
                        priority_stats.completed_jobs += 1;
                        priority_stats.total_tokens += tokens.len();
                        priority_stats.avg_wait_time = (
                            priority_stats.avg_wait_time.as_nanos() * (priority_stats.completed_jobs - 1) as u128 +
                            wait_time.as_nanos()
                        ) as u128 / priority_stats.completed_jobs as u128;
                        priority_stats.max_wait_time = priority_stats.max_wait_time.max(wait_time);
                    }

                    let _ = job.response_sender.send(Ok(output));
                }

                stats.completed_jobs += jobs.len();
                stats.total_processing_time += start_time.elapsed();

                // Update metrics
                let mut metrics = metrics.lock().await;
                metrics.record_queue_batch(
                    jobs.len(),
                    token_ids.iter().map(|t| t.len()).sum(),
                    start_time.elapsed(),
                ).await;
            }
            Err(e) => {
                let mut stats = stats.write().await;
                stats.failed_jobs += jobs.len();
                for job in jobs {
                    let _ = job.response_sender.send(Err(e.clone()));
                }
            }
        }
    }

    /// Get current queue size
    pub async fn queue_size(&self) -> usize {
        let queue = self.queue.lock().await;
        queue.len()
    }

    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> QueueStatsSnapshot {
        let stats = self.stats.read().await;
        let queue_size = self.queue.lock().await.len();

        QueueStatsSnapshot {
            current_size: queue_size,
            total_jobs: stats.total_jobs,
            completed_jobs: stats.completed_jobs,
            failed_jobs: stats.failed_jobs,
            priority_stats: stats.priority_stats.clone(),
        }
    }

    /// Shutdown the processor
    pub async fn shutdown(&self) -> Result<()> {
        // Signal worker to shut down
        if let Some(tx) = self.shutdown_signal.lock().await.take() {
            let _ = tx.send(());
        }

        // Wait for queue to empty
        while self.queue_size().await > 0 {
            time::sleep(Duration::from_millis(100)).await;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct QueueStatsSnapshot {
    pub current_size: usize,
    pub total_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub priority_stats: Vec<PriorityStats>,
}

/// Handle for awaiting queue job completion
#[derive(Debug)]
pub struct QueueHandle {
    id: usize,
    receiver: tokio::sync::oneshot::Receiver<Result<ProcessingOutput>>,
}

impl QueueHandle {
    /// Create a new queue handle
    pub(crate) fn new(
        id: usize,
        receiver: tokio::sync::oneshot::Receiver<Result<ProcessingOutput>>,
    ) -> Self {
        Self { id, receiver }
    }

    /// Get the job ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Wait for the job to complete
    pub async fn wait(self) -> Result<ProcessingOutput> {
        self.receiver.await.map_err(|_| EngineError::QueueError {
            message: "Job cancelled".to_string(),
            queue_size: 0,
        })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn create_test_processor() -> Result<(QueueProcessor, Arc<TestRuntime>)> {
        let runtime = Arc::new(TestRuntime::default());
        let tokenizer = Arc::new(LlamaTokenizer::from_file("models/tokenizer.json").await?);
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(Arc::new(EngineConfig::default()))));
        let config = Arc::new(EngineConfig::default());
        
        let processor = QueueProcessor::new(
            runtime.clone(),
            tokenizer,
            metrics,
            config,
        );

        Ok((processor, runtime))
    }

    #[derive(Default)]
    struct TestRuntime {
        processed: Arc<Mutex<usize>>,
    }

    #[async_trait::async_trait]
    impl ModelRuntime for TestRuntime {
        async fn process_batch_with_tokens(
            &self,
            tokens: &[Vec<u32>],
        ) -> Result<Vec<ProcessingOutput>> {
            let mut processed = self.processed.lock().await;
            *processed += tokens.len();
            
            Ok(tokens.iter().map(|t| ProcessingOutput {
                text: "Test output".to_string(),
                tokens: t.clone(),
                processing_time: Duration::from_millis(100),
            }).collect())
        }
    }

    #[tokio::test]
    async fn test_queue_processing() -> Result<()> {
        let (processor, runtime) = create_test_processor().await?;
        
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

        assert!(!result1.text.is_empty());
        assert!(!result2.text.is_empty());
        
        let processed = *runtime.processed.lock().await;
        assert_eq!(processed, 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_queue_priority() -> Result<()> {
        let (processor, _) = create_test_processor().await?;
        
        // Enqueue low priority first
        let low = processor.enqueue_with_priority(
            "low priority".to_string(),
            Priority::Low
        ).await?;
        
        // Then high priority
        let high = processor.enqueue_with_priority(
            "high priority".to_string(),
            Priority::High
        ).await?;

        // High priority should complete first
        let (high_result, low_result) = tokio::join!(high.await, low.await);
        
        let stats = processor.get_queue_stats().await;
        assert_eq!(stats.completed_jobs, 2);
        assert!(stats.priority_stats.iter().any(|s| s.priority == Priority::High));
        assert!(stats.priority_stats.iter().any(|s| s.priority == Priority::Low));

        high_result?;
        low_result?;
        
        Ok(())
    }

    #[tokio::test]
    async fn test_queue_size_limit() -> Result<()> {
        let mut config = EngineConfig::default();
        config.processing.queue_size = Some(2);
        
        let runtime = Arc::new(TestRuntime::default());
        let tokenizer = Arc::new(LlamaTokenizer::from_file("models/tokenizer.json").await?);
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(Arc::new(config.clone()))));
        let processor = QueueProcessor::new(
            runtime,
            tokenizer,
            metrics,
            Arc::new(config),
        );

        // Fill queue
        processor.enqueue_with_priority("test1".to_string(), Priority::Medium).await?;
        processor.enqueue_with_priority("test2".to_string(), Priority::Medium).await?;

        // Should fail when queue is full
        let result = processor.enqueue_with_priority(
            "test3".to_string(),
            Priority::Medium
        ).await;
        
        assert!(matches!(result, Err(EngineError::QueueError { .. })));
        
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_processing() -> Result<()> {
        let mut config = EngineConfig::default();
        config.processing.batch_size = Some(2);
        
        let (processor, runtime) = create_test_processor().await?;

        // Submit multiple jobs
        let handles: Vec<_> = (0..4).map(|i| {
            processor.enqueue_with_priority(
                format!("test{}", i),
                Priority::Medium
            )
        }).collect();

        // Wait for all results
        for handle in futures::future::join_all(handles).await {
            handle?.await?;
        }

        // Should have been processed in batches
        let processed = *runtime.processed.lock().await;
        assert_eq!(processed, 4);
        
        let stats = processor.get_queue_stats().await;
        assert_eq!(stats.completed_jobs, 4);

        Ok(())
    }

    #[tokio::test]
    async fn test_shutdown() -> Result<()> {
        let (processor, _) = create_test_processor().await?;
        
        // Submit some jobs
        let handles: Vec<_> = (0..3).map(|i| {
            processor.enqueue_with_priority(
                format!("test{}", i),
                Priority::Medium
            )
        }).collect();

        // Initiate shutdown
        let shutdown = processor.shutdown();
        
        // Wait for all results
        for handle in futures::future::join_all(handles).await {
            handle?.await?;
        }

        // Wait for shutdown
        shutdown.await?;
        
        assert_eq!(processor.queue_size().await, 0);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_error_handling() -> Result<()> {
        let (processor, runtime) = create_test_processor().await?;
        
        // Force an error by making the runtime fail
        {
            let mut processed = runtime.processed.lock().await;
            *processed = usize::MAX;  // This will cause next process to fail
        }

        let handle = processor.enqueue_with_priority(
            "error test".to_string(),
            Priority::High
        ).await?;

        assert!(handle.await.is_err());
        
        let stats = processor.get_queue_stats().await;
        assert_eq!(stats.failed_jobs, 1);
        
        Ok(())
    }
}