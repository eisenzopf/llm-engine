use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use anyhow::Result;
use dashmap::DashMap;

use crate::{
    config::EngineConfig,
    error::EngineError,
    gpu::GpuManager,
    metrics::MetricsCollector,
    processing::{ProcessingOutput, StreamHandle, QueueHandle},
};

use super::{
    loader::ModelLoader,
    runtime::ModelRuntime,
};

/// Manages model loading and processing across GPUs
pub struct ModelManager {
    config: Arc<EngineConfig>,
    gpu_manager: Arc<GpuManager>,
    metrics: Arc<Mutex<MetricsCollector>>,
    
    // Runtime state
    runtimes: DashMap<usize, Arc<ModelRuntime>>,
    loader: Arc<ModelLoader>,
    
    // Processing queues
    processing_queues: Arc<RwLock<Vec<ProcessingQueue>>>,
    next_job_id: Arc<Mutex<usize>>,
}

/// Represents a processing queue for a GPU
#[derive(Debug)]
struct ProcessingQueue {
    device_id: usize,
    pending_jobs: Vec<PendingJob>,
    current_batch: Option<BatchInProgress>,
}

/// A job waiting to be processed
#[derive(Debug)]
struct PendingJob {
    id: usize,
    input: String,
    response_sender: tokio::sync::oneshot::Sender<Result<ProcessingOutput>>,
}

/// A batch currently being processed
#[derive(Debug)]
struct BatchInProgress {
    inputs: Vec<String>,
    start_time: std::time::Instant,
    response_senders: Vec<tokio::sync::oneshot::Sender<Result<ProcessingOutput>>>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub device_id: usize,
    pub model_type: String,
    pub parameters: usize,
    pub max_sequence_length: usize,
}

impl ModelManager {
    /// Create a new model manager
    pub async fn new(
        config: Arc<EngineConfig>,
        gpu_manager: Arc<GpuManager>,
    ) -> Result<Self> {
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(config.clone())));
        let loader = Arc::new(ModelLoader::new(config.clone()));

        let manager = Self {
            config: config.clone(),
            gpu_manager,
            metrics,
            runtimes: DashMap::new(),
            loader,
            processing_queues: Arc::new(RwLock::new(Vec::new())),
            next_job_id: Arc::new(Mutex::new(0)),
        };

        // Initialize queues for each GPU
        let mut queues = manager.processing_queues.write().await;
        for device_id in 0..manager.gpu_manager.available_gpus() {
            queues.push(ProcessingQueue {
                device_id,
                pending_jobs: Vec::new(),
                current_batch: None,
            });
        }

        // Load models based on processing mode
        manager.initialize_models().await?;

        Ok(manager)
    }

    async fn initialize_models(&self) -> Result<()> {
        let available_gpus = self.gpu_manager.available_gpus();
        let mut handles = Vec::new();

        for device_id in 0..available_gpus {
            let config = self.config.clone();
            let loader = self.loader.clone();
            let metrics = self.metrics.clone();

            let handle = tokio::spawn(async move {
                let runtime = loader.load_model(device_id).await?;
                Ok::<_, anyhow::Error>((device_id, Arc::new(runtime)))
            });
            handles.push(handle);
        }

        for handle in handles {
            let (device_id, runtime) = handle.await.map_err(|e| EngineError::InitializationError {
                message: format!("Failed to join model loading task: {}", e),
                source: None,
            })??;
            self.runtimes.insert(device_id, runtime);
        }

        Ok(())
    }

    /// Initialize streaming mode
    async fn initialize_streaming_mode(&self) -> Result<()> {
        // Load model on each GPU
        for device_id in 0..self.gpu_manager.available_gpus() {
            let runtime = self.loader.load_model(device_id).await?;
            self.runtimes.insert(device_id, Arc::new(runtime));
        }
        Ok(())
    }

    /// Initialize batch mode
    async fn initialize_batch_mode(&self) -> Result<()> {
        // Similar to streaming, but configure for batch processing
        for device_id in 0..self.gpu_manager.available_gpus() {
            let runtime = self.loader.load_model(device_id).await?;
            self.runtimes.insert(device_id, Arc::new(runtime));
            
            // Start batch processing worker
            self.spawn_batch_worker(device_id).await?;
        }
        Ok(())
    }

    /// Initialize queue mode
    async fn initialize_queue_mode(&self) -> Result<()> {
        // Load model and start queue workers
        for device_id in 0..self.gpu_manager.available_gpus() {
            let runtime = self.loader.load_model(device_id).await?;
            self.runtimes.insert(device_id, Arc::new(runtime));
            
            // Start queue processing worker
            self.spawn_queue_worker(device_id).await?;
        }
        Ok(())
    }

    /// Create stream handles for processing
    pub async fn create_stream_handles(&self) -> Result<Vec<StreamHandle>> {
        let mut handles = Vec::new();
        for device_id in 0..self.gpu_manager.available_gpus() {
            let runtime = self.runtimes.get(&device_id)
                .ok_or_else(|| EngineError::InitializationError {
                    message: format!("Runtime not initialized for device {}", device_id),
                    source: None,
                })?;

            let (input_tx, input_rx) = tokio::sync::mpsc::channel(100);
            let (output_tx, output_rx) = tokio::sync::mpsc::channel(100);

            // Spawn processing task
            let runtime = runtime.clone();
            let metrics = self.metrics.clone();

            tokio::spawn(async move {
                self.process_stream(runtime, input_rx, output_tx, metrics).await;
            });

            handles.push(StreamHandle::new(device_id, input_tx, output_rx));
        }

        Ok(handles)
    }

    /// Process a batch of inputs
    pub async fn process_batch(&self, inputs: Vec<String>) -> Result<Vec<ProcessingOutput>> {
        if self.config.processing.mode != ProcessingMode::Batch {
            return Err(EngineError::ConfigurationError { 
                message: "Not configured for batch mode".to_string(),
                parameter: "mode".to_string(),
            }.into());
        }

        let batch_size = self.config.processing.batch_size.unwrap_or(32);
        let mut results = Vec::with_capacity(inputs.len());

        for chunk in inputs.chunks(batch_size) {
            let device_id = self.select_device_for_batch().await?;
            let runtime = self.runtimes.get(&device_id)
                .ok_or_else(|| EngineError::InitializationError {
                    message: format!("Runtime not initialized for device {}", device_id),
                    source: None,
                })?;

            let chunk_results = runtime.process_batch(chunk.to_vec()).await?;
            results.extend(chunk_results);
        }

        Ok(results)
    }

    /// Enqueue an input for processing
    pub async fn enqueue(&self, input: String) -> Result<QueueHandle> {
        if self.config.processing.mode != ProcessingMode::Queue {
            return Err(EngineError::ConfigurationError { 
                message: "Not configured for queue mode".to_string(),
                parameter: "mode".to_string(),
            }.into());
        }

        let job_id = {
            let mut id = self.next_job_id.lock().await;
            *id += 1;
            *id
        };

        let (tx, rx) = tokio::sync::oneshot::channel();
        let job = PendingJob {
            id: job_id,
            input,
            response_sender: tx,
        };

        // Add to queue
        let device_id = self.select_device_for_job().await?;
        let mut queues = self.processing_queues.write().await;
        queues[device_id].pending_jobs.push(job);

        Ok(QueueHandle::new(job_id, rx))
    }

    /// Select a device for batch processing
    async fn select_device_for_batch(&self) -> Result<usize> {
        // Simple round-robin for now
        Ok(0) // TODO: Implement proper device selection
    }

    /// Select a device for a new job
    async fn select_device_for_job(&self) -> Result<usize> {
        // Simple round-robin for now
        Ok(0) // TODO: Implement proper device selection
    }

    /// Spawn a batch processing worker
    async fn spawn_batch_worker(&self, device_id: usize) -> Result<()> {
        // Implementation details here
        Ok(())
    }

    /// Spawn a queue processing worker
    async fn spawn_queue_worker(&self, device_id: usize) -> Result<()> {
        // Implementation details here
        Ok(())
    }

    /// Stream processing loop
    async fn process_stream(
        runtime: Arc<ModelRuntime>,
        mut input_rx: tokio::sync::mpsc::Receiver<String>,
        output_tx: tokio::sync::mpsc::Sender<ProcessingOutput>,
        metrics: Arc<Mutex<MetricsCollector>>,
    ) {
        while let Some(input) = input_rx.recv().await {
            let start_time = std::time::Instant::now();
            match runtime.process(input).await {
                Ok(output) => {
                    // Update metrics
                    if let Ok(mut metrics) = metrics.try_lock() {
                        metrics.record_stream_processing(
                            output.tokens.len(),
                            start_time.elapsed(),
                        ).await.ok();
                    }

                    if output_tx.send(output).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("Processing error: {:?}", e);
                    // Send error output
                    let error_output = ProcessingOutput::error(e.to_string());
                    if output_tx.send(error_output).await.is_err() {
                        break;
                    }
                }
            }
        }
    }

    /// Shutdown and cleanup
    pub async fn shutdown(&self) -> Result<()> {
        // Shutdown all runtimes
        for runtime in self.runtimes.iter() {
            runtime.value().shutdown().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn create_test_manager() -> Result<ModelManager> {
        let config = Arc::new(EngineConfig::default());
        let gpu_manager = Arc::new(GpuManager::new(
            config.clone(),
            Arc::new(Mutex::new(MetricsCollector::new(config.clone()))),
        ).await?);
        
        ModelManager::new(config, gpu_manager).await
    }

    #[tokio::test]
    async fn test_streaming_mode() {
        let manager = create_test_manager().await.unwrap();
        let handles = manager.create_stream_handles().await;
        assert!(handles.is_ok());
    }

    #[tokio::test]
    async fn test_batch_mode() {
        let manager = create_test_manager().await.unwrap();
        let results = manager.process_batch(vec!["test".to_string()]).await;
        assert!(results.is_ok());
    }

    #[tokio::test]
    async fn test_queue_mode() {
        let manager = create_test_manager().await.unwrap();
        let handle = manager.enqueue("test".to_string()).await;
        assert!(handle.is_ok());
    }

    #[tokio::test]
    async fn test_shutdown() {
        let manager = create_test_manager().await.unwrap();
        assert!(manager.shutdown().await.is_ok());
    }
}