use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, warn};

use crate::{
    error::{EngineError, Result},
    model::ModelRuntime,
    metrics::MetricsCollector,
    types::ProcessingOutput,
};

use super::common::{ProcessingStats, ProcessingResult};
use super::Processor;

/// Handles streaming mode processing
pub struct StreamProcessor {
    runtime: Arc<ModelRuntime>,
    metrics: Arc<Mutex<MetricsCollector>>,
    active_streams: Arc<Mutex<Vec<StreamState>>>,
}

/// State for an active stream
struct StreamState {
    id: usize,
    created_at: std::time::Instant,
    processed_tokens: usize,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new(runtime: Arc<ModelRuntime>, metrics: Arc<Mutex<MetricsCollector>>) -> Self {
        Self {
            runtime,
            metrics,
            active_streams: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a new stream handle
    pub async fn create_handle(&self) -> Result<StreamHandle> {
        let (input_tx, input_rx) = mpsc::channel(100);
        let (output_tx, output_rx) = mpsc::channel(100);

        let stream_id = {
            let mut streams = self.active_streams.lock().await;
            let id = streams.len();
            streams.push(StreamState {
                id,
                created_at: std::time::Instant::now(),
                processed_tokens: 0,
            });
            id
        };

        // Spawn processing task
        let runtime = self.runtime.clone();
        let metrics = self.metrics.clone();
        let active_streams = self.active_streams.clone();

        tokio::spawn(async move {
            Self::process_stream(
                stream_id,
                runtime,
                input_rx,
                output_tx,
                metrics,
                active_streams,
            ).await;
        });

        Ok(StreamHandle::new(stream_id, input_tx, output_rx))
    }

    /// Process stream messages
    async fn process_stream(
        stream_id: usize,
        runtime: Arc<ModelRuntime>,
        mut input_rx: mpsc::Receiver<String>,
        output_tx: mpsc::Sender<ProcessingOutput>,
        metrics: Arc<Mutex<MetricsCollector>>,
        active_streams: Arc<Mutex<Vec<StreamState>>>,
    ) {
        debug!("Starting stream processor {}", stream_id);

        while let Some(input) = input_rx.recv().await {
            let start_time = std::time::Instant::now();
            
            match runtime.process(input).await {
                Ok(output) => {
                    // Update metrics
                    {
                        let mut metrics = metrics.lock().await;
                        metrics.record_processing(
                            stream_id,
                            output.tokens.len(),
                            start_time.elapsed(),
                        ).await;
                    }

                    // Update stream state
                    {
                        let mut streams = active_streams.lock().await;
                        if let Some(stream) = streams.get_mut(stream_id) {
                            stream.processed_tokens += output.tokens.len();
                        }
                    }

                    // Send output
                    if output_tx.send(output).await.is_err() {
                        warn!("Failed to send output for stream {}", stream_id);
                        break;
                    }
                }
                Err(e) => {
                    warn!("Processing error on stream {}: {:?}", stream_id, e);
                    // Continue processing despite errors
                }
            }
        }

        debug!("Stream processor {} shutting down", stream_id);

        // Cleanup stream state
        let mut streams = active_streams.lock().await;
        if let Some(index) = streams.iter().position(|s| s.id == stream_id) {
            streams.remove(index);
        }
    }
}

#[async_trait::async_trait]
impl Processor for StreamProcessor {
    async fn shutdown(&self) -> Result<()> {
        // Wait for all streams to complete
        let mut streams = self.active_streams.lock().await;
        streams.clear();
        Ok(())
    }

    async fn get_stats(&self) -> ProcessingStats {
        let streams = self.active_streams.lock().await;
        
        ProcessingStats {
            active_streams: streams.len(),
            total_processed_tokens: streams.iter()
                .map(|s| s.processed_tokens)
                .sum(),
            uptime: streams.iter()
                .map(|s| s.created_at.elapsed())
                .max()
                .unwrap_or_default(),
        }
    }
}

/// Handle for interacting with a stream
#[derive(Debug)]
pub struct StreamHandle {
    id: usize,
    input_tx: mpsc::Sender<String>,
    output_rx: mpsc::Receiver<ProcessingOutput>,
}

impl StreamHandle {
    /// Create a new stream handle
    pub(crate) fn new(
        id: usize,
        input_tx: mpsc::Sender<String>,
        output_rx: mpsc::Receiver<ProcessingOutput>,
    ) -> Self {
        Self {
            id,
            input_tx,
            output_rx,
        }
    }

    /// Get the stream ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Process input and get output
    pub async fn process(&mut self, input: String) -> Result<ProcessingOutput> {
        self.input_tx.send(input).await.map_err(|_| {
            EngineError::ProcessingError {
                message: "Stream closed".to_string(),
                source: None,
            }
        })?;

        self.output_rx.recv().await.ok_or_else(|| {
            EngineError::ProcessingError {
                message: "Stream closed".to_string(),
                source: None,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn create_test_processor() -> (StreamProcessor, Arc<TestRuntime>) {
        let runtime = Arc::new(TestRuntime::default());
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(
            Arc::new(crate::config::EngineConfig::default())
        )));
        
        let processor = StreamProcessor::new(runtime.clone(), metrics);
        (processor, runtime)
    }

    #[derive(Default)]
    struct TestRuntime {
        processed: Arc<Mutex<usize>>,
    }

    #[async_trait::async_trait]
    impl ModelRuntime for TestRuntime {
        async fn process(&self, input: String) -> Result<ProcessingOutput> {
            let mut processed = self.processed.lock().await;
            *processed += 1;
            
            Ok(ProcessingOutput {
                text: format!("Processed: {}", input),
                tokens: vec![],
                processing_time: Duration::from_millis(100),
            })
        }
    }

    #[tokio::test]
    async fn test_stream_processing() -> Result<()> {
        let (processor, runtime) = create_test_processor().await;
        
        let mut handle = processor.create_handle().await?;
        
        // Process some inputs
        let output = handle.process("test 1".to_string()).await?;
        assert!(output.text.contains("test 1"));
        
        let output = handle.process("test 2".to_string()).await?;
        assert!(output.text.contains("test 2"));
        
        // Verify runtime was called
        let processed = runtime.processed.lock().await;
        assert_eq!(*processed, 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_streams() -> Result<()> {
        let (processor, _) = create_test_processor().await;
        
        let mut handles = vec![];
        for _ in 0..3 {
            handles.push(processor.create_handle().await?);
        }
        
        let stats = processor.get_stats().await;
        assert_eq!(stats.active_streams, 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_stream_shutdown() -> Result<()> {
        let (processor, _) = create_test_processor().await;
        
        let _handle = processor.create_handle().await?;
        processor.shutdown().await?;
        
        let stats = processor.get_stats().await;
        assert_eq!(stats.active_streams, 0);

        Ok(())
    }
}