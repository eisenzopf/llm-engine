// Location: src/processing/stream.rs

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, warn};
use anyhow::Result;

use crate::{
    config::EngineConfig,
    error::EngineError,
    model::{ModelRuntime, LlamaTokenizer},
    metrics::MetricsCollector,
    types::ProcessingOutput,
};

/// Handles streaming mode processing
pub struct StreamProcessor {
    runtime: Arc<ModelRuntime>,
    tokenizer: Arc<LlamaTokenizer>,
    metrics: Arc<Mutex<MetricsCollector>>,
    active_streams: Arc<Mutex<Vec<StreamState>>>,
    config: Arc<EngineConfig>,
}

/// State for an active stream
#[derive(Debug)]
struct StreamState {
    id: usize,
    created_at: std::time::Instant,
    processed_tokens: usize,
    current_context: Vec<u32>,
    device_id: usize,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new(
        runtime: Arc<ModelRuntime>,
        tokenizer: Arc<LlamaTokenizer>,
        metrics: Arc<Mutex<MetricsCollector>>,
        config: Arc<EngineConfig>,
    ) -> Self {
        Self {
            runtime,
            tokenizer,
            metrics,
            active_streams: Arc::new(Mutex::new(Vec::new())),
            config,
        }
    }

    /// Create a new stream handle
    pub async fn create_handle(&self) -> Result<StreamHandle> {
        let (input_tx, input_rx) = mpsc::channel(self.config.processing.concurrency);
        let (output_tx, output_rx) = mpsc::channel(100);

        let stream_id = {
            let mut streams = self.active_streams.lock().await;
            let id = streams.len();
            streams.push(StreamState {
                id,
                created_at: std::time::Instant::now(),
                processed_tokens: 0,
                current_context: Vec::new(),
                device_id: self.runtime.device_index(),
            });
            id
        };

        // Spawn processing task
        let runtime = self.runtime.clone();
        let tokenizer = self.tokenizer.clone();
        let metrics = self.metrics.clone();
        let active_streams = self.active_streams.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            Self::process_stream(
                stream_id,
                runtime,
                tokenizer,
                input_rx,
                output_tx,
                metrics,
                active_streams,
                config,
            ).await;
        });

        Ok(StreamHandle::new(stream_id, input_tx, output_rx))
    }

    /// Process stream messages
    pub async fn process_stream(
        &self,
        runtime: Arc<ModelRuntime>,
        mut input_rx: mpsc::Receiver<String>,
        output_tx: mpsc::Sender<ProcessingOutput>,
        metrics: Arc<Mutex<MetricsCollector>>,
    ) {
        while let Some(input) = input_rx.recv().await {
            let start_time = std::time::Instant::now();
            
            match runtime.process(&input).await {
                Ok(output) => {
                    // Update metrics
                    if let Ok(mut metrics) = metrics.try_lock() {
                        metrics.record_batch_processing(
                            1,
                            output.tokens.len(),
                            start_time.elapsed(),
                        );
                    }

                    if output_tx.send(output).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    if output_tx.send(ProcessingOutput::new_error(&e.to_string())).await.is_err() {
                        break;
                    }
                }
            }
        }
    }

    /// Get the current number of active streams
    pub async fn active_stream_count(&self) -> usize {
        self.active_streams.lock().await.len()
    }

    /// Get statistics for all active streams
    pub async fn get_stream_stats(&self) -> Vec<StreamStats> {
        let streams = self.active_streams.lock().await;
        streams.iter().map(|state| StreamStats {
            stream_id: state.id,
            uptime: state.created_at.elapsed(),
            processed_tokens: state.processed_tokens,
            device_id: state.device_id,
            context_tokens: state.current_context.len(),
        }).collect()
    }

    /// Shutdown all streams
    pub async fn shutdown(&self) -> Result<()> {
        let mut streams = self.active_streams.lock().await;
        streams.clear();
        Ok(())
    }
}

/// Statistics for a single stream
#[derive(Debug, Clone)]
pub struct StreamStats {
    pub stream_id: usize,
    pub uptime: std::time::Duration,
    pub processed_tokens: usize,
    pub device_id: usize,
    pub context_tokens: usize,
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
            EngineError::StreamError {
                stream_id: self.id,
                message: "Stream closed".to_string(),
            }
        })?;

        self.output_rx.recv().await.ok_or_else(|| {
            EngineError::StreamError {
                stream_id: self.id,
                message: "Stream closed".to_string(),
            }
        })?
    }

    /// Close the stream
    pub async fn close(self) -> Result<()> {
        // Dropping self will close the channels
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn create_test_processor() -> Result<(StreamProcessor, Arc<TestRuntime>)> {
        let runtime = Arc::new(TestRuntime::default());
        let tokenizer = Arc::new(LlamaTokenizer::from_file("models/tokenizer.json").await?);
        let metrics = Arc::new(Mutex::new(MetricsCollector::new(Arc::new(EngineConfig::default()))));
        let config = Arc::new(EngineConfig::default());
        
        let processor = StreamProcessor::new(
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
        async fn process_with_tokens(&self, tokens: &[u32]) -> Result<ProcessingOutput> {
            let mut processed = self.processed.lock().await;
            *processed += 1;
            
            Ok(ProcessingOutput {
                text: "Test output".to_string(),
                tokens: tokens.to_vec(),
                processing_time: Duration::from_millis(100),
            })
        }
    }

    #[tokio::test]
    async fn test_stream_processing() -> Result<()> {
        let (processor, runtime) = create_test_processor().await?;
        
        let mut handle = processor.create_handle().await?;
        
        // Process some inputs
        let output = handle.process("Test input".to_string()).await?;
        assert!(!output.text.is_empty());
        assert!(!output.tokens.is_empty());
        
        let processed = runtime.processed.lock().await;
        assert_eq!(*processed, 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_stream_context() -> Result<()> {
        let (processor, _) = create_test_processor().await?;
        
        let mut handle = processor.create_handle().await?;
        
        // Process multiple messages
        handle.process("First message".to_string()).await?;
        let output = handle.process("Second message".to_string()).await?;
        
        // Check that context was maintained
        let stats = processor.get_stream_stats().await;
        assert_eq!(stats.len(), 1);
        assert!(stats[0].context_tokens > 0);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_stream_cleanup() -> Result<()> {
        let (processor, _) = create_test_processor().await?;
        
        let handle = processor.create_handle().await?;
        assert_eq!(processor.active_stream_count().await, 1);
        
        drop(handle);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(processor.active_stream_count().await, 0);
        
        Ok(())
    }
}