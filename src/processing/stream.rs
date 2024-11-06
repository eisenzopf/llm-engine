// Location: src/processing/stream.rs

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, warn};

use crate::{
    config::EngineConfig,
    error::{EngineError, Result},
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
    async fn process_stream(
        stream_id: usize,
        runtime: Arc<ModelRuntime>,
        tokenizer: Arc<LlamaTokenizer>,
        mut input_rx: mpsc::Receiver<String>,
        output_tx: mpsc::Sender<ProcessingOutput>,
        metrics: Arc<Mutex<MetricsCollector>>,
        active_streams: Arc<Mutex<Vec<StreamState>>>,
        config: Arc<EngineConfig>,
    ) {
        debug!("Starting stream processor {}", stream_id);

        while let Some(input) = input_rx.recv().await {
            let start_time = std::time::Instant::now();
            
            // Get current state
            let mut streams = active_streams.lock().await;
            let state = streams.iter_mut().find(|s| s.id == stream_id)
                .expect("Stream state not found");

            // Add instruction template if configured
            let input = if !input.starts_with("### Instruction:") {
                config.generation.instruction_template.replace("{}", &input)
            } else {
                input
            };

            // Tokenize input
            let mut input_tokens = match tokenizer.encode(&input, true).await {
                Ok(tokens) => tokens,
                Err(e) => {
                    warn!("Tokenization error on stream {}: {:?}", stream_id, e);
                    if output_tx.send(ProcessingOutput::error(e)).await.is_err() {
                        break;
                    }
                    continue;
                }
            };

            // Add context from previous interaction if any
            if !state.current_context.is_empty() {
                let context_size = config.model.max_sequence_length / 4;
                input_tokens.splice(
                    0..0, 
                    state.current_context.iter()
                        .rev()
                        .take(context_size)
                        .rev()
                        .cloned()
                );
            }

            // Ensure we don't exceed maximum sequence length
            if input_tokens.len() > config.model.max_sequence_length {
                let truncate_len = input_tokens.len() - config.model.max_sequence_length;
                input_tokens.drain(0..truncate_len);
            }

            // Process through model
            match runtime.process_with_tokens(&input_tokens).await {
                Ok(output) => {
                    // Update state
                    state.processed_tokens += output.tokens.len();
                    state.current_context.extend(&output.tokens);

                    // Trim context if needed
                    if state.current_context.len() > config.model.max_sequence_length {
                        let excess = state.current_context.len() - config.model.max_sequence_length;
                        state.current_context.drain(0..excess);
                    }

                    // Update metrics
                    {
                        let mut metrics = metrics.lock().await;
                        metrics.record_stream_processing(
                            stream_id,
                            output.tokens.len(),
                            start_time.elapsed(),
                        ).await;
                    }

                    // Send output
                    if output_tx.send(output).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Processing error on stream {}: {:?}", stream_id, e);
                    if output_tx.send(ProcessingOutput::error(e)).await.is_err() {
                        break;
                    }
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