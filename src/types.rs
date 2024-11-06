//! Common type definitions used throughout the engine

use std::time::Duration;
use serde::Serialize;
se serde::Deserialize;

/// Processing output from the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingOutput {
    /// Generated text
    pub text: String,
    /// Generated tokens
    pub tokens: Vec<String>,
    /// Time taken to process
    pub processing_time: Duration,
}

impl ProcessingOutput {
    pub fn error(error: impl Into<String>) -> Self {
        Self {
            text: error.into(),
            tokens: Vec::new(),
            processing_time: Duration::default(),
        }
    }
}

/// Handle for stream processing
#[derive(Debug)]
pub struct StreamHandle {
    /// Device ID this handle is associated with
    device_id: usize,
    /// Channel for sending inputs
    input_sender: tokio::sync::mpsc::Sender<String>,
    /// Channel for receiving outputs
    output_receiver: tokio::sync::mpsc::Receiver<ProcessingOutput>,
}

impl StreamHandle {
    /// Create a new stream handle
    pub(crate) fn new(
        device_id: usize,
        input_sender: tokio::sync::mpsc::Sender<String>,
        output_receiver: tokio::sync::mpsc::Receiver<ProcessingOutput>,
    ) -> Self {
        Self {
            device_id,
            input_sender,
            output_receiver,
        }
    }

    /// Get the device ID this handle is associated with
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Send input for processing
    pub async fn process(&mut self, input: String) -> crate::Result<ProcessingOutput> {
        self.input_sender.send(input).await
            .map_err(|_| crate::error::EngineError::ProcessingError {
                message: "Failed to send input".to_string(),
                source: None,
            })?;
            
        self.output_receiver.recv().await
            .ok_or_else(|| crate::error::EngineError::ProcessingError {
                message: "Failed to receive output".to_string(),
                source: None,
            })
    }
}

/// Handle for queue processing
#[derive(Debug)]
pub struct QueueHandle {
    /// Job ID in the queue
    job_id: usize,
    /// Channel for receiving the result
    receiver: tokio::sync::oneshot::Receiver<crate::Result<ProcessingOutput>>,
}

impl QueueHandle {
    /// Create a new queue handle
    pub(crate) fn new(
        job_id: usize,
        receiver: tokio::sync::oneshot::Receiver<crate::Result<ProcessingOutput>>,
    ) -> Self {
        Self {
            job_id,
            receiver,
        }
    }

    /// Get the job ID
    pub fn job_id(&self) -> usize {
        self.job_id
    }

    /// Wait for the processing result
    pub async fn wait(self) -> crate::Result<ProcessingOutput> {
        self.receiver.await
            .map_err(|_| crate::error::EngineError::ProcessingError {
                message: "Failed to receive result".to_string(),
                source: None,
            })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_stream_handle() {
        let (input_tx, mut input_rx) = tokio::sync::mpsc::channel(1);
        let (output_tx, output_rx) = tokio::sync::mpsc::channel(1);
        
        let mut handle = StreamHandle::new(0, input_tx, output_rx);
        
        // Spawn echo processor
        tokio::spawn(async move {
            while let Some(input) = input_rx.recv().await {
                output_tx.send(ProcessingOutput {
                    text: input,
                    tokens: vec![],
                    processing_time: Duration::from_millis(100),
                }).await.unwrap();
            }
        });
        
        let result = handle.process("test".to_string()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "test");
    }

    #[tokio::test]
    async fn test_queue_handle() {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = QueueHandle::new(1, rx);
        
        tx.send(Ok(ProcessingOutput {
            text: "test".to_string(),
            tokens: vec![],
            processing_time: Duration::from_millis(100),
        })).unwrap();
        
        let result = handle.wait().await;
        assert!(result.is_ok());
    }
}