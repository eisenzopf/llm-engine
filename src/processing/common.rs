// Location: src/processing/common.rs

use std::time::Duration;
use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::Result;
use candle_core::Tensor;

use crate::{
    config::EngineConfig,
    error::EngineError,
    model::LlamaTokenizer,
    types::ProcessingOutput,
};

/// Common processing statistics
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    /// Number of sequences processed
    pub total_sequences: usize,
    
    /// Total tokens processed
    pub total_tokens: usize,
    
    /// Total processing time
    pub total_time: Duration,
    
    /// Average tokens per second
    pub tokens_per_second: f32,
    
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    
    /// Current batch statistics if applicable
    pub batch_stats: Option<BatchStats>,
    
    /// Current queue statistics if applicable
    pub queue_stats: Option<QueueStats>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current GPU memory usage in bytes
    pub gpu_memory_used: usize,
    
    /// Peak GPU memory usage in bytes
    pub peak_gpu_memory: usize,
    
    /// Current token cache size
    pub token_cache_size: usize,
}

/// Batch processing statistics
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Current batch size
    pub current_batch_size: usize,
    
    /// Average sequence length in batch
    pub avg_sequence_length: usize,
    
    /// GPU utilization percentage
    pub gpu_utilization: f32,
}

/// Queue processing statistics
#[derive(Debug, Clone)]
pub struct QueueStats {
    /// Current queue depth
    pub queue_depth: usize,
    
    /// Average wait time
    pub avg_wait_time: Duration,
    
    /// Maximum wait time
    pub max_wait_time: Duration,
}

/// Token sequence management
#[derive(Debug)]
pub struct TokenSequence {
    /// Input token IDs
    pub token_ids: Vec<u32>,
    
    /// Attention mask
    pub attention_mask: Vec<u8>,
    
    /// Position IDs
    pub position_ids: Vec<u32>,
    
    /// Sequence length
    pub length: usize,
}

impl TokenSequence {
    /// Create a new token sequence
    pub fn new(token_ids: Vec<u32>) -> Self {
        let length = token_ids.len();
        Self {
            attention_mask: vec![1; length],
            position_ids: (0..length as u32).collect(),
            length,
            token_ids,
        }
    }

    /// Pad sequence to target length
    pub fn pad_to_length(&mut self, target_length: usize, pad_token_id: u32) {
        if self.length >= target_length {
            return;
        }

        let pad_length = target_length - self.length;
        self.token_ids.extend(vec![pad_token_id; pad_length]);
        self.attention_mask.extend(vec![0; pad_length]);
        self.position_ids.extend(vec![self.length as u32; pad_length]);
        self.length = target_length;
    }

    /// Convert to tensors
    pub async fn to_tensors(&self, device: &candle_core::Device) -> Result<SequenceTensors> {
        Ok(SequenceTensors {
            token_ids: Tensor::new(&self.token_ids, device)?,
            attention_mask: Tensor::new(&self.attention_mask, device)?,
            position_ids: Tensor::new(&self.position_ids, device)?,
        })
    }
}

/// Tensor representation of a sequence
#[derive(Debug)]
pub struct SequenceTensors {
    pub token_ids: Tensor,
    pub attention_mask: Tensor,
    pub position_ids: Tensor,
}

/// Batch assembly helper
pub struct BatchAssembler {
    config: Arc<EngineConfig>,
    tokenizer: Arc<LlamaTokenizer>,
    current_batch: Vec<TokenSequence>,
    max_length: usize,
}

impl BatchAssembler {
    /// Create new batch assembler
    pub fn new(config: Arc<EngineConfig>, tokenizer: Arc<LlamaTokenizer>) -> Self {
        Self {
            config,
            tokenizer,
            current_batch: Vec::new(),
            max_length: 0,
        }
    }

    /// Add sequence to batch
    pub async fn add(&mut self, input: &str) -> Result<bool> {
        // Tokenize input
        let tokens = self.tokenizer.encode(input, true).await?;
        let mut sequence = TokenSequence::new(tokens);
        
        // Check if adding would exceed maximum batch size
        if self.current_batch.len() >= self.config.processing.batch_size.unwrap_or(32) {
            return Ok(false);
        }

        // Update maximum length
        self.max_length = self.max_length.max(sequence.length);
        
        // Add to batch
        self.current_batch.push(sequence);
        
        Ok(true)
    }

    /// Finalize batch for processing
    pub async fn finalize(&mut self, device: &candle_core::Device) -> Result<BatchTensors> {
        // Pad all sequences to max length
        let pad_token_id = self.tokenizer.pad_token_id()
            .ok_or_else(|| EngineError::ProcessingError {
                message: "No pad token ID available".to_string(),
                source: None,
            })?;

        for sequence in &mut self.current_batch {
            sequence.pad_to_length(self.max_length, pad_token_id);
        }

        // Convert to tensors
        let batch_size = self.current_batch.len();
        let mut token_ids = Vec::with_capacity(batch_size * self.max_length);
        let mut attention_mask = Vec::with_capacity(batch_size * self.max_length);
        let mut position_ids = Vec::with_capacity(batch_size * self.max_length);

        for sequence in &self.current_batch {
            token_ids.extend(&sequence.token_ids);
            attention_mask.extend(&sequence.attention_mask);
            position_ids.extend(&sequence.position_ids);
        }

        Ok(BatchTensors {
            token_ids: Tensor::new(&token_ids, device)?
                .reshape((batch_size, self.max_length))?,
            attention_mask: Tensor::new(&attention_mask, device)?
                .reshape((batch_size, self.max_length))?,
            position_ids: Tensor::new(&position_ids, device)?
                .reshape((batch_size, self.max_length))?,
            sequence_lengths: self.current_batch.iter()
                .map(|s| s.length)
                .collect(),
        })
    }

    /// Clear the current batch
    pub fn clear(&mut self) {
        self.current_batch.clear();
        self.max_length = 0;
    }
}

/// Tensor representation of a batch
#[derive(Debug)]
pub struct BatchTensors {
    pub token_ids: Tensor,
    pub attention_mask: Tensor,
    pub position_ids: Tensor,
    pub sequence_lengths: Vec<usize>,
}

/// Output processing helper
pub struct OutputProcessor {
    config: Arc<EngineConfig>,
    tokenizer: Arc<LlamaTokenizer>,
}

impl OutputProcessor {
    /// Create new output processor
    pub fn new(config: Arc<EngineConfig>, tokenizer: Arc<LlamaTokenizer>) -> Self {
        Self {
            config,
            tokenizer,
        }
    }

    /// Process model outputs
    pub async fn process_outputs(
        &self,
        tokens: Vec<Vec<u32>>,
        processing_time: Duration,
    ) -> Result<Vec<ProcessingOutput>> {
        let mut outputs = Vec::with_capacity(tokens.len());
        
        for token_ids in tokens {
            // Decode tokens
            let text = self.tokenizer.decode(&token_ids, true)?;
            
            // Clean up instruction template if present
            let text = if let Some(idx) = text.find("### Response:") {
                text[idx + 13..].trim().to_string()
            } else {
                text
            };

            outputs.push(ProcessingOutput {
                text,
                tokens: token_ids,
                processing_time,
            });
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_sequence() -> Result<()> {
        let sequence = TokenSequence::new(vec![1, 2, 3, 4]);
        assert_eq!(sequence.length, 4);
        assert_eq!(sequence.attention_mask, vec![1, 1, 1, 1]);
        assert_eq!(sequence.position_ids, vec![0, 1, 2, 3]);
        Ok(())
    }

    #[tokio::test]
    async fn test_sequence_padding() -> Result<()> {
        let mut sequence = TokenSequence::new(vec![1, 2, 3]);
        sequence.pad_to_length(5, 0);
        
        assert_eq!(sequence.length, 5);
        assert_eq!(sequence.token_ids, vec![1, 2, 3, 0, 0]);
        assert_eq!(sequence.attention_mask, vec![1, 1, 1, 0, 0]);
        assert_eq!(sequence.position_ids, vec![0, 1, 2, 3, 3]);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_assembly() -> Result<()> {
        let config = Arc::new(EngineConfig::default());
        let tokenizer = Arc::new(LlamaTokenizer::from_file("models/tokenizer.json").await?);
        let device = candle_core::Device::Cpu;
        
        let mut assembler = BatchAssembler::new(config, tokenizer);
        
        // Add sequences
        assembler.add("Test 1").await?;
        assembler.add("Test 2").await?;
        
        // Finalize batch
        let tensors = assembler.finalize(&device).await?;
        
        assert_eq!(tensors.sequence_lengths.len(), 2);
        assert!(tensors.token_ids.dims()[0] == 2); // batch size
        
        Ok(())
    }

    #[tokio::test]
    async fn test_output_processing() -> Result<()> {
        let config = Arc::new(EngineConfig::default());
        let tokenizer = Arc::new(LlamaTokenizer::from_file("models/tokenizer.json").await?);
        
        let processor = OutputProcessor::new(config, tokenizer);
        
        let token_sequences = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ];
        
        let outputs = processor.process_outputs(
            token_sequences,
            Duration::from_millis(100),
        ).await?;
        
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].tokens.len(), 3);
        assert_eq!(outputs[1].tokens.len(), 3);
        
        Ok(())
    }
}