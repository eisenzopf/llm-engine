// Location: src/model/tokenizer.rs

use std::sync::Arc;
use anyhow::Result;
use tokenizers::{Tokenizer as HfTokenizer, Token};
use tokenizers::models::bpe::BPE;
use std::path::Path;
use std::collections::HashMap;

use crate::error::EngineError;

pub struct LlamaTokenizer {
    /// HuggingFace tokenizer
    tokenizer: Arc<HfTokenizer>,
    /// Special token IDs
    special_tokens: SpecialTokens,
    /// Token metadata
    metadata: TokenizerMetadata,
    /// Token statistics
    stats: Arc<tokio::sync::RwLock<TokenizerStats>>,
}

#[derive(Debug, Clone)]
struct SpecialTokens {
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    unk_token_id: Option<u32>,
    mask_token_id: Option<u32>,
}

#[derive(Debug, Clone)]
struct TokenizerMetadata {
    vocab_size: usize,
    model_max_length: usize,
    truncation_side: TruncationSide,
    padding_side: PaddingSide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TruncationSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaddingSide {
    Left,
    Right,
}

#[derive(Debug, Default)]
struct TokenizerStats {
    total_processed: usize,
    total_tokens: usize,
    avg_tokens_per_sequence: f32,
    token_frequency: HashMap<u32, usize>,
}

impl LlamaTokenizer {
    /// Create a new tokenizer from a file
    pub async fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = HfTokenizer::from_file(path)
            .map_err(|e| EngineError::InitializationError {
                message: format!("Failed to load tokenizer: {}", e),
                source: None,
            })?;

        // Extract special tokens
        let special_tokens = SpecialTokens {
            bos_token_id: tokenizer.token_to_id("<s>"),
            eos_token_id: tokenizer.token_to_id("</s>"),
            pad_token_id: tokenizer.token_to_id("<pad>"),
            unk_token_id: tokenizer.token_to_id("<unk>"),
            mask_token_id: tokenizer.token_to_id("<mask>"),
        };

        // Get metadata from tokenizer config
        let metadata = TokenizerMetadata {
            vocab_size: tokenizer.get_vocab_size(),
            model_max_length: 32768, // Llama's default
            truncation_side: TruncationSide::Right,
            padding_side: PaddingSide::Right,
        };

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            special_tokens,
            metadata,
            stats: Arc::new(tokio::sync::RwLock::new(TokenizerStats::default())),
        })
    }

    /// Encode text to token IDs
    pub async fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(
            text,
            add_special_tokens,
        ).map_err(|e| EngineError::ProcessingError {
            message: format!("Tokenization failed: {}", e),
            source: None,
        })?;

        let tokens: Vec<u32> = encoding.get_ids().to_vec();

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_processed += 1;
        stats.total_tokens += tokens.len();
        stats.avg_tokens_per_sequence = stats.total_tokens as f32 / stats.total_processed as f32;
        
        for token in &tokens {
            *stats.token_frequency.entry(*token).or_insert(0) += 1;
        }

        Ok(tokens)
    }

    /// Encode a batch of texts
    pub async fn encode_batch(
        &self,
        texts: &[String],
        add_special_tokens: bool,
        padding: bool,
    ) -> Result<EncodingBatch> {
        let encodings = self.tokenizer.encode_batch(
            texts,
            add_special_tokens,
        ).map_err(|e| EngineError::ProcessingError {
            message: format!("Batch tokenization failed: {}", e),
            source: None,
        })?;

        let mut max_len = 0;
        let mut token_ids = Vec::with_capacity(texts.len());
        let mut attention_masks = Vec::with_capacity(texts.len());

        for encoding in &encodings {
            let ids = encoding.get_ids();
            max_len = max_len.max(ids.len());
            token_ids.push(ids.to_vec());
            attention_masks.push(vec![1u8; ids.len()]);
        }

        // Add padding if requested
        if padding {
            for (ids, mask) in token_ids.iter_mut().zip(attention_masks.iter_mut()) {
                if ids.len() < max_len {
                    let pad_len = max_len - ids.len();
                    match self.metadata.padding_side {
                        PaddingSide::Right => {
                            ids.extend(vec![self.special_tokens.pad_token_id.unwrap_or(0); pad_len]);
                            mask.extend(vec![0u8; pad_len]);
                        }
                        PaddingSide::Left => {
                            ids.splice(0..0, vec![self.special_tokens.pad_token_id.unwrap_or(0); pad_len]);
                            mask.splice(0..0, vec![0u8; pad_len]);
                        }
                    }
                }
            }
        }

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_processed += texts.len();
        for ids in &token_ids {
            stats.total_tokens += ids.len();
            for token in ids {
                *stats.token_frequency.entry(*token).or_insert(0) += 1;
            }
        }
        stats.avg_tokens_per_sequence = stats.total_tokens as f32 / stats.total_processed as f32;

        Ok(EncodingBatch {
            token_ids,
            attention_masks,
            sequence_lengths: encodings.iter().map(|e| e.get_ids().len()).collect(),
        })
    }

    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer.decode(
            tokens,
            skip_special_tokens,
        ).map_err(|e| EngineError::ProcessingError {
            message: format!("Decoding failed: {}", e),
            source: None,
        })
    }

    /// Decode a batch of token sequences
    pub fn decode_batch(
        &self,
        sequences: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        self.tokenizer.decode_batch(
            sequences,
            skip_special_tokens,
        ).map_err(|e| EngineError::ProcessingError {
            message: format!("Batch decoding failed: {}", e),
            source: None,
        })
    }

    /// Get token ID for a string
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Get string for a token ID
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.metadata.vocab_size
    }

    /// Get the maximum sequence length
    pub fn model_max_length(&self) -> usize {
        self.metadata.model_max_length
    }

    /// Get tokenizer statistics
    pub async fn get_stats(&self) -> TokenizerStatsSnapshot {
        let stats = self.stats.read().await;
        TokenizerStatsSnapshot {
            total_processed: stats.total_processed,
            total_tokens: stats.total_tokens,
            avg_tokens_per_sequence: stats.avg_tokens_per_sequence,
            vocab_coverage: stats.token_frequency.len() as f32 / self.vocab_size() as f32,
        }
    }
}

/// Batch encoding result
#[derive(Debug, Clone)]
pub struct EncodingBatch {
    pub token_ids: Vec<Vec<u32>>,
    pub attention_masks: Vec<Vec<u8>>,
    pub sequence_lengths: Vec<usize>,
}

/// Tokenizer statistics snapshot
#[derive(Debug, Clone)]
pub struct TokenizerStatsSnapshot {
    pub total_processed: usize,
    pub total_tokens: usize,
    pub avg_tokens_per_sequence: f32,
    pub vocab_coverage: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    async fn create_test_tokenizer() -> Result<LlamaTokenizer> {
        LlamaTokenizer::from_file("models/tokenizer.json").await
    }

    #[tokio::test]
    async fn test_basic_tokenization() -> Result<()> {
        let tokenizer = create_test_tokenizer().await?;
        
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text, true).await?;
        assert!(!tokens.is_empty());
        
        let decoded = tokenizer.decode(&tokens, true)?;
        assert_eq!(decoded.trim(), text);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_tokenization() -> Result<()> {
        let tokenizer = create_test_tokenizer().await?;
        
        let texts = vec![
            "Hello".to_string(),
            "World".to_string(),
        ];
        
        let batch = tokenizer.encode_batch(&texts, true, true).await?;
        assert_eq!(batch.token_ids.len(), 2);
        assert_eq!(batch.attention_masks.len(), 2);
        
        // Check padding
        assert_eq!(batch.token_ids[0].len(), batch.token_ids[1].len());
        
        Ok(())
    }

    #[tokio::test]
    async fn test_special_tokens() -> Result<()> {
        let tokenizer = create_test_tokenizer().await?;
        
        assert!(tokenizer.special_tokens.bos_token_id.is_some());
        assert!(tokenizer.special_tokens.eos_token_id.is_some());
        
        Ok(())
    }

    #[tokio::test]
    async fn test_statistics() -> Result<()> {
        let tokenizer = create_test_tokenizer().await?;
        
        let text = "This is a test sentence.";
        tokenizer.encode(text, true).await?;
        
        let stats = tokenizer.get_stats().await;
        assert_eq!(stats.total_processed, 1);
        assert!(stats.total_tokens > 0);
        assert!(stats.vocab_coverage > 0.0);
        
        Ok(())
    }
}