// Location: src/model/runtime.rs

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, Llama, Cache, LlamaEosToks};
use tokenizers::Tokenizer;
use std::sync::Arc;
use tokio::sync::Mutex;

use candle_transformers::generation::LogitsProcessor;
use crate::processing::ProcessingOutput;

use crate::{
    config::EngineConfig,
    error::EngineError,
    gpu_utils::GpuMonitor,
};

pub struct ModelRuntime {
    model: Arc<Llama>,
    device: Device,
    device_id: usize,
    config: Arc<Config>,
    tokenizer: Arc<Tokenizer>,
    state: Arc<Mutex<RuntimeState>>,
    gpu_monitor: GpuMonitor,
    generation_config: GenerationConfig,
}

#[derive(Clone)]
struct GenerationConfig {
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    repetition_penalty: f32,
    repetition_context_size: usize,
}

struct RuntimeState {
    total_processed: usize,
    total_tokens_generated: usize,
    current_batch_size: usize,
    last_processing_time: Option<std::time::Duration>,
    peak_memory_usage: usize,
    cache: Option<Cache>,
}

impl ModelRuntime {
    pub async fn new(
        device_id: usize,
        device: Device,
        model: Llama,
        config: Config,
        tokenizer: Tokenizer,
        app_config: &EngineConfig,
    ) -> Result<Self> {
        let generation_config = GenerationConfig {
            max_tokens: app_config.generation.max_output_tokens,
            temperature: app_config.generation.temperature,
            top_p: app_config.generation.top_p,
            repetition_penalty: app_config.generation.repetition_penalty,
            repetition_context_size: app_config.generation.repetition_context_size,
        };

        let state = RuntimeState {
            total_processed: 0,
            total_tokens_generated: 0,
            current_batch_size: 1,
            last_processing_time: None,
            peak_memory_usage: 0,
            cache: None,
        };

        Ok(Self {
            model: Arc::new(model),
            device,
            device_id,
            config: Arc::new(config),
            tokenizer: Arc::new(tokenizer),
            state: Arc::new(Mutex::new(state)),
            gpu_monitor: GpuMonitor::new(device_id, device.clone()),
            generation_config,
        })
    }

    /// Process a single input
    pub async fn process(&self, input: String) -> Result<ProcessingOutput> {
        let start_time = std::time::Instant::now();

        // Get or create cache
        let mut cache = self.state.lock().await;
        if cache.cache.is_none() {
            cache.cache = Some(Cache::new(
                true,
                DType::BF16,
                &self.config,
                &self.device,
            )?);
        }
        
        // Process with proper error conversion
        let result = self.process_internal(
            input,
            cache.cache.as_mut().unwrap(),
            start_time,
        ).await;

        // Update metrics regardless of success/failure
        if let Ok(ref output) = result {
            let mut metrics = self.metrics.lock().await;
            metrics.record_processing(
                output.tokens.len(),
                start_time.elapsed(),
            ).await?;
        }

        result
    }

    /// Process a batch of inputs
    pub async fn process_batch(&self, inputs: Vec<String>) -> Result<Vec<ProcessingOutput>> {
        let start_time = std::time::Instant::now();
        let batch_size = inputs.len();

        // Tokenize all inputs
        let mut max_length = 0;
        let mut token_tensors = Vec::with_capacity(batch_size);
        
        for input in &inputs {
            let tokens = self.tokenize(input)?;
            max_length = max_length.max(tokens.dims()[0]);
            token_tensors.push(tokens);
        }

        // Pad sequences to same length
        let mut padded_tokens = Vec::with_capacity(batch_size * max_length);
        for mut tokens in token_tensors {
            if tokens.dims()[0] < max_length {
                tokens = self.pad_tensor(&tokens, max_length)?;
            }
            padded_tokens.extend_from_slice(&tokens.to_vec1()?);
        }

        // Create input tensor
        let input_tensor = Tensor::from_slice(&padded_tokens, &self.device)?
            .reshape((batch_size, max_length))?;

        // Initialize cache
        let mut cache = Cache::new(
            true,
            DType::BF16,
            &self.config,
            &self.device,
        )?;

        let mut outputs = Vec::with_capacity(batch_size);
        let mut logits_processor = candle_transformers::generation::LogitsProcessor::new(
            299792458,
            self.generation_config.temperature,
            Some(self.generation_config.top_p),
        );

        // Generate tokens for each sequence
        for i in 0..batch_size {
            let mut generated = vec![];
            let mut current = input_tensor.get(i)?.unsqueeze(0)?;
            let mut pos = 0;

            for _ in 0..self.generation_config.max_tokens {
                let logits = self.model.forward(&current, pos, &mut cache)?;
                let logits = logits.to_dtype(DType::F32)?;
                
                let next_token = logits_processor.sample(&logits)?;
                generated.push(next_token);

                if let Some(eos_token) = &self.config.eos_token_id {
                    match eos_token {
                        LlamaEosToks::Single(token) if *token == next_token => break,
                        LlamaEosToks::Multiple(tokens) if tokens.contains(&next_token) => break,
                        _ => {}
                    }
                }

                current = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                pos += 1;
            }

            outputs.push(ProcessingOutput {
                text: self.decode(&generated)?,
                tokens: generated,
                processing_time: start_time.elapsed(),
            });
        }

        // Update state
        let mut state = self.state.lock().await;
        state.total_processed += batch_size;
        state.current_batch_size = batch_size;
        state.last_processing_time = Some(start_time.elapsed());

        Ok(outputs)
    }

    // Process a batch with tokens
    pub async fn process_batch_with_tokens(
        &self,
        token_ids: Vec<Vec<u32>>,
        attention_masks: Vec<Vec<u8>>,
        sequence_lengths: Vec<usize>,
    ) -> Result<Vec<ProcessingOutput>> {
        // Create tensors from raw data
        let batch_size = token_ids.len();
        let max_len = token_ids.iter().map(|x| x.len()).max().unwrap_or(0);
        
        let token_tensor = Tensor::new(
            token_ids.iter().flat_map(|x| x.iter()).copied().collect::<Vec<_>>().as_slice(),
            &self.device
        )?.reshape(&[batch_size as i64, max_len as i64])?;

        let mask_tensor = Tensor::new(
            attention_masks.iter().flat_map(|x| x.iter()).copied().collect::<Vec<_>>().as_slice(),
            &self.device
        )?.reshape(&[batch_size as i64, max_len as i64])?;

        // Process through model
        let outputs = self.model.forward_t(
            &token_tensor,
            &mask_tensor,
            &self.generation_config,
            &mut self.state.lock().await.cache.as_mut().unwrap()
        )?;

        // Convert outputs to ProcessingOutput structs
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let output_slice = outputs.get(i)?;
            let tokens = output_slice.to_vec1::<u32>()?;
            
            results.push(ProcessingOutput {
                text: self.tokenizer.decode(&tokens, true)?,
                tokens: tokens.into_iter().map(|t| t.to_string()).collect(),
                processing_time: std::time::Instant::now().duration_since(self.start_time),
            });
        }

        Ok(results)
    }

    /// Tokenize input text
    fn tokenize(&self, text: &str) -> Result<Tensor> {
        let tokens = self.tokenizer.encode(text, true)
            .map_err(|e| EngineError::ProcessingError { 
                message: format!("Tokenization failed: {}", e),
                source: None,
            })?;
        
        Ok(Tensor::new(tokens.get_ids(), &self.device)?)
    }

    /// Pad tensor to specified length
    fn pad_tensor(&self, tensor: &Tensor, target_len: usize) -> Result<Tensor> {
        let current_len = tensor.dims()[0];
        if current_len >= target_len {
            return Ok(tensor.clone());
        }

        let padding = vec![0; target_len - current_len];
        let mut data = tensor.to_vec1()?;
        data.extend(padding);
        
        Ok(Tensor::new(data.as_slice(), &self.device)?)
    }

    /// Decode tokens to text
    fn decode(&self, tokens: &[usize]) -> Result<String> {
        // Convert usize tokens to u32 for tokenizer
        let u32_tokens: Vec<u32> = tokens.iter()
            .map(|&t| t as u32)
            .collect();

        self.tokenizer.decode(&u32_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }

    /// Get runtime statistics
    pub async fn get_stats(&self) -> RuntimeStats {
        let state = self.state.lock().await;
        RuntimeStats {
            total_processed: state.total_processed,
            total_tokens_generated: state.total_tokens_generated,
            current_batch_size: state.current_batch_size,
            last_processing_time: state.last_processing_time,
            peak_memory_usage: state.peak_memory_usage,
        }
    }

    /// Force cleanup of GPU resources
    pub async fn cleanup(&self) -> Result<()> {
        let mut state = self.state.lock().await;
        state.cache = None;
        
        if let Device::Cuda(_) = self.device {
            unsafe {
                candle_core::cuda_backend::cuda::cuStreamSynchronize(0)?;
                candle_core::cuda_backend::cuda::cuMemGetInfo(
                    &mut 0 as *mut usize,
                    &mut 0 as *mut usize,
                )?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeStats {
    pub total_processed: usize,
    pub total_tokens_generated: usize,
    pub current_batch_size: usize,
    pub last_processing_time: Option<std::time::Duration>,
    pub peak_memory_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    async fn create_test_runtime() -> Result<ModelRuntime> {
        let device = Device::Cpu;
        let config = Config::default();
        let app_config = EngineConfig::default();
        
        let tokenizer = Tokenizer::from_file(
            Path::new("models/tokenizer.json")
        ).unwrap();
        
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Llama::load(vb, &config)?;

        ModelRuntime::new(
            0,
            device,
            model,
            config,
            tokenizer,
            &app_config,
        ).await
    }

    #[tokio::test]
    async fn test_single_processing() -> Result<()> {
        let runtime = create_test_runtime().await?;
        
        let output = runtime.process("Test input".to_string()).await?;
        assert!(!output.text.is_empty());
        assert!(!output.tokens.is_empty());
        
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_processing() -> Result<()> {
        let runtime = create_test_runtime().await?;
        
        let inputs = vec![
            "Test 1".to_string(),
            "Test 2".to_string(),
        ];
        
        let outputs = runtime.process_batch(inputs).await?;
        assert_eq!(outputs.len(), 2);
        
        Ok(())
    }
}