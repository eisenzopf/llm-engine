use anyhow::Result;
use llm_engine::{
    LLMEngine,
    config::{EngineConfig, ProcessingMode},
    utils::prelude::*,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    setup_logging(LogConfig {
        level: tracing::Level::INFO,
        timestamps: true,
        ..Default::default()
    })?;

    // Configure the engine
    let config = EngineConfig {
        processing: ProcessingConfig {
            mode: ProcessingMode::Streaming,
            concurrency: 1,
            ..Default::default()
        },
        model: ModelConfig {
            model_path: PathBuf::from("models/llama-7b"),
            max_sequence_length: 2048,
            parameters: ModelParameters {
                temperature: 0.7,
                top_p: 0.95,
                max_tokens: 100,
            },
        },
        ..Default::default()
    };

    info!("Initializing engine...");
    let engine = LLMEngine::builder()
        .with_config(config)
        .build()
        .await?;

    // Get a stream handle
    let mut handles = engine.get_stream_handles().await?;
    let mut handle = handles.pop().unwrap();

    info!("Engine initialized. Starting interaction loop...");

    // Process inputs in a loop
    loop {
        print!("> ");
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "exit" {
            break;
        }

        // Profile the processing
        let span = profile_span!("text_generation");
        let result = handle.process(input.to_string()).await?;
        drop(span);

        println!("Generated: {}", result.text);
        
        // Log performance metrics
        info!(
            tokens = result.tokens.len(),
            time_ms = result.processing_time.as_millis(),
            "Processing complete"
        );
    }

    info!("Shutting down...");
    engine.shutdown().await?;
    Ok(())
}