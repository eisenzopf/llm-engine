use anyhow::Result;
use llm_engine::{
    LLMEngine,
    config::{EngineConfig, ProcessingMode},
    utils::prelude::*,
};
use std::sync::Arc;
use std::time::Instant;
use std::fs::File;
use std::io::{BufRead, BufReader};
use indicatif::{ProgressBar, ProgressStyle};

#[tokio::main]
async fn main() -> Result<()> {
    setup_logging(LogConfig {
        level: tracing::Level::INFO,
        timestamps: true,
        ..Default::default()
    })?;

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input_file> <output_file>", args[0]);
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];

    // Configure the engine for batch processing
    let config = EngineConfig {
        processing: ProcessingConfig {
            mode: ProcessingMode::Batch,
            batch_size: Some(32),
            concurrency: 4,
            auto_adjust_batch: true,
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
        gpu: GpuConfig {
            memory_threshold: 0.9,
            dynamic_batch_adjustment: true,
            flash_attention: true,
            ..Default::default()
        },
        ..Default::default()
    };

    info!("Initializing engine...");
    let engine = LLMEngine::builder()
        .with_config(config)
        .build()
        .await?;

    // Read input file
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<std::io::Result<_>>()?;
    let total_lines = lines.len();

    // Create output file
    let mut output = File::create(output_file)?;
    writeln!(output, "input,output,processing_time_ms,tokens")?;

    // Set up progress bar
    let progress = ProgressBar::new(total_lines as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%) - {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    // Process in batches
    let mut processed = 0;
    let start_time = Instant::now();
    let batch_size = engine.get_config().processing.batch_size.unwrap_or(32);

    for chunk in lines.chunks(batch_size) {
        let span = profile_span!("batch_processing");
        let results = engine.process_batch(chunk.to_vec()).await?;
        drop(span);

        // Write results
        for (input, result) in chunk.iter().zip(results.iter()) {
            writeln!(
                output,
                "{},{},{},{}",
                input.replace(",", "\\,"),
                result.text.replace(",", "\\,"),
                result.processing_time.as_millis(),
                result.tokens.len(),
            )?;
        }

        processed += chunk.len();
        progress.set_position(processed as u64);
        progress.set_message(format!(
            "{:.1} sequences/sec", 
            processed as f64 / start_time.elapsed().as_secs_f64()
        ));
    }

    progress.finish_with_message("Processing complete");

    // Print performance summary
    let total_time = start_time.elapsed();
    let throughput = total_lines as f64 / total_time.as_secs_f64();

    info!("Processing Summary:");
    info!("  Total sequences: {}", total_lines);
    info!("  Total time: {:.2?}", total_time);
    info!("  Throughput: {:.1} sequences/sec", throughput);

    // Get GPU utilization stats
    let metrics = engine.get_metrics().await?;
    info!("GPU Statistics:");
    for (gpu_id, stats) in metrics.gpu_stats.iter().enumerate() {
        info!(
            "  GPU {}: {:.1}% utilization, {:.1}GB memory used",
            gpu_id,
            stats.compute_utilization * 100.0,
            stats.memory_used as f64 / 1024.0 / 1024.0 / 1024.0
        );
    }

    engine.shutdown().await?;
    Ok(())
}