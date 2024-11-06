use anyhow::Result;
use llm_engine::{
    LLMEngine,
    config::{EngineConfig, ProcessingMode},
    utils::prelude::*,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::sleep;
use rand::Rng;

// Simulated request with priority
struct Request {
    text: String,
    priority: Priority,
}

#[derive(Debug, Clone, Copy)]
enum Priority {
    High,
    Medium,
    Low,
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_logging(LogConfig {
        level: tracing::Level::INFO,
        timestamps: true,
        ..Default::default()
    })?;

    // Configure the engine for queue processing
    let config = EngineConfig {
        processing: ProcessingConfig {
            mode: ProcessingMode::Queue,
            queue_size: Some(1000),
            concurrency: 2,
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
    let engine = Arc::new(
        LLMEngine::builder()
            .with_config(config)
            .build()
            .await?
    );

    // Create channels for request handling
    let (request_tx, mut request_rx) = mpsc::channel::<Request>(100);
    let (result_tx, mut result_rx) = mpsc::channel(100);

    // Spawn request generator
    let request_tx_clone = request_tx.clone();
    tokio::spawn(async move {
        generate_requests(request_tx_clone).await;
    });

    // Spawn result processor
    let result_processor = tokio::spawn(async move {
        let mut high_priority_latency = Vec::new();
        let mut medium_priority_latency = Vec::new();
        let mut low_priority_latency = Vec::new();

        while let Some((priority, latency)) = result_rx.recv().await {
            match priority {
                Priority::High => high_priority_latency.push(latency),
                Priority::Medium => medium_priority_latency.push(latency),
                Priority::Low => low_priority_latency.push(latency),
            }
        }

        // Print latency statistics
        println!("\nLatency Statistics:");
        print_latency_stats("High Priority", high_priority_latency);
        print_latency_stats("Medium Priority", medium_priority_latency);
        print_latency_stats("Low Priority", low_priority_latency);
    });

    // Process requests
    let engine_clone = engine.clone();
    let processor = tokio::spawn(async move {
        while let Some(request) = request_rx.recv().await {
            let start_time = Instant::now();
            let priority_str = format!("{:?}", request.priority);

            // Enqueue request with appropriate priority
            let handle = match request.priority {
                Priority::High => engine_clone.enqueue_with_priority(
                    request.text,
                    QueuePriority::High,
                ).await?,
                Priority::Medium => engine_clone.enqueue_with_priority(
                    request.text,
                    QueuePriority::Medium,
                ).await?,
                Priority::Low => engine_clone.enqueue_with_priority(
                    request.text,
                    QueuePriority::Low,
                ).await?,
            };

            // Process result
            let result = handle.await?;
            let latency = start_time.elapsed();

            info!(
                priority = priority_str,
                latency_ms = latency.as_millis(),
                tokens = result.tokens.len(),
                "Request processed"
            );

            result_tx.send((request.priority, latency)).await?;
        }
        Ok::<_, anyhow::Error>(())
    });

    // Wait for completion or timeout
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("Shutting down...");
        }
        _ = tokio::time::sleep(Duration::from_secs(300)) => {
            info!("Benchmark complete");
        }
    }

    // Cleanup
    drop(request_tx);
    processor.await??;
    result_processor.await?;
    engine.shutdown().await?;
    
    Ok(())
}

async fn generate_requests(tx: mpsc::Sender<Request>) {
    let mut rng = rand::thread_rng();
    let priorities = [Priority::High, Priority::Medium, Priority::Low];

    for i in 0..1000 {
        let priority = priorities[rng.gen_range(0..3)];
        let request = Request {
            text: format!("Test request {}", i),
            priority,
        };

        tx.send(request).await.unwrap();
        
        // Random delay between requests
        sleep(Duration::from_millis(rng.gen_range(10..100))).await;
    }
}

fn print_latency_stats(label: &str, mut latencies: Vec<Duration>) {
    if latencies.is_empty() {
        println!("{}: No requests processed", label);
        return;
    }

    latencies.sort();
    let total: Duration = latencies.iter().sum();
    let avg = total / latencies.len() as u32;
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    println!("{} Latencies:", label);
    println!("  Count: {}", latencies.len());
    println!("  Average: {:.2?}", avg);
    println!("  P50: {:.2?}", p50);
    println!("  P95: {:.2?}", p95);
    println!("  P99: {:.2?}", p99);
}