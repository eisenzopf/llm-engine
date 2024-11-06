use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use std::collections::HashMap;

/// Tracks token usage and provides statistics
#[derive(Clone)]
pub struct TokenCounter {
    stats: Arc<TokenStats>,
    history: Arc<RwLock<TokenHistory>>,
    config: Arc<TokenConfig>,
}

/// Token usage statistics
pub struct TokenStats {
    total_input_tokens: AtomicUsize,
    total_output_tokens: AtomicUsize,
    total_requests: AtomicUsize,
    failed_requests: AtomicUsize,
}

/// Configuration for token counting
#[derive(Clone)]
pub struct TokenConfig {
    /// Maximum tokens per request
    pub max_tokens_per_request: usize,
    /// History window size in seconds
    pub history_window_secs: u64,
    /// Rate limit per minute
    pub rate_limit_per_minute: Option<usize>,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_request: 4096,
            history_window_secs: 3600, // 1 hour
            rate_limit_per_minute: None,
        }
    }
}

/// Historical token usage data
#[derive(Default)]
struct TokenHistory {
    /// Token usage entries by timestamp
    entries: Vec<TokenEntry>,
    /// Token usage by model
    model_usage: HashMap<String, ModelUsage>,
}

/// Single token usage entry
#[derive(Clone)]
struct TokenEntry {
    timestamp: Instant,
    input_tokens: usize,
    output_tokens: usize,
    model: String,
    processing_time: Duration,
}

/// Usage statistics for a specific model
#[derive(Default, Clone)]
struct ModelUsage {
    total_tokens: usize,
    request_count: usize,
    average_tokens_per_request: f64,
    total_processing_time: Duration,
}

impl TokenCounter {
    /// Create a new token counter
    pub fn new(config: TokenConfig) -> Self {
        Self {
            stats: Arc::new(TokenStats {
                total_input_tokens: AtomicUsize::new(0),
                total_output_tokens: AtomicUsize::new(0),
                total_requests: AtomicUsize::new(0),
                failed_requests: AtomicUsize::new(0),
            }),
            history: Arc::new(RwLock::new(TokenHistory::default())),
            config: Arc::new(config),
        }
    }

    /// Record token usage for a request
    pub fn record_usage(
        &self,
        input_tokens: usize,
        output_tokens: usize,
        model: &str,
        processing_time: Duration,
    ) -> Result<(), TokenError> {
        // Check token limits
        if input_tokens + output_tokens > self.config.max_tokens_per_request {
            return Err(TokenError::TokenLimitExceeded {
                requested: input_tokens + output_tokens,
                limit: self.config.max_tokens_per_request,
            });
        }

        // Check rate limits if configured
        if let Some(limit) = self.config.rate_limit_per_minute {
            let usage = self.get_recent_usage(Duration::from_secs(60));
            if usage >= limit {
                return Err(TokenError::RateLimitExceeded {
                    limit,
                    window: Duration::from_secs(60),
                });
            }
        }

        // Update statistics
        self.stats.total_input_tokens.fetch_add(input_tokens, Ordering::SeqCst);
        self.stats.total_output_tokens.fetch_add(output_tokens, Ordering::SeqCst);
        self.stats.total_requests.fetch_add(1, Ordering::SeqCst);

        // Update history
        let mut history = self.history.write();
        
        // Cleanup old entries
        let cutoff = Instant::now() - Duration::from_secs(self.config.history_window_secs);
        history.entries.retain(|e| e.timestamp > cutoff);
        
        // Add new entry
        history.entries.push(TokenEntry {
            timestamp: Instant::now(),
            input_tokens,
            output_tokens,
            model: model.to_string(),
            processing_time,
        });

        // Update model usage statistics
        let entry = history.model_usage.entry(model.to_string())
            .or_default();
        
        entry.total_tokens += input_tokens + output_tokens;
        entry.request_count += 1;
        entry.average_tokens_per_request = entry.total_tokens as f64 / entry.request_count as f64;
        entry.total_processing_time += processing_time;

        Ok(())
    }

    /// Record a failed request
    pub fn record_failure(&self, reason: &str) {
        self.stats.failed_requests.fetch_add(1, Ordering::SeqCst);
        
        tracing::warn!(
            reason = reason,
            total_failures = self.stats.failed_requests.load(Ordering::SeqCst),
            "Token processing failure"
        );
    }

    /// Get usage in the specified time window
    pub fn get_recent_usage(&self, window: Duration) -> usize {
        let history = self.history.read();
        let cutoff = Instant::now() - window;
        
        history.entries.iter()
            .filter(|e| e.timestamp > cutoff)
            .map(|e| e.input_tokens + e.output_tokens)
            .sum()
    }

    /// Get token usage by model
    pub fn get_model_usage(&self) -> HashMap<String, TokenUsageStats> {
        let history = self.history.read();
        
        history.model_usage.iter()
            .map(|(model, usage)| {
                (model.clone(), TokenUsageStats {
                    total_tokens: usage.total_tokens,
                    request_count: usage.request_count,
                    average_tokens_per_request: usage.average_tokens_per_request,
                    average_processing_time: if usage.request_count > 0 {
                        usage.total_processing_time / usage.request_count as u32
                    } else {
                        Duration::default()
                    },
                })
            })
            .collect()
    }

    /// Get current token usage snapshot
    pub fn snapshot(&self) -> TokenUsageSnapshot {
        TokenUsageSnapshot {
            total_input_tokens: self.stats.total_input_tokens.load(Ordering::SeqCst),
            total_output_tokens: self.stats.total_output_tokens.load(Ordering::SeqCst),
            total_requests: self.stats.total_requests.load(Ordering::SeqCst),
            failed_requests: self.stats.failed_requests.load(Ordering::SeqCst),
            usage_by_model: self.get_model_usage(),
        }
    }
}

/// Token processing error
#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("Token limit exceeded: requested {requested} tokens, limit is {limit}")]
    TokenLimitExceeded {
        requested: usize,
        limit: usize,
    },

    #[error("Rate limit exceeded: {limit} tokens per {window:?}")]
    RateLimitExceeded {
        limit: usize,
        window: Duration,
    },
}

/// Token usage statistics for a model
#[derive(Debug, Clone)]
pub struct TokenUsageStats {
    pub total_tokens: usize,
    pub request_count: usize,
    pub average_tokens_per_request: f64,
    pub average_processing_time: Duration,
}

/// Snapshot of current token usage
#[derive(Debug, Clone)]
pub struct TokenUsageSnapshot {
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
    pub total_requests: usize,
    pub failed_requests: usize,
    pub usage_by_model: HashMap<String, TokenUsageStats>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_counting() {
        let counter = TokenCounter::new(TokenConfig::default());
        
        // Record some usage
        counter.record_usage(100, 50, "gpt-3", Duration::from_millis(100))
            .unwrap();
            
        let snapshot = counter.snapshot();
        assert_eq!(snapshot.total_input_tokens, 100);
        assert_eq!(snapshot.total_output_tokens, 50);
        assert_eq!(snapshot.total_requests, 1);
    }

    #[test]
    fn test_token_limits() {
        let config = TokenConfig {
            max_tokens_per_request: 1000,
            ..Default::default()
        };
        let counter = TokenCounter::new(config);
        
        // Try to exceed limit
        let result = counter.record_usage(800, 300, "gpt-3", Duration::from_millis(100));
        assert!(matches!(result, Err(TokenError::TokenLimitExceeded { .. })));
    }

    #[test]
    fn test_rate_limiting() {
        let config = TokenConfig {
            rate_limit_per_minute: Some(1000),
            ..Default::default()
        };
        let counter = TokenCounter::new(config);
        
        // Record usage up to limit
        for _ in 0..9 {
            counter.record_usage(100, 50, "gpt-3", Duration::from_millis(100))
                .unwrap();
        }
        
        // Next request should fail
        let result = counter.record_usage(100, 50, "gpt-3", Duration::from_millis(100));
        assert!(matches!(result, Err(TokenError::RateLimitExceeded { .. })));
    }

    #[test]
    fn test_model_usage_tracking() {
        let counter = TokenCounter::new(TokenConfig::default());
        
        // Record usage for different models
        counter.record_usage(100, 50, "gpt-3", Duration::from_millis(100))
            .unwrap();
        counter.record_usage(200, 100, "gpt-4", Duration::from_millis(200))
            .unwrap();
        
        let usage = counter.get_model_usage();
        assert_eq!(usage.len(), 2);
        
        let gpt3_stats = usage.get("gpt-3").unwrap();
        assert_eq!(gpt3_stats.total_tokens, 150);
        assert_eq!(gpt3_stats.request_count, 1);
        
        let gpt4_stats = usage.get("gpt-4").unwrap();
        assert_eq!(gpt4_stats.total_tokens, 300);
        assert_eq!(gpt4_stats.request_count, 1);
    }

    #[test]
    fn test_history_window() {
        let config = TokenConfig {
            history_window_secs: 1,
            ..Default::default()
        };
        let counter = TokenCounter::new(config);
        
        // Record usage
        counter.record_usage(100, 50, "gpt-3", Duration::from_millis(100))
            .unwrap();
            
        // Wait for window to expire
        std::thread::sleep(Duration::from_secs(2));
        
        let usage = counter.get_recent_usage(Duration::from_secs(1));
        assert_eq!(usage, 0);
    }

    #[test]
    fn test_failure_tracking() {
        let counter = TokenCounter::new(TokenConfig::default());
        
        counter.record_failure("Test failure");
        
        let snapshot = counter.snapshot();
        assert_eq!(snapshot.failed_requests, 1);
    }
}