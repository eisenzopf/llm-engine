use std::time::Duration;

/// Common processing statistics
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    /// Number of active streams (streaming mode)
    pub active_streams: usize,
    
    /// Total tokens processed
    pub total_processed_tokens: usize,
    
    /// Total uptime
    pub uptime: Duration,
    
    /// Current batch size (batch mode)
    pub current_batch_size: Option<usize>,
    
    /// Current queue size (queue mode)
    pub current_queue_size: Option<usize>,
    
    /// Average wait time in queue
    pub average_wait_time: Option<Duration>,
    
    /// Average processing time per item
    pub average_processing_time: Option<Duration>,
}

impl ProcessingStats {
    /// Calculate tokens per second
    pub fn tokens_per_second(&self) -> f32 {
        let secs = self.uptime.as_secs_f32();
        if secs > 0.0 {
            self.total_processed_tokens as f32 / secs
        } else {
            0.0
        }
    }

    /// Get utilization percentage
    pub fn utilization(&self) -> f32 {
        if let Some(proc_time) = self.average_processing_time {
            proc_time.as_secs_f32() / self.uptime.as_secs_f32() * 100.0
        } else {
            0.0
        }
    }
}

/// Processing result with timing information
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Output text
    pub text: String,
    
    /// Generated tokens
    pub tokens: Vec<String>,
    
    /// Time spent waiting in queue
    pub wait_time: Option<Duration>,
    
    /// Time spent processing
    pub processing_time: Duration,
    
    /// Total tokens processed
    pub token_count: usize,
    
    /// Device used for processing
    pub device_id: usize,
}

impl ProcessingResult {
    /// Calculate tokens per second for this result
    pub fn tokens_per_second(&self) -> f32 {
        let secs = self.processing_time.as_secs_f32();
        if secs > 0.0 {
            self.token_count as f32 / secs
        } else {
            0.0
        }
    }

    /// Get total elapsed time including wait time
    pub fn total_time(&self) -> Duration {
        self.wait_time.unwrap_or_default() + self.processing_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processing_stats() {
        let stats = ProcessingStats {
            active_streams: 1,
            total_processed_tokens: 1000,
            uptime: Duration::from_secs(10),
            current_batch_size: Some(32),
            current_queue_size: None,
            average_wait_time: Some(Duration::from_millis(100)),
            average_processing_time: Some(Duration::from_millis(50)),
        };

        assert_eq!(stats.tokens_per_second(), 100.0);
        assert!(stats.utilization() > 0.0);
    }

    #[test]
    fn test_processing_result() {
        let result = ProcessingResult {
            text: "test".to_string(),
            tokens: vec!["test".to_string()],
            wait_time: Some(Duration::from_millis(100)),
            processing_time: Duration::from_millis(50),
            token_count: 10,
            device_id: 0,
        };

        assert_eq!(result.tokens_per_second(), 200.0);
        assert_eq!(result.total_time(), Duration::from_millis(150));
    }
}