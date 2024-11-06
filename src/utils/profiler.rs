use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use parking_lot::{Mutex, RwLock};
use lazy_static::lazy_static;
use tokio::time::Duration;
use tokio::time::Instant;

lazy_static! {
    static ref GLOBAL_PROFILER: Arc<Profiler> = Arc::new(Profiler::new());
}

/// Performance profiler for tracking execution times
#[derive(Debug)]
pub struct Profiler {
    spans: RwLock<HashMap<String, SpanStats>>,
    active_spans: Mutex<Vec<ActiveSpan>>,
}

/// Statistics for a profiled span
#[derive(Debug, Clone)]
struct SpanStats {
    count: usize,
    total_time: Duration,
    min_time: Duration,
    max_time: Duration,
    avg_time: Duration,
}

/// Currently active profiling span
#[derive(Debug)]
struct ActiveSpan {
    name: String,
    start_time: Instant,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            spans: RwLock::new(HashMap::new()),
            active_spans: Mutex::new(Vec::new()),
        }
    }

    /// Get the global profiler instance
    pub fn global() -> Arc<Profiler> {
        GLOBAL_PROFILER.clone()
    }

    /// Start a new profiling span
    pub fn start_span(&self, name: impl Into<String>) -> ProfilerGuard {
        let name = name.into();
        let start_time = Instant::now();
        
        self.active_spans.lock().unwrap().push(ActiveSpan {
            name: name.clone(),
            start_time,
        });

        ProfilerGuard {
            profiler: self,
            name,
            start_time,
        }
    }

    /// Record a completed span
    fn record_span(&self, name: &str, duration: Duration) {
        let mut spans = self.spans.write();
        let stats = spans.entry(name.to_string()).or_insert_with(|| SpanStats {
            count: 0,
            total_time: Duration::default(),
            min_time: duration,
            max_time: duration,
            avg_time: duration,
        });

        stats.count += 1;
        stats.total_time += duration;
        stats.min_time = stats.min_time.min(duration);
        stats.max_time = stats.max_time.max(duration);
        stats.avg_time = stats.total_time / stats.count as u32;
    }

    /// Get statistics for all spans
    pub fn get_stats(&self) -> HashMap<String, ProfileStats> {
        let spans = self.spans.read();
        spans.iter()
            .map(|(name, stats)| {
                (name.clone(), ProfileStats {
                    count: stats.count,
                    total_time: stats.total_time,
                    min_time: stats.min_time,
                    max_time: stats.max_time,
                    avg_time: stats.avg_time,
                })
            })
            .collect()
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.spans.write().clear();
        self.active_spans.lock().unwrap().clear();
    }
}

/// Guard for a profiling span
pub struct ProfilerGuard<'a> {
    profiler: &'a Profiler,
    name: String,
    start_time: Instant,
}

impl<'a> Drop for ProfilerGuard<'a> {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        self.profiler.record_span(&self.name, duration);
        
        // Remove from active spans
        let mut active_spans = self.profiler.active_spans.lock().unwrap();
        if let Some(pos) = active_spans.iter().position(|s| s.name == self.name) {
            active_spans.remove(pos);
        }
    }
}

impl<'a> Drop for ProfilerGuard<'a> {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        
        // Use parking_lot's mutex API
        let mut spans = self.profiler.spans.write();
        if let Some(stats) = spans.get_mut(&self.name) {
            stats.update(duration);
        } else {
            spans.insert(self.name.clone(), SpanStats::new(duration));
        }
        
        let mut active = self.profiler.active_spans.lock();
        if let Some(pos) = active.iter().position(|s| s.name == self.name) {
            active.remove(pos);
        }
    }
}

/// Public statistics for a profiled span
#[derive(Debug, Clone)]
pub struct ProfileStats {
    pub count: usize,
    pub total_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub avg_time: Duration,
}

impl ProfileStats {
    /// Calculate operations per second
    pub fn ops_per_second(&self) -> f64 {
        self.count as f64 / self.total_time.as_secs_f64()
    }
}

/// Convenience macro for profiling a scope
#[macro_export]
macro_rules! profile_span {
    ($name:expr) => {
        let _guard = crate::utils::Profiler::global().start_span($name);
    };
}

/// Span for detailed performance profiling
pub struct ProfileSpan {
    name: String,
    start: Instant,
    events: Vec<ProfileEvent>,
}

/// Event within a profile span
#[derive(Debug)]
struct ProfileEvent {
    name: String,
    timestamp: Instant,
    data: HashMap<String, String>,
}

impl ProfileSpan {
    /// Create a new profile span
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
            events: Vec::new(),
        }
    }

    /// Record an event with optional data
    pub fn event(&mut self, name: impl Into<String>) -> &mut Self {
        self.events.push(ProfileEvent {
            name: name.into(),
            timestamp: Instant::now(),
            data: HashMap::new(),
        });
        self
    }

    /// Add data to the last event
    pub fn data(&mut self, key: impl Into<String>, value: impl ToString) -> &mut Self {
        if let Some(event) = self.events.last_mut() {
            event.data.insert(key.into(), value.to_string());
        }
        self
    }

    /// Complete the span and return timing information
    pub fn complete(self) -> SpanTiming {
        let duration = self.start.elapsed();
        let mut event_timings = Vec::new();

        let mut last_time = self.start;
        for event in self.events {
            let event_duration = event.timestamp.duration_since(last_time);
            event_timings.push(EventTiming {
                name: event.name,
                duration: event_duration,
                data: event.data,
            });
            last_time = event.timestamp;
        }

        SpanTiming {
            name: self.name,
            total_duration: duration,
            events: event_timings,
        }
    }
}

/// Timing information for a completed span
#[derive(Debug)]
pub struct SpanTiming {
    pub name: String,
    pub total_duration: Duration,
    pub events: Vec<EventTiming>,
}

/// Timing information for an event
#[derive(Debug)]
pub struct EventTiming {
    pub name: String,
    pub duration: Duration,
    pub data: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_profiling() {
        let profiler = Profiler::new();
        
        // Profile some operations
        {
            let _guard = profiler.start_span("test_op");
            thread::sleep(Duration::from_millis(100));
        }

        let stats = profiler.get_stats();
        assert_eq!(stats.len(), 1);
        
        let test_stats = stats.get("test_op").unwrap();
        assert_eq!(test_stats.count, 1);
        assert!(test_stats.total_time >= Duration::from_millis(100));
    }

    #[test]
    fn test_nested_spans() {
        let profiler = Profiler::new();
        
        {
            let _outer = profiler.start_span("outer");
            thread::sleep(Duration::from_millis(50));
            
            {
                let _inner = profiler.start_span("inner");
                thread::sleep(Duration::from_millis(50));
            }
        }

        let stats = profiler.get_stats();
        assert_eq!(stats.len(), 2);
        
        assert!(stats.get("outer").unwrap().total_time >= Duration::from_millis(100));
        assert!(stats.get("inner").unwrap().total_time >= Duration::from_millis(50));
    }

    #[test]
    fn test_profile_macro() {
        {
            profile_span!("macro_test");
            thread::sleep(Duration::from_millis(50));
        }

        let stats = Profiler::global().get_stats();
        assert!(stats.contains_key("macro_test"));
    }

    #[test]
    fn test_detailed_profiling() {
        let mut span = ProfileSpan::new("complex_operation");
        
        span.event("start")
            .data("batch_size", 32);
        thread::sleep(Duration::from_millis(50));
        
        span.event("middle_stage")
            .data("progress", "50%");
        thread::sleep(Duration::from_millis(50));
        
        span.event("complete")
            .data("status", "success");
        
        let timing = span.complete();
        
        assert_eq!(timing.events.len(), 3);
        assert!(timing.total_duration >= Duration::from_millis(100));
        assert_eq!(timing.events[0].data.get("batch_size").unwrap(), "32");
    }

    #[test]
    fn test_stats_calculation() {
        let profiler = Profiler::new();
        
        // Run operation multiple times
        for _ in 0..5 {
            let _guard = profiler.start_span("repeated_op");
            thread::sleep(Duration::from_millis(20));
        }

        let stats = profiler.get_stats().get("repeated_op").unwrap().clone();
        
        assert_eq!(stats.count, 5);
        assert!(stats.avg_time >= Duration::from_millis(20));
        assert!(stats.min_time <= stats.avg_time);
        assert!(stats.max_time >= stats.avg_time);
        assert!(stats.ops_per_second() > 0.0);
    }
}