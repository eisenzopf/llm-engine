use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use std::collections::HashMap;

/// Tracks memory usage across the system
#[derive(Clone)]
pub struct MemoryTracker {
    stats: Arc<MemoryStats>,
    allocations: Arc<RwLock<HashMap<String, AllocationInfo>>>,
}

/// Memory usage statistics
pub struct MemoryStats {
    current_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    total_allocations: AtomicUsize,
    failed_allocations: AtomicUsize,
}

/// Information about a memory allocation
#[derive(Debug, Clone)]
struct AllocationInfo {
    size: u64,
    timestamp: Instant,
    stack_trace: String,
}

/// Snapshot of memory statistics
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub total_allocations: usize,
    pub failed_allocations: usize,
    pub allocation_rate: f64,
    pub largest_allocation: u64,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self {
            stats: Arc::new(MemoryStats {
                current_bytes: AtomicU64::new(0),
                peak_bytes: AtomicU64::new(0),
                total_allocations: AtomicUsize::new(0),
                failed_allocations: AtomicUsize::new(0),
            }),
            allocations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Track a memory allocation
    pub fn track_allocation(&self, size: u64, context: &str) {
        let current = self.stats.current_bytes.fetch_add(size, Ordering::SeqCst);
        self.stats.peak_bytes.fetch_max(current + size, Ordering::SeqCst);
        self.stats.total_allocations.fetch_add(1, Ordering::SeqCst);
        
        let stack_trace = std::backtrace::Backtrace::force_capture()
            .to_string();

        let mut allocations = self.allocations.write();
        allocations.insert(context.to_string(), AllocationInfo {
            size,
            timestamp: Instant::now(),
            stack_trace,
        });
    }

    /// Track a failed allocation attempt
    pub fn track_failed_allocation(&self, size: u64, reason: &str) {
        self.stats.failed_allocations.fetch_add(1, Ordering::SeqCst);
        
        tracing::warn!(
            size_bytes = size,
            reason = reason,
            "Failed memory allocation"
        );
    }

    /// Release tracked memory
    pub fn track_release(&self, size: u64, context: &str) {
        self.stats.current_bytes.fetch_sub(size, Ordering::SeqCst);
        
        let mut allocations = self.allocations.write();
        allocations.remove(context);
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> u64 {
        self.stats.current_bytes.load(Ordering::SeqCst)
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> u64 {
        self.stats.peak_bytes.load(Ordering::SeqCst)
    }

    /// Get memory statistics snapshot
    pub fn snapshot(&self) -> MemorySnapshot {
        let allocations = self.allocations.read();
        let allocation_times: Vec<_> = allocations.values()
            .map(|info| info.timestamp)
            .collect();
        
        let now = Instant::now();
        let oldest = allocation_times.iter().min()
            .copied()
            .unwrap_or(now);
        
        let time_range = now.duration_since(oldest).as_secs_f64();
        let allocation_rate = if time_range > 0.0 {
            allocations.len() as f64 / time_range
        } else {
            0.0
        };

        let largest_allocation = allocations.values()
            .map(|info| info.size)
            .max()
            .unwrap_or(0);

        MemorySnapshot {
            current_bytes: self.stats.current_bytes.load(Ordering::SeqCst),
            peak_bytes: self.stats.peak_bytes.load(Ordering::SeqCst),
            total_allocations: self.stats.total_allocations.load(Ordering::SeqCst),
            failed_allocations: self.stats.failed_allocations.load(Ordering::SeqCst),
            allocation_rate,
            largest_allocation,
        }
    }

    /// Get allocation information for debugging
    pub fn get_allocation_info(&self, context: &str) -> Option<(u64, Duration, String)> {
        let allocations = self.allocations.read();
        allocations.get(context).map(|info| (
            info.size,
            info.timestamp.elapsed(),
            info.stack_trace.clone()
        ))
    }

    /// Reset statistics
    pub fn reset(&self) {
        self.stats.current_bytes.store(0, Ordering::SeqCst);
        self.stats.peak_bytes.store(0, Ordering::SeqCst);
        self.stats.total_allocations.store(0, Ordering::SeqCst);
        self.stats.failed_allocations.store(0, Ordering::SeqCst);
        self.allocations.write().clear();
    }
}

/// Convenience macro for memory tracking
#[macro_export]
macro_rules! track_memory {
    ($tracker:expr, $size:expr, $context:expr) => {
        let _guard = MemoryAllocationGuard::new($tracker, $size, $context);
    };
}

/// RAII guard for memory tracking
pub struct MemoryAllocationGuard<'a> {
    tracker: &'a MemoryTracker,
    size: u64,
    context: String,
}

impl<'a> MemoryAllocationGuard<'a> {
    pub fn new(tracker: &'a MemoryTracker, size: u64, context: impl Into<String>) -> Self {
        let context = context.into();
        tracker.track_allocation(size, &context);
        Self {
            tracker,
            size,
            context,
        }
    }
}

impl<'a> Drop for MemoryAllocationGuard<'a> {
    fn drop(&mut self) {
        self.tracker.track_release(self.size, &self.context);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracking() {
        let tracker = MemoryTracker::new();
        
        // Track some allocations
        tracker.track_allocation(1024, "test1");
        assert_eq!(tracker.current_usage(), 1024);
        
        tracker.track_allocation(2048, "test2");
        assert_eq!(tracker.current_usage(), 3072);
        assert_eq!(tracker.peak_usage(), 3072);
        
        // Release memory
        tracker.track_release(1024, "test1");
        assert_eq!(tracker.current_usage(), 2048);
        assert_eq!(tracker.peak_usage(), 3072); // Peak stays the same
    }

    #[test]
    fn test_memory_guard() {
        let tracker = MemoryTracker::new();
        
        {
            let _guard = MemoryAllocationGuard::new(&tracker, 1024, "guarded");
            assert_eq!(tracker.current_usage(), 1024);
        }
        
        // Memory should be released after guard is dropped
        assert_eq!(tracker.current_usage(), 0);
    }

    #[test]
    fn test_allocation_tracking() {
        let tracker = MemoryTracker::new();
        
        tracker.track_allocation(1024, "tracked_alloc");
        
        if let Some((size, duration, stack_trace)) = tracker.get_allocation_info("tracked_alloc") {
            assert_eq!(size, 1024);
            assert!(duration.as_nanos() > 0);
            assert!(!stack_trace.is_empty());
        } else {
            panic!("Expected allocation info");
        }
    }

    #[test]
    fn test_memory_snapshot() {
        let tracker = MemoryTracker::new();
        
        // Add some allocations
        for i in 0..5 {
            tracker.track_allocation(1024 * (i + 1), &format!("test{}", i));
            std::thread::sleep(Duration::from_millis(10));
        }
        
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.current_bytes, 15360); // Sum of all allocations
        assert_eq!(snapshot.total_allocations, 5);
        assert!(snapshot.allocation_rate > 0.0);
        assert_eq!(snapshot.largest_allocation, 5120);
    }

    #[test]
    fn test_failed_allocations() {
        let tracker = MemoryTracker::new();
        
        tracker.track_failed_allocation(1024 * 1024, "Out of memory");
        
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.failed_allocations, 1);
    }

    #[test]
    fn test_reset() {
        let tracker = MemoryTracker::new();
        
        tracker.track_allocation(1024, "test");
        assert!(tracker.current_usage() > 0);
        
        tracker.reset();
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.peak_usage(), 0);
        
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.total_allocations, 0);
    }
}