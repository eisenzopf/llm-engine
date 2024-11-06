use std::fmt;
use std::sync::Once;
use tracing::{Level};
use tracing_subscriber::{
    fmt::{format::FmtSpan, self as fmt_subscriber},
    EnvFilter,
};

static INIT: Once = Once::new();

/// Logging configuration options
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Minimum log level
    pub level: Level,
    /// Whether to include timestamps
    pub timestamps: bool,
    /// Whether to include source code locations
    pub source_location: bool,
    /// Whether to log spans
    pub log_spans: bool,
    /// Output file path (None for stdout)
    pub file_path: Option<String>,
    /// Maximum log file size in bytes
    pub max_file_size: Option<u64>,
    /// Maximum number of log files to keep
    pub max_files: Option<usize>,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: Level::INFO,
            timestamps: true,
            source_location: true,
            log_spans: true,
            file_path: None,
            max_file_size: Some(10 * 1024 * 1024), // 10MB
            max_files: Some(5),
        }
    }
}

/// Initialize logging system
pub fn setup_logging(config: LogConfig) -> Result<(), String> {
    let mut result = Ok(());
    
    INIT.call_once(|| {
        result = setup_logging_internal(config);
    });
    
    result
}

fn setup_logging_internal(config: LogConfig) -> Result<(), String> {
    let filter = EnvFilter::from_default_env()
        .add_directive(config.level.into());

    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(true)
        .with_file(config.source_location)
        .with_line_number(config.source_location)
        .with_span_events(if config.log_spans {
            FmtSpan::FULL
        } else {
            FmtSpan::NONE
        });

    let subscriber = if config.timestamps {
        subscriber.with_timer()
    } else {
        subscriber
    };

    if let Some(path) = config.file_path {
        use std::fs::OpenOptions;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("Failed to open log file: {}", e))?;

        subscriber.with_writer(std::sync::Mutex::new(file))
            .try_init()
            .map_err(|e| format!("Failed to set global subscriber: {}", e))?;
    } else {
        subscriber
            .try_init()
            .map_err(|e| format!("Failed to set global subscriber: {}", e))?;
    }

    Ok(())
}

/// Structured log event
#[derive(Debug)]
impl LogEvent<'_> {
    pub fn new(level: Level, message: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
            fields: Vec::new(),
        }
    }

    pub fn field(mut self, key: &'static str, value: impl fmt::Display) -> Self {
        self.fields.push((key, value.to_string()));
        self
    }

    pub fn emit(self) {
        let span = tracing::span!(
            self.level,
            "log_event",
            message = %self.message,
        );
        let _guard = span.enter();

        for (key, value) in self.fields {
            match self.level {
                Level::ERROR => tracing::error!(%key = %value),
                Level::WARN => tracing::warn!(%key = %value),
                Level::INFO => tracing::info!(%key = %value),
                Level::DEBUG => tracing::debug!(%key = %value),
                Level::TRACE => tracing::trace!(%key = %value),
            }
        }
    }
}

/// Log a structured error message
pub fn error(message: impl Into<String>) -> LogEvent<'static> {
    LogEvent::new(Level::ERROR, message)
}

/// Log a structured warning message
pub fn warn(message: impl Into<String>) -> LogEvent<'static> {
    LogEvent::new(Level::WARN, message)
}

/// Log a structured info message
pub fn info(message: impl Into<String>) -> LogEvent<'static> {
    LogEvent::new(Level::INFO, message)
}

/// Log a structured debug message
pub fn debug(message: impl Into<String>) -> LogEvent<'static> {
    LogEvent::new(Level::DEBUG, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;

    #[test]
    fn test_log_initialization() {
        let config = LogConfig::default();
        assert!(setup_logging(config).is_ok());
    }

    #[test]
    fn test_file_logging() {
        let dir = tempdir().unwrap();
        let log_path = dir.path().join("test.log");
        
        let config = LogConfig {
            file_path: Some(log_path.to_str().unwrap().to_string()),
            ..Default::default()
        };
        
        setup_logging(config).unwrap();
        
        // Generate some log messages
        error("Test error").field("key", "value").emit();
        info("Test info").emit();
        
        // Verify log file contents
        let contents = fs::read_to_string(log_path).unwrap();
        assert!(contents.contains("Test error"));
        assert!(contents.contains("Test info"));
        assert!(contents.contains("key=value"));
    }

    #[test]
    fn test_structured_logging() {
        let config = LogConfig::default();
        setup_logging(config).unwrap();

        info("Processing batch")
            .field("batch_size", 32)
            .field("device", "cuda:0")
            .field("memory_used", "8.5GB")
            .emit();
    }

    #[test]
    fn test_log_levels() {
        let config = LogConfig {
            level: Level::DEBUG,
            ..Default::default()
        };
        setup_logging(config).unwrap();

        error("Error message").emit();
        warn("Warning message").emit();
        info("Info message").emit();
        debug("Debug message").emit();
    }
}