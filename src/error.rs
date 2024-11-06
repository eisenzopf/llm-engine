use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;
use crate::ProcessingMode;

// Add Result type alias
pub type Result<T> = std::result::Result<T, EngineError>;

#[derive(Debug)]
pub enum EngineError {
    InitializationError {
        message: String,
        source: Option<Box<dyn StdError + Send + Sync>>,
    },
    GPUError {
        device_id: usize,
        message: String,
        recoverable: bool,
    },
    ModelError {
        message: String,
        source: Option<Box<dyn StdError + Send + Sync>>,
    },
    TimeoutError {
        duration: Duration,
        operation: String,
    },
    ResourceError {
        message: String,
        resource_type: ResourceType,
    },
    ConfigurationError {
        message: String,
        parameter: String,
    },
    ProcessingError {
        message: String,
        source: Option<Box<dyn StdError + Send + Sync>>,
    },
    StreamError {
        stream_id: usize,
        message: String,
    },
    InvalidMode {
        expected: ProcessingMode,
        actual: ProcessingMode,
    },
    DeviceError {
        message: String,
    },
    QueueError {
        message: String,
        queue_size: usize,
    },
}

/// Types of resources that can cause allocation errors
#[derive(Debug, Clone, Copy)]
pub enum ResourceType {
    GPU,
    Memory,
    Threads,
    Other,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InitializationError { message, .. } => {
                write!(f, "Initialization error: {}", message)
            }
            EngineError::GPUError { device_id, message, .. } => {
                write!(f, "GPU {} error: {}", device_id, message)
            }
            EngineError::ModelError { message, .. } => {
                write!(f, "Model error: {}", message)
            }
            EngineError::TimeoutError { duration, operation } => {
                write!(f, "Operation '{}' timed out after {:?}", operation, duration)
            }
            EngineError::ResourceError { message, resource_type } => {
                write!(f, "{:?} resource error: {}", resource_type, message)
            }
            EngineError::ConfigurationError { message, parameter } => {
                write!(f, "Configuration error for {}: {}", parameter, message)
            }
            EngineError::QueueError { message, queue_size } => {
                write!(f, "Queue error (size {}): {}", queue_size, message)
            }
        }
    }
}

impl StdError for EngineError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            EngineError::InitializationError { source, .. } => source.as_ref().map(|s| s.as_ref()),
            EngineError::ModelError { source, .. } => source.as_ref().map(|s| s.as_ref()),
            _ => None,
        }
    }
}

/// Extension trait for error handling utilities
pub(crate) trait ErrorExt {
    fn is_recoverable(&self) -> bool;
    fn should_reduce_batch(&self) -> bool;
}

impl ErrorExt for EngineError {
    fn is_recoverable(&self) -> bool {
        match self {
            EngineError::GPUError { recoverable, .. } => *recoverable,
            EngineError::TimeoutError { .. } => true,
            EngineError::ResourceError { resource_type: ResourceType::Memory, .. } => true,
            _ => false,
        }
    }

    fn should_reduce_batch(&self) -> bool {
        matches!(self,
            EngineError::GPUError { message, .. } if message.contains("out of memory") ||
            EngineError::ResourceError { resource_type: ResourceType::Memory, .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let error = EngineError::GPUError {
            device_id: 0,
            message: "Out of memory".to_string(),
            recoverable: true,
        };
        assert_eq!(error.to_string(), "GPU 0 error: Out of memory");
    }

    #[test]
    fn test_error_recovery_classification() {
        let error = EngineError::GPUError {
            device_id: 0,
            message: "Out of memory".to_string(),
            recoverable: true,
        };
        assert!(error.is_recoverable());
        assert!(error.should_reduce_batch());
    }
}