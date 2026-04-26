use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors crossing the L2 boundary between [`crate::MlOps`] implementations
/// and the PyO3 wheel.
///
/// All variants are `Serialize`/`Deserialize` so error payloads can travel
/// through `MlOps::call_op` envelopes without losing structure.
#[derive(Error, Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum MlError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("inference failed: {0}")]
    InferenceFailed(String),

    #[error("operation '{0}' is not supported by this build")]
    UnsupportedOp(String),

    #[error("payload schema mismatch in '{op}': {message}")]
    SchemaMismatch { op: String, message: String },

    #[error("internal error: {0}")]
    Internal(String),
}

impl MlError {
    pub fn invalid<S: Into<String>>(message: S) -> Self {
        MlError::InvalidRequest(message.into())
    }

    pub fn inference<S: Into<String>>(message: S) -> Self {
        MlError::InferenceFailed(message.into())
    }

    pub fn unsupported<S: Into<String>>(op: S) -> Self {
        MlError::UnsupportedOp(op.into())
    }

    pub fn schema<S1: Into<String>, S2: Into<String>>(op: S1, message: S2) -> Self {
        MlError::SchemaMismatch {
            op: op.into(),
            message: message.into(),
        }
    }

    pub fn internal<S: Into<String>>(message: S) -> Self {
        MlError::Internal(message.into())
    }
}
