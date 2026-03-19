use std::path::PathBuf;

use crate::config::Backend;

/// Errors that can occur during STT operations.
#[derive(Debug, thiserror::Error)]
pub enum SttError {
    #[error("model not found: {path}")]
    ModelNotFound { path: PathBuf },

    #[error("backend unavailable: {backend:?} — {reason}")]
    BackendUnavailable { backend: Backend, reason: String },

    #[error("transcription failed: {0}")]
    TranscriptionFailed(String),

    #[error("not implemented: {0}")]
    NotImplemented(String),
}
