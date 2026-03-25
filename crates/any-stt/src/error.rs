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

    #[error("invalid audio: {0}")]
    InvalidAudio(String),

    #[error("not implemented: {0}")]
    NotImplemented(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_model_not_found() {
        let e = SttError::ModelNotFound {
            path: PathBuf::from("/tmp/model.bin"),
        };
        assert!(e.to_string().contains("/tmp/model.bin"));
    }

    #[test]
    fn error_display_backend_unavailable() {
        let e = SttError::BackendUnavailable {
            backend: Backend::Cuda,
            reason: "no GPU".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("Cuda"));
        assert!(msg.contains("no GPU"));
    }

    #[test]
    fn error_display_transcription_failed() {
        let e = SttError::TranscriptionFailed("timeout".into());
        assert!(e.to_string().contains("timeout"));
    }

    #[test]
    fn error_display_invalid_audio() {
        let e = SttError::InvalidAudio("empty".into());
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn error_display_not_implemented() {
        let e = SttError::NotImplemented("QNN".into());
        assert!(e.to_string().contains("QNN"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(SttError::NotImplemented("test".into()));
        assert!(!e.to_string().is_empty());
    }
}
