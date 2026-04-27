//! FastConformer encoder (rel-pos attention, no Longformer).
//!
//! Shared shape with `reazonspeech-backend::encoder` — only the attention
//! variant differs. Future cleanup: extract the encoder body into
//! `fastconformer-core` once both crates have a working forward.
//!
//! ## Status
//! Stub. See `reazonspeech-backend/src/encoder.rs` for the parallel
//! TODO list — most of the work transfers here unchanged.

use fastconformer_core::{Config, MelSpectrogram};

/// Encoder output: `[n_frames_after_subsample, d_model]` row-major f32.
pub struct EncoderOutput {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub d_model: usize,
}

pub struct FastConformerEncoder {
    cfg: Config,
}

impl FastConformerEncoder {
    pub fn from_gguf(_gguf: &gguf_loader::GgufFile, cfg: Config) -> Result<Self, String> {
        Ok(Self { cfg })
    }

    pub fn forward(&self, _mel: &MelSpectrogram) -> Result<EncoderOutput, String> {
        Err("FastConformerEncoder::forward not yet implemented (parakeet)".into())
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }
}
