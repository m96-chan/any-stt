//! FastConformer encoder (NeMo variant, Longformer local + global attention).
//!
//! Each encoder block follows the Conformer pattern with two feed-forward
//! "macaron" halves around attention and convolution:
//!
//! ```text
//! x ──►┐
//!      ▼
//!    ff1   → 0.5 × FFN (pre-LN, swish, linear, dropout)
//!      │
//!      ▼
//!    attn  → Longformer rel-pos (local window 256, 1 global token)
//!      │
//!      ▼
//!    conv  → pointwise / depthwise / GLU / batchnorm (kernel 9)
//!      │
//!      ▼
//!    ff2   → 0.5 × FFN
//!      │
//!      ▼
//!   ln_post
//! ```
//!
//! ## Status
//! Stub. The structures are defined so that QNN / Vulkan graph builders in
//! `qnn-backend` and `nnapi-backend` can target them, but the CPU ggml
//! implementation is not yet wired.
//!
//! ## NPU offload plan
//! - `attn.q/k/v/out` Linear projections → NPU MatMul (INT8 preferred)
//! - `conv.pw1/dw/pw2` Convolutions → NPU Conv1d (target layers with
//!   depth < 16 for NNAPI)
//! - `ff1/ff2.fc1/fc2` → NPU MatMul
//! - LayerNorm + softmax + residuals → CPU
//! - Positional bias (pos_bias_u/v) → CPU (small)

use crate::config::ReazonSpeechConfig;
use crate::mel::MelSpectrogram;

/// Encoder output: acoustic representation ready for the RNN-T joint network.
///
/// Layout: `[n_frames_after_subsample, d_model]` row-major f32.
pub struct EncoderOutput {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub d_model: usize,
}

/// FastConformer encoder handle.
///
/// Holds dequantized weights + any NPU graph handles. Thread-safe as long
/// as `forward` is not called concurrently on the same instance.
pub struct FastConformerEncoder {
    cfg: ReazonSpeechConfig,
    // TODO(#N4): fields for weights + optional NPU graph handle
}

impl FastConformerEncoder {
    /// Build an encoder from the GGUF model.
    ///
    /// # Errors
    /// Returns a descriptive error if expected tensors are missing.
    pub fn from_gguf(
        _gguf: &gguf_loader::GgufFile,
        cfg: ReazonSpeechConfig,
    ) -> Result<Self, String> {
        // TODO(#N4): dequantize all encoder.* tensors and cache.
        //   For each layer L in 0..cfg.n_layers, load:
        //     - enc.block.L.ff1.{ln,fc1,fc2}.{weight,bias}
        //     - enc.block.L.attn.{ln,q,k,v,out,pos,pos_bias_u,pos_bias_v}
        //     - enc.block.L.conv.{ln,pw1,dw,bn,pw2}.{weight,bias,running_*}
        //     - enc.block.L.ff2.{ln,fc1,fc2}.{weight,bias}
        //     - enc.block.L.ln_post.{weight,bias}
        //   Plus enc.subsample.*, enc.pos.pe, enc.ln_post.*
        Ok(Self { cfg })
    }

    /// Run the encoder on a mel spectrogram.
    pub fn forward(&self, _mel: &MelSpectrogram) -> Result<EncoderOutput, String> {
        // TODO(#N4): implement forward pass.
        //   1. Subsampling (striding conv2d ×8) → [T/8, d_model]
        //   2. Add positional encoding (relative)
        //   3. For each block:
        //      a. ff1 half-residual
        //      b. multi-head attention (Longformer local + global)
        //      c. conv module
        //      d. ff2 half-residual
        //      e. ln_post
        //   4. Final layernorm
        Err("FastConformerEncoder::forward not yet implemented".into())
    }

    pub fn config(&self) -> &ReazonSpeechConfig {
        &self.cfg
    }
}
