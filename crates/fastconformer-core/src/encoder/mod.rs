//! FastConformer encoder.
//!
//! Forward path:
//! ```text
//!   mel [T, n_mels]
//!     ↓ subsample (Conv2d ×3 stride 2 + Linear)
//!   feat [T/8, d_model]
//!     ↓ block × n_layers   (FF1 → MHA → Conv → FF2 → LN_post)
//!   enc  [T/8, d_model]
//! ```
//!
//! ## Status
//! Pure-Rust f32 forward. Produces shape-correct outputs from any GGUF
//! that follows the converter's tensor naming. Numerical equivalence
//! against NeMo is validated via `tests/layer_reference.rs` once
//! reference fixtures are present.
//!
//! ## NPU offload (future)
//! Each block's attention and conv MatMuls can be replaced by
//! qnn-backend / nnapi-backend graphs with the same input/output shape.
//! The composition logic in [`FastConformerEncoder::forward`] stays.

pub mod attention;
pub mod block;
pub mod conv_module;
pub mod ff;
pub mod ops;
pub mod subsample;

pub use ops::{
    batch_norm_1d_inference, glu_channels, layer_norm, layer_norm_rows, linear, matvec_add,
    softmax_inplace, swish_inplace,
};

use crate::config::{AttentionType, Config};
pub use attention::MultiHeadAttention;
pub use block::ConformerBlock;
pub use conv_module::ConvModule;
pub use ff::FeedForward;
pub use subsample::{subsampled_mel_dim, subsampled_time_dim, Subsample};

/// Output of the encoder forward pass.
#[derive(Debug)]
pub struct EncoderOutput {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub d_model: usize,
}

/// Full FastConformer encoder.
pub struct FastConformerEncoder {
    pub cfg: Config,
    pub subsample: Subsample,
    pub blocks: Vec<ConformerBlock>,
    /// Final LayerNorm γ over the last block's output (length d_model).
    pub ln_post_gamma: Vec<f32>,
    pub ln_post_beta: Vec<f32>,
}

impl FastConformerEncoder {
    /// Run the encoder on a `[time, n_mels]` mel spectrogram.
    pub fn forward(&self, mel: &[f32], n_frames: usize) -> Result<EncoderOutput, String> {
        let dm = self.cfg.d_model as usize;
        if self.cfg.attention_type == AttentionType::RelPosLocalAttn {
            // TODO: Longformer (local + global) attention. Until that's
            // wired, callers using ReazonSpeech will get an early error
            // here rather than silently using vanilla attention.
            return Err(
                "FastConformerEncoder: Longformer attention not yet implemented; \
                 only AttentionType::RelPos is supported in this build"
                    .into(),
            );
        }

        // 1) Subsample.
        let (mut x, t_out) = self.subsample.forward(mel, n_frames);

        // 2) N Conformer blocks.
        for block in &self.blocks {
            block.forward(&mut x, t_out);
        }

        // 3) Final LN.
        layer_norm_rows(&mut x, t_out, dm, &self.ln_post_gamma, &self.ln_post_beta, 1e-5);

        Ok(EncoderOutput {
            data: x,
            n_frames: t_out,
            d_model: dm,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_block(d_model: usize, n_heads: usize, conv_kernel: usize) -> ConformerBlock {
        ConformerBlock {
            ff1: FeedForward {
                ln_gamma: vec![1.0; d_model],
                ln_beta: vec![0.0; d_model],
                fc1_weight: vec![0.0; 4 * d_model * d_model],
                fc1_bias: vec![0.0; 4 * d_model],
                fc2_weight: vec![0.0; d_model * 4 * d_model],
                fc2_bias: vec![0.0; d_model],
                d_model,
            },
            attn: MultiHeadAttention {
                d_model,
                n_heads,
                head_dim: d_model / n_heads,
                ln_gamma: vec![1.0; d_model],
                ln_beta: vec![0.0; d_model],
                q_weight: vec![0.0; d_model * d_model],
                q_bias: vec![0.0; d_model],
                k_weight: vec![0.0; d_model * d_model],
                v_weight: vec![0.0; d_model * d_model],
                v_bias: vec![0.0; d_model],
                out_weight: vec![0.0; d_model * d_model],
                out_bias: vec![0.0; d_model],
                pos_weight: vec![0.0; d_model * d_model],
                pos_bias_u: vec![0.0; n_heads * (d_model / n_heads)],
                pos_bias_v: vec![0.0; n_heads * (d_model / n_heads)],
            },
            conv: ConvModule {
                d_model,
                conv_kernel_size: conv_kernel,
                ln_gamma: vec![1.0; d_model],
                ln_beta: vec![0.0; d_model],
                pw1_weight: vec![0.0; 2 * d_model * d_model],
                pw1_bias: vec![0.0; 2 * d_model],
                dw_weight: vec![0.0; d_model * conv_kernel],
                dw_bias: vec![0.0; d_model],
                bn_gamma: vec![1.0; d_model],
                bn_beta: vec![0.0; d_model],
                bn_running_mean: vec![0.0; d_model],
                bn_running_var: vec![1.0; d_model],
                pw2_weight: vec![0.0; d_model * d_model],
                pw2_bias: vec![0.0; d_model],
            },
            ff2: FeedForward {
                ln_gamma: vec![1.0; d_model],
                ln_beta: vec![0.0; d_model],
                fc1_weight: vec![0.0; 4 * d_model * d_model],
                fc1_bias: vec![0.0; 4 * d_model],
                fc2_weight: vec![0.0; d_model * 4 * d_model],
                fc2_bias: vec![0.0; d_model],
                d_model,
            },
            ln_post_gamma: vec![1.0; d_model],
            ln_post_beta: vec![0.0; d_model],
            d_model,
        }
    }

    #[test]
    fn zero_encoder_produces_finite_output_with_correct_shape() {
        // Tiny config: 4 mels, 8 d_model, 1 layer, 2 heads.
        let mut cfg = Config::dummy_parakeet_tdt_v3();
        cfg.feat_in = 8;
        cfg.n_mels = 8;
        cfg.d_model = 8;
        cfg.n_layers = 1;
        cfg.n_heads = 2;
        cfg.conv_kernel_size = 3;

        let n_mels_out = subsampled_mel_dim(8); // 8→4→2→1 = 1
        let channels = 8;
        let feat_per_step = channels * n_mels_out;

        let subsample = Subsample {
            n_mels: 8,
            d_model: 8,
            n_mels_out,
            channels,
            conv0_weight: vec![0.0; channels * 1 * 9],
            conv0_bias: vec![0.0; channels],
            conv1_weight: vec![0.0; channels * channels * 9],
            conv1_bias: vec![0.0; channels],
            conv2_weight: vec![0.0; channels * channels * 9],
            conv2_bias: vec![0.0; channels],
            out_weight: vec![0.0; 8 * feat_per_step],
            out_bias: vec![0.0; 8],
        };

        let blocks = vec![zero_block(8, 2, 3)];
        let enc = FastConformerEncoder {
            cfg,
            subsample,
            blocks,
            ln_post_gamma: vec![1.0; 8],
            ln_post_beta: vec![0.0; 8],
        };

        let mel = vec![0.0_f32; 16 * 8];
        let out = enc.forward(&mel, 16).unwrap();
        assert_eq!(out.d_model, 8);
        assert_eq!(out.n_frames, subsampled_time_dim(16));
        assert_eq!(out.data.len(), out.n_frames * out.d_model);
        for v in &out.data {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn longformer_attention_returns_clear_error() {
        let mut cfg = Config::dummy_reazonspeech_nemo_v2();
        cfg.n_layers = 0; // skip any block work
        cfg.feat_in = 8;
        cfg.n_mels = 8;
        cfg.d_model = 8;

        let n_mels_out = subsampled_mel_dim(8);
        let channels = 8;
        let feat_per_step = channels * n_mels_out;
        let subsample = Subsample {
            n_mels: 8,
            d_model: 8,
            n_mels_out,
            channels,
            conv0_weight: vec![0.0; channels * 9],
            conv0_bias: vec![0.0; channels],
            conv1_weight: vec![0.0; channels * channels * 9],
            conv1_bias: vec![0.0; channels],
            conv2_weight: vec![0.0; channels * channels * 9],
            conv2_bias: vec![0.0; channels],
            out_weight: vec![0.0; 8 * feat_per_step],
            out_bias: vec![0.0; 8],
        };
        let enc = FastConformerEncoder {
            cfg,
            subsample,
            blocks: vec![],
            ln_post_gamma: vec![1.0; 8],
            ln_post_beta: vec![0.0; 8],
        };
        let mel = vec![0.0_f32; 16 * 8];
        let err = enc.forward(&mel, 16).unwrap_err();
        assert!(err.contains("Longformer"), "got: {err}");
    }
}
