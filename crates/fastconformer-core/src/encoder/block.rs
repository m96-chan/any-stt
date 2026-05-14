//! Full Conformer block.
//!
//! ```text
//! x ← x + 0.5 * FF1(x)         // macaron half
//! x ← x + MHA(x)
//! x ← x + ConvModule(x)
//! x ← x + 0.5 * FF2(x)         // macaron half
//! x ← LN_post(x)
//! ```

use crate::encoder::attention::{AttentionMode, MultiHeadAttention};
use crate::encoder::conv_module::ConvModule;
use crate::encoder::ff::FeedForward;
use crate::encoder::ops::layer_norm_rows;

pub struct ConformerBlock {
    pub ff1: FeedForward,
    pub attn: MultiHeadAttention,
    /// Attention pattern. `Full` for parakeet, `LocalGlobal` for
    /// reazonspeech (Longformer).
    pub attn_mode: AttentionMode,
    pub conv: ConvModule,
    pub ff2: FeedForward,
    /// Final LayerNorm γ (length d_model).
    pub ln_post_gamma: Vec<f32>,
    /// Final LayerNorm β.
    pub ln_post_beta: Vec<f32>,
    pub d_model: usize,
}

impl ConformerBlock {
    /// Apply the full block forward in place: `x ← block(x)`.
    /// `x` is `[time, d_model]` row-major.
    pub fn forward(&self, x: &mut [f32], time: usize) {
        self.forward_masked(x, time, time);
    }

    /// Same as `forward` but applies a pad_mask before the depthwise conv,
    /// zeroing frames `[valid_frames..time]`. Used by the encoder when the
    /// input has trailing padded frames.
    pub fn forward_masked(&self, x: &mut [f32], time: usize, valid_frames: usize) {
        debug_assert_eq!(x.len(), time * self.d_model);
        self.ff1.forward_residual(x, time);
        self.attn.forward_residual_with_mode(x, time, self.attn_mode);
        self.conv.forward_residual_masked(x, time, valid_frames);
        self.ff2.forward_residual(x, time);
        layer_norm_rows(
            x,
            time,
            self.d_model,
            &self.ln_post_gamma,
            &self.ln_post_beta,
            1e-5,
        );
    }

    /// Forward with intermediate dumps; for `tests/layer_reference.rs`.
    pub fn forward_dump(
        &self,
        x: &mut [f32],
        time: usize,
        valid_frames: usize,
        mut dump: impl FnMut(&str, &[f32]),
    ) {
        self.ff1.forward_residual(x, time);
        dump("after_ff1", x);
        self.attn.forward_residual_with_mode(x, time, self.attn_mode);
        dump("after_attn", x);
        self.conv.forward_residual_masked(x, time, valid_frames);
        dump("after_conv", x);
        self.ff2.forward_residual(x, time);
        dump("after_ff2", x);
        layer_norm_rows(
            x,
            time,
            self.d_model,
            &self.ln_post_gamma,
            &self.ln_post_beta,
            1e-5,
        );
        dump("block_out", x);
    }
}
