//! Full Conformer block.
//!
//! ```text
//! x ← x + 0.5 * FF1(x)         // macaron half
//! x ← x + MHA(x)
//! x ← x + ConvModule(x)
//! x ← x + 0.5 * FF2(x)         // macaron half
//! x ← LN_post(x)
//! ```

use crate::encoder::attention::MultiHeadAttention;
use crate::encoder::conv_module::ConvModule;
use crate::encoder::ff::FeedForward;
use crate::encoder::ops::layer_norm_rows;

pub struct ConformerBlock {
    pub ff1: FeedForward,
    pub attn: MultiHeadAttention,
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
        debug_assert_eq!(x.len(), time * self.d_model);
        self.ff1.forward_residual(x, time);
        self.attn.forward_residual(x, time);
        self.conv.forward_residual(x, time);
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
}
