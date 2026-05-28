//! Conformer feed-forward "macaron" half.
//!
//! Each Conformer block has two FF halves wrapping the attention + conv
//! modules. The output is scaled by 0.5 (the macaron coefficient) before
//! the residual add.
//!
//! ```text
//! y = LN(x) → fc1 → swish → fc2 → 0.5 * y
//! ```
//!
//! `fc1` expands `d_model → 4 * d_model`, `fc2` projects back to `d_model`.

use crate::encoder::ops::{layer_norm, linear_batch, swish_inplace};

/// Weights for one FF macaron half.
pub struct FeedForward {
    /// `[d_model]`
    pub ln_gamma: Vec<f32>,
    /// `[d_model]`
    pub ln_beta: Vec<f32>,
    /// `[4 * d_model, d_model]`
    pub fc1_weight: Vec<f32>,
    /// `[4 * d_model]`
    pub fc1_bias: Vec<f32>,
    /// `[d_model, 4 * d_model]`
    pub fc2_weight: Vec<f32>,
    /// `[d_model]`
    pub fc2_bias: Vec<f32>,
    pub d_model: usize,
}

impl FeedForward {
    /// Apply one macaron half-residual step in place: `x ← x + 0.5 * FF(x)`.
    /// `x` is `[time, d_model]` row-major.
    /// Batched: all time frames processed together through GEMM.
    pub fn forward_residual(&self, x: &mut [f32], time: usize) {
        debug_assert_eq!(x.len(), time * self.d_model);
        let dm = self.d_model;
        let dff = 4 * dm;

        // 1) Layer-normalize all frames into a buffer
        let mut ln_buf = vec![0.0_f32; time * dm];
        ln_buf.copy_from_slice(x);
        for t in 0..time {
            layer_norm(&mut ln_buf[t * dm..(t + 1) * dm], &self.ln_gamma, &self.ln_beta, 1e-5);
        }

        // 2) Batched fc1: [time, d_model] → [time, 4*d_model]
        let mut hidden = vec![0.0_f32; time * dff];
        linear_batch(&self.fc1_weight, &self.fc1_bias, &ln_buf, &mut hidden, time, dm, dff);
        swish_inplace(&mut hidden);

        // 3) Batched fc2: [time, 4*d_model] → [time, d_model]
        let mut ff_out = vec![0.0_f32; time * dm];
        linear_batch(&self.fc2_weight, &self.fc2_bias, &hidden, &mut ff_out, time, dff, dm);

        // 4) Residual with 0.5 macaron scale
        for i in 0..x.len() {
            x[i] += 0.5 * ff_out[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_ff(d_model: usize) -> FeedForward {
        let dff = 4 * d_model;
        // Layer norm: γ=1, β=0 (identity-like after normalization).
        let ln_gamma = vec![1.0; d_model];
        let ln_beta = vec![0.0; d_model];
        // fc1 zero → output zero → swish(0) = 0 → fc2 zero → 0
        let fc1_weight = vec![0.0; dff * d_model];
        let fc1_bias = vec![0.0; dff];
        let fc2_weight = vec![0.0; d_model * dff];
        let fc2_bias = vec![0.0; d_model];
        FeedForward {
            ln_gamma,
            ln_beta,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            d_model,
        }
    }

    #[test]
    fn zero_weights_keep_input_unchanged() {
        let ff = identity_ff(4);
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 frames × 4
        let original = x.clone();
        ff.forward_residual(&mut x, 2);
        // FF(x) = 0 → residual leaves x unchanged.
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "expected unchanged, {a} vs {b}");
        }
    }

    #[test]
    fn nonzero_bias_adds_constant_through_residual() {
        // Set fc2 bias only — output is then 0.5 * fc2_bias added to x.
        let mut ff = identity_ff(3);
        ff.fc2_bias = vec![2.0, 4.0, 6.0]; // 0.5 × bias = [1, 2, 3]
        let mut x = vec![10.0, 20.0, 30.0]; // 1 frame
        ff.forward_residual(&mut x, 1);
        assert!((x[0] - 11.0).abs() < 1e-5);
        assert!((x[1] - 22.0).abs() < 1e-5);
        assert!((x[2] - 33.0).abs() < 1e-5);
    }
}
