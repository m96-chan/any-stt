//! Conformer convolution module.
//!
//! ```text
//! x [T, d_model]
//!   ↓ LayerNorm
//!   ↓ Pointwise Conv1d (d_model → 2*d_model)
//!   ↓ GLU (split last dim)
//!   ↓ Depthwise Conv1d (kernel=conv_kernel_size, pad symmetric)
//!   ↓ BatchNorm 1d (running stats)
//!   ↓ Swish
//!   ↓ Pointwise Conv1d (d_model → d_model)
//!   = output [T, d_model]
//! ```
//!
//! Pointwise Conv1d = batched linear per frame. Depthwise Conv1d means
//! every output channel only mixes its own input channel along time.

use crate::encoder::ops::{
    batch_norm_1d_inference, glu_channels, layer_norm, linear, swish_inplace,
};

pub struct ConvModule {
    pub d_model: usize,
    pub conv_kernel_size: usize,

    // LN (γ, β each [d_model])
    pub ln_gamma: Vec<f32>,
    pub ln_beta: Vec<f32>,

    // Pointwise 1: [2*d_model, d_model]
    pub pw1_weight: Vec<f32>,
    pub pw1_bias: Vec<f32>,

    // Depthwise: per-channel 1D kernel of length `conv_kernel_size`.
    /// `[d_model, conv_kernel_size]` row-major (channel-major).
    pub dw_weight: Vec<f32>,
    pub dw_bias: Vec<f32>, // `[d_model]`

    // BatchNorm running stats (γ, β, running_mean, running_var each [d_model])
    pub bn_gamma: Vec<f32>,
    pub bn_beta: Vec<f32>,
    pub bn_running_mean: Vec<f32>,
    pub bn_running_var: Vec<f32>,

    // Pointwise 2: [d_model, d_model]
    pub pw2_weight: Vec<f32>,
    pub pw2_bias: Vec<f32>,
}

impl ConvModule {
    /// Apply conv-module residual: `x ← x + ConvModule(x)`.
    /// `x` is `[time, d_model]` row-major.
    pub fn forward_residual(&self, x: &mut [f32], time: usize) {
        let dm = self.d_model;
        let dff = 2 * dm;
        debug_assert_eq!(x.len(), time * dm);

        // Side buffers.
        let mut ln_buf = vec![0.0_f32; dm];
        let mut pw1_out = vec![0.0_f32; dff];
        let mut glu_out = vec![0.0_f32; time * dm]; // we need full time series before depthwise
        let mut after_dw = vec![0.0_f32; time * dm];
        let mut after_pw2 = vec![0.0_f32; time * dm];

        // 1) LN + PW1 + GLU per frame → glu_out [time, dm]
        for t in 0..time {
            ln_buf.copy_from_slice(&x[t * dm..(t + 1) * dm]);
            layer_norm(&mut ln_buf, &self.ln_gamma, &self.ln_beta, 1e-5);
            linear(&self.pw1_weight, &self.pw1_bias, &ln_buf, &mut pw1_out);
            glu_channels(&pw1_out, &mut glu_out[t * dm..(t + 1) * dm]);
        }

        // 2) Depthwise Conv1d along time for each channel. Symmetric
        //    padding so output length = time.
        depthwise_conv1d(
            &glu_out,
            time,
            dm,
            self.conv_kernel_size,
            &self.dw_weight,
            &self.dw_bias,
            &mut after_dw,
        );

        // 3) BN + Swish (in place on after_dw).
        batch_norm_1d_inference(
            &mut after_dw,
            time,
            dm,
            &self.bn_gamma,
            &self.bn_beta,
            &self.bn_running_mean,
            &self.bn_running_var,
            1e-5,
        );
        swish_inplace(&mut after_dw);

        // 4) Pointwise 2 per frame.
        for t in 0..time {
            let src = &after_dw[t * dm..(t + 1) * dm];
            linear(
                &self.pw2_weight,
                &self.pw2_bias,
                src,
                &mut after_pw2[t * dm..(t + 1) * dm],
            );
        }

        // 5) Residual.
        for i in 0..time * dm {
            x[i] += after_pw2[i];
        }
    }
}

/// Depthwise 1D convolution with symmetric "same" padding.
/// `x` is `[time, channels]` row-major. `weight` is `[channels, kernel]`.
/// Output is `[time, channels]` row-major. `bias` is `[channels]`.
fn depthwise_conv1d(
    x: &[f32],
    time: usize,
    channels: usize,
    kernel: usize,
    weight: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), time * channels);
    debug_assert_eq!(weight.len(), channels * kernel);
    debug_assert_eq!(bias.len(), channels);
    debug_assert_eq!(out.len(), time * channels);

    let pad = kernel / 2;
    for t in 0..time {
        for c in 0..channels {
            let kw = &weight[c * kernel..(c + 1) * kernel];
            let mut s = bias[c];
            for k in 0..kernel {
                let src_t = t as isize + k as isize - pad as isize;
                if src_t < 0 || src_t >= time as isize {
                    continue;
                }
                s += x[src_t as usize * channels + c] * kw[k];
            }
            out[t * channels + c] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_conv_module(d_model: usize, kernel: usize) -> ConvModule {
        ConvModule {
            d_model,
            conv_kernel_size: kernel,
            ln_gamma: vec![1.0; d_model],
            ln_beta: vec![0.0; d_model],
            pw1_weight: vec![0.0; 2 * d_model * d_model],
            pw1_bias: vec![0.0; 2 * d_model],
            dw_weight: vec![0.0; d_model * kernel],
            dw_bias: vec![0.0; d_model],
            bn_gamma: vec![1.0; d_model],
            bn_beta: vec![0.0; d_model],
            bn_running_mean: vec![0.0; d_model],
            bn_running_var: vec![1.0; d_model],
            pw2_weight: vec![0.0; d_model * d_model],
            pw2_bias: vec![0.0; d_model],
        }
    }

    #[test]
    fn zero_weights_keep_input_unchanged() {
        let cm = zero_conv_module(4, 3);
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 frames × 4
        let orig = x.clone();
        cm.forward_residual(&mut x, 2);
        // PW1=0 → GLU(0) = 0 → DW = bias = 0 → BN(0) = -mean/std + β = 0
        // Wait: BN of [0..0] with running_mean=0, var=1 → (0-0)*1 + 0 = 0
        // Swish(0)=0, PW2(0) + bias(0) = 0. Residual: x += 0.
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn pw2_bias_adds_constant() {
        let mut cm = zero_conv_module(3, 3);
        cm.pw2_bias = vec![1.0, 2.0, 3.0];
        // After zero PW1 → 0 → GLU(0) = 0 → DW(0)+0_bias = 0 → BN(0)=0 →
        // Swish(0)=0 → PW2(0)+bias = bias. So residual adds bias.
        let mut x = vec![10.0, 20.0, 30.0];
        cm.forward_residual(&mut x, 1);
        assert!((x[0] - 11.0).abs() < 1e-5);
        assert!((x[1] - 22.0).abs() < 1e-5);
        assert!((x[2] - 33.0).abs() < 1e-5);
    }

    #[test]
    fn depthwise_conv1d_identity_kernel_passes_through() {
        // 1 channel, kernel=3, weights=[0,1,0] → output[t] = input[t].
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![0.0, 1.0, 0.0];
        let bias = vec![0.0];
        let mut out = vec![0.0; 4];
        depthwise_conv1d(&x, 4, 1, 3, &weight, &bias, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn depthwise_conv1d_pads_symmetrically() {
        // 1 channel, kernel=3, weights=[1,0,0] → output[t] = input[t-1]
        // (left padding = zero for t=0).
        let x = vec![1.0_f32, 2.0, 3.0];
        let weight = vec![1.0, 0.0, 0.0];
        let bias = vec![0.0];
        let mut out = vec![0.0; 3];
        depthwise_conv1d(&x, 3, 1, 3, &weight, &bias, &mut out);
        assert_eq!(out, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn depthwise_conv1d_two_channels_independent() {
        // 2 channels, each with identity kernel.
        let x = vec![1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0]; // [t, c]
        let weight = vec![0.0, 1.0, 0.0,   0.0, 1.0, 0.0]; // 2 channels × 3
        let bias = vec![0.0, 0.0];
        let mut out = vec![0.0; 6];
        depthwise_conv1d(&x, 3, 2, 3, &weight, &bias, &mut out);
        assert_eq!(out, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    }
}
