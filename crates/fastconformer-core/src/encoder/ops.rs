//! Pure-Rust f32 primitives shared across encoder modules.
//!
//! These exist so the FastConformer forward pass can be assembled and
//! validated against Python reference outputs before any ggml / SIMD
//! optimization. Each function is straightforward, branch-free, and
//! documented with its tensor shape contract.

/// `out += W · x` where W is `[rows, cols]` row-major and x has length
/// `cols`. `out` has length `rows`.
pub fn matvec_add(w: &[f32], x: &[f32], out: &mut [f32]) {
    let rows = out.len();
    let cols = x.len();
    debug_assert_eq!(w.len(), rows * cols, "matvec_add: shape mismatch");
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut s = 0.0_f32;
        for c in 0..cols {
            s += row[c] * x[c];
        }
        out[r] += s;
    }
}

/// Apply a fully-connected layer: `y = W · x + b`.
/// `weight` is `[out_dim, in_dim]`, `bias` is `[out_dim]`.
pub fn linear(weight: &[f32], bias: &[f32], x: &[f32], out: &mut [f32]) {
    out.copy_from_slice(bias);
    matvec_add(weight, x, out);
}

/// LayerNorm across the last dimension.
/// `x` is `[features]`, modified in place. `gamma` and `beta` are also
/// `[features]`. `eps` is the standard 1e-5.
pub fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), gamma.len());
    debug_assert_eq!(x.len(), beta.len());
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v - mean) * inv_std * gamma[i] + beta[i];
    }
}

/// LayerNorm applied row-wise to a `[rows, cols]` matrix in place.
pub fn layer_norm_rows(
    x: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    debug_assert_eq!(x.len(), rows * cols);
    for r in 0..rows {
        layer_norm(&mut x[r * cols..(r + 1) * cols], gamma, beta, eps);
    }
}

/// Swish activation (a.k.a. SiLU): `x · sigmoid(x)`. NeMo Conformer uses
/// this in feed-forward modules.
pub fn swish_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= 1.0 / (1.0 + (-*v).exp());
    }
}

/// GLU (Gated Linear Unit): split `x` along the channel axis into two
/// halves `a, b`; output `a * sigmoid(b)`. Length of `out` is half of
/// the input.
pub fn glu_channels(x: &[f32], out: &mut [f32]) {
    let n = out.len();
    debug_assert_eq!(x.len(), 2 * n);
    let (a, b) = x.split_at(n);
    for i in 0..n {
        out[i] = a[i] * (1.0 / (1.0 + (-b[i]).exp()));
    }
}

/// Softmax across the last dimension of a vector. Replaces in place.
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// In-place batch-norm 1d using stored running statistics (inference
/// mode only — no update of moving averages). `x` is `[time, channels]`
/// row-major; `gamma`/`beta`/`running_mean`/`running_var` are
/// `[channels]`.
pub fn batch_norm_1d_inference(
    x: &mut [f32],
    time: usize,
    channels: usize,
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    eps: f32,
) {
    debug_assert_eq!(x.len(), time * channels);
    let mut inv_std = vec![0.0_f32; channels];
    for c in 0..channels {
        inv_std[c] = 1.0 / (running_var[c] + eps).sqrt();
    }
    for t in 0..time {
        let row = &mut x[t * channels..(t + 1) * channels];
        for c in 0..channels {
            row[c] = (row[c] - running_mean[c]) * inv_std[c] * gamma[c] + beta[c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_add_basic() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0, 1.0];
        let mut out = vec![0.0_f32; 2];
        matvec_add(&w, &x, &mut out);
        assert_eq!(out, vec![6.0, 15.0]);
    }

    #[test]
    fn linear_with_bias_adds_then_mat_muls() {
        let w = vec![2.0, 0.0, 0.0, 2.0]; // 2x2 identity * 2
        let b = vec![1.0, -1.0];
        let x = vec![3.0, 4.0];
        let mut out = vec![0.0; 2];
        linear(&w, &b, &x, &mut out);
        // out = 2 * [3,4] + [1,-1] = [7, 7]
        assert_eq!(out, vec![7.0, 7.0]);
    }

    #[test]
    fn layer_norm_centers_to_unit_var() {
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        layer_norm(&mut x, &gamma, &beta, 1e-5);
        // After LN, mean ≈ 0 and variance ≈ 1.
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn layer_norm_applies_gamma_beta() {
        let gamma = vec![2.0, 2.0];
        let beta = vec![1.0, -1.0];
        let mut x = vec![0.0, 0.0]; // mean 0, var 0 → (x-0)/eps_clamped = 0
        layer_norm(&mut x, &gamma, &beta, 1.0); // big eps to keep stable
        // result ≈ 0 * gamma + beta = beta
        assert!((x[0] - 1.0).abs() < 1e-3);
        assert!((x[1] + 1.0).abs() < 1e-3);
    }

    #[test]
    fn swish_zero_is_zero() {
        let mut x = vec![0.0_f32];
        swish_inplace(&mut x);
        assert!(x[0].abs() < 1e-6);
    }

    #[test]
    fn swish_large_positive_close_to_x() {
        let mut x = vec![10.0_f32];
        swish_inplace(&mut x);
        // sigmoid(10) ≈ 1, swish(10) ≈ 10
        assert!((x[0] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn swish_large_negative_close_to_zero() {
        let mut x = vec![-10.0_f32];
        swish_inplace(&mut x);
        // sigmoid(-10) ≈ 0, swish ≈ 0
        assert!(x[0].abs() < 1e-3);
    }

    #[test]
    fn glu_zero_b_halves_a() {
        // a = [1,2,3,4], b = [0,0,0,0] → sigmoid(0)=0.5 → [0.5, 1, 1.5, 2]
        let x = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let mut out = vec![0.0; 4];
        glu_channels(&x, &mut out);
        assert_eq!(out, vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        softmax_inplace(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        // Monotonically increasing input → monotonically increasing output.
        for w in x.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn softmax_handles_large_inputs_without_overflow() {
        let mut x = vec![1e6, 1e6 + 1.0, 1e6 + 2.0];
        softmax_inplace(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn batch_norm_1d_subtracts_running_mean() {
        // 2 frames × 3 channels, running_mean=[1,2,3], var=1, gamma=1, beta=0
        // Each frame is [1,2,3] → (x - mean) * 1 * 1 + 0 = 0.
        let mut x = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let rmean = vec![1.0, 2.0, 3.0];
        let rvar = vec![1.0; 3];
        batch_norm_1d_inference(&mut x, 2, 3, &gamma, &beta, &rmean, &rvar, 1e-5);
        for v in &x {
            assert!(v.abs() < 1e-3);
        }
    }
}
