//! Optimized f32 primitives for FastConformer encoder.
//!
//! NEON SIMD on aarch64, rayon parallelism for large matmul.
//! Falls back to scalar on other architectures.

use rayon::prelude::*;

/// Dot product of two f32 slices using NEON SIMD on aarch64.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    debug_assert_eq!(n, b.len());

    let chunks = n / 16;
    let remainder = n % 16;

    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let ap = a.as_ptr();
        let bp = b.as_ptr();

        for i in 0..chunks {
            let off = i * 16;
            let a0 = vld1q_f32(ap.add(off));
            let a1 = vld1q_f32(ap.add(off + 4));
            let a2 = vld1q_f32(ap.add(off + 8));
            let a3 = vld1q_f32(ap.add(off + 12));
            let b0 = vld1q_f32(bp.add(off));
            let b1 = vld1q_f32(bp.add(off + 4));
            let b2 = vld1q_f32(bp.add(off + 8));
            let b3 = vld1q_f32(bp.add(off + 12));
            sum0 = vfmaq_f32(sum0, a0, b0);
            sum1 = vfmaq_f32(sum1, a1, b1);
            sum2 = vfmaq_f32(sum2, a2, b2);
            sum3 = vfmaq_f32(sum3, a3, b3);
        }

        let combined = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        let mut s = vaddvq_f32(combined);

        // Handle remainder
        let tail_start = chunks * 16;
        for i in 0..remainder {
            s += a[tail_start + i] * b[tail_start + i];
        }
        s
    }
}

/// Scalar fallback for non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

/// `out += W · x` where W is `[rows, cols]` row-major and x has length
/// `cols`. `out` has length `rows`.
///
/// Uses rayon for rows > 128, NEON SIMD for dot products.
pub fn matvec_add(w: &[f32], x: &[f32], out: &mut [f32]) {
    let rows = out.len();
    let cols = x.len();
    debug_assert_eq!(w.len(), rows * cols, "matvec_add: shape mismatch");

    if rows > 128 {
        // Parallel: split output rows across threads
        let results: Vec<f32> = (0..rows)
            .into_par_iter()
            .map(|r| {
                let row = &w[r * cols..(r + 1) * cols];
                dot_f32(row, x)
            })
            .collect();
        for (o, r) in out.iter_mut().zip(results.iter()) {
            *o += r;
        }
    } else {
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            out[r] += dot_f32(row, x);
        }
    }
}

/// Apply a fully-connected layer: `y = W · x + b`.
/// `weight` is `[out_dim, in_dim]`, `bias` is `[out_dim]`.
pub fn linear(weight: &[f32], bias: &[f32], x: &[f32], out: &mut [f32]) {
    out.copy_from_slice(bias);
    matvec_add(weight, x, out);
}

/// LayerNorm across the last dimension.
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

/// Swish activation (a.k.a. SiLU): `x · sigmoid(x)`.
pub fn swish_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= 1.0 / (1.0 + (-*v).exp());
    }
}

/// GLU (Gated Linear Unit): split `x` along the channel axis into two
/// halves `a, b`; output `a * sigmoid(b)`.
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

/// In-place batch-norm 1d using stored running statistics (inference mode).
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
        let w = vec![2.0, 0.0, 0.0, 2.0];
        let b = vec![1.0, -1.0];
        let x = vec![3.0, 4.0];
        let mut out = vec![0.0; 2];
        linear(&w, &b, &x, &mut out);
        assert_eq!(out, vec![7.0, 7.0]);
    }

    #[test]
    fn layer_norm_centers_to_unit_var() {
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        layer_norm(&mut x, &gamma, &beta, 1e-5);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn dot_f32_matches_scalar() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..1024).map(|i| (1024 - i) as f32 * 0.01).collect();
        let simd_result = dot_f32(&a, &b);
        let scalar_result: f32 = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
        assert!((simd_result - scalar_result).abs() < 0.1,
            "simd={simd_result} scalar={scalar_result}");
    }

    #[test]
    fn matvec_add_large_parallel() {
        let rows = 256;
        let cols = 512;
        let w: Vec<f32> = (0..rows * cols).map(|i| (i % 7) as f32 * 0.001).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i % 5) as f32 * 0.01).collect();
        let mut out_par = vec![0.0_f32; rows];
        let mut out_seq = vec![0.0_f32; rows];
        matvec_add(&w, &x, &mut out_par);
        // Sequential reference
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            for c in 0..cols {
                out_seq[r] += row[c] * x[c];
            }
        }
        for (p, s) in out_par.iter().zip(out_seq.iter()) {
            assert!((p - s).abs() < 0.01, "par={p} seq={s}");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        softmax_inplace(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn batch_norm_1d_subtracts_running_mean() {
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

/// Batched linear: `out = X · W^T + bias` where X is `[rows, in_dim]`,
/// W is `[out_dim, in_dim]`, bias is `[out_dim]`, out is `[rows, out_dim]`.
/// Much faster than per-row matvec because of cache locality and SIMD throughput.
pub fn linear_batch(
    weight: &[f32],   // [out_dim, in_dim] row-major
    bias: &[f32],     // [out_dim]
    x: &[f32],        // [rows, in_dim] row-major
    out: &mut [f32],   // [rows, out_dim] row-major
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(bias.len(), out_dim);
    debug_assert_eq!(x.len(), rows * in_dim);
    debug_assert_eq!(out.len(), rows * out_dim);

    // Initialize output with bias
    for r in 0..rows {
        out[r * out_dim..(r + 1) * out_dim].copy_from_slice(bias);
    }

    // GEMM: out += X · W^T
    // For each output column (weight row), compute dot products with all input rows
    // Row-parallel GEMM: each row independently computes dot products with all weight rows
    use rayon::prelude::*;
    let row_results: Vec<Vec<f32>> = (0..rows)
        .into_par_iter()
        .map(|r| {
            let x_row = &x[r * in_dim..(r + 1) * in_dim];
            let mut result = vec![0.0_f32; out_dim];
            for j in 0..out_dim {
                let w_row = &weight[j * in_dim..(j + 1) * in_dim];
                result[j] = dot_f32(x_row, w_row);
            }
            result
        })
        .collect();
    for (r, row) in row_results.iter().enumerate() {
        for j in 0..out_dim {
            out[r * out_dim + j] += row[j];
        }
    }
}
