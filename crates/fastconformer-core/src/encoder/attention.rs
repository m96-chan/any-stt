//! Multi-head relative-position attention (Transformer-XL style).
//!
//! Used by parakeet-tdt-0.6b-v3 (regular rel-pos) and the non-Longformer
//! layers of any future Conformer. Longformer (local + global) is a
//! follow-up — see [`AttentionType::RelPosLocalAttn`] in `config.rs`.
//!
//! Forward (one block):
//!
//! ```text
//!   x_norm = LN(x)
//!   Q = x_norm · W_q  + b_q     → [T, H, D]
//!   K = x_norm · W_k             → [T, H, D]
//!   V = x_norm · W_v  + b_v     → [T, H, D]
//!
//!   P = pos_emb_table · W_pos   → [2T-1, H, D]   // relative positions
//!
//!   AC = (Q + u_bias) · Kᵀ       per head        → [T, T]
//!   BD = (Q + v_bias) · Pᵀ       per head        → [T, 2T-1]
//!   BD' = rel_shift(BD)                          → [T, T]
//!   score = (AC + BD') / sqrt(D)
//!   attn  = softmax(score)
//!   out_h = attn · V_h
//!
//!   out = concat(out_h) · W_o + b_o
//!   x ← x + out
//! ```

use crate::encoder::ops::{layer_norm_rows, softmax_inplace};

pub struct MultiHeadAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,

    // LN (γ, β each [d_model])
    pub ln_gamma: Vec<f32>,
    pub ln_beta: Vec<f32>,

    // Q / K / V / out projections: each [d_model, d_model]
    pub q_weight: Vec<f32>,
    pub q_bias: Vec<f32>,
    pub k_weight: Vec<f32>,
    pub v_weight: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub out_weight: Vec<f32>,
    pub out_bias: Vec<f32>,

    // Positional projection: [d_model, d_model]
    pub pos_weight: Vec<f32>,

    // Per-head learned position biases: each [n_heads, head_dim]
    pub pos_bias_u: Vec<f32>,
    pub pos_bias_v: Vec<f32>,
}

impl MultiHeadAttention {
    /// Apply attention residual: `x ← x + MHA(x)`.
    /// `x` is `[time, d_model]` row-major.
    pub fn forward_residual(&self, x: &mut [f32], time: usize) {
        let dm = self.d_model;
        let h = self.n_heads;
        let d = self.head_dim;
        debug_assert_eq!(x.len(), time * dm);
        debug_assert_eq!(d * h, dm);

        // 1) LN on a copy.
        let mut x_norm = x.to_vec();
        layer_norm_rows(&mut x_norm, time, dm, &self.ln_gamma, &self.ln_beta, 1e-5);

        // 2) Q, K, V projections.
        let mut q = vec![0.0_f32; time * dm];
        let mut k = vec![0.0_f32; time * dm];
        let mut v = vec![0.0_f32; time * dm];
        for t in 0..time {
            let src = &x_norm[t * dm..(t + 1) * dm];
            let qd = &mut q[t * dm..(t + 1) * dm];
            qd.copy_from_slice(&self.q_bias);
            mat_vec_add(&self.q_weight, src, qd, dm, dm);

            let kd = &mut k[t * dm..(t + 1) * dm];
            // K typically has no bias in NeMo; use zero if not present.
            mat_vec_add(&self.k_weight, src, kd, dm, dm);

            let vd = &mut v[t * dm..(t + 1) * dm];
            vd.copy_from_slice(&self.v_bias);
            mat_vec_add(&self.v_weight, src, vd, dm, dm);
        }

        // 3) Positional embeddings: build sinusoidal table of length 2T-1
        //    for offsets [-(T-1) .. T-1], project through W_pos.
        let pos_len = 2 * time - 1;
        let pe_table = sinusoidal_pos_emb(pos_len, dm);
        let mut pe_proj = vec![0.0_f32; pos_len * dm];
        for i in 0..pos_len {
            let src = &pe_table[i * dm..(i + 1) * dm];
            let dst = &mut pe_proj[i * dm..(i + 1) * dm];
            mat_vec_add(&self.pos_weight, src, dst, dm, dm);
        }

        // 4) Per-head computation.
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let mut output = vec![0.0_f32; time * dm];
        let mut score = vec![0.0_f32; time]; // reused row buffer

        for hd in 0..h {
            // Slice helpers — closure-style indexing.
            let q_h = |t: usize, c: usize| q[t * dm + hd * d + c];
            let k_h = |t: usize, c: usize| k[t * dm + hd * d + c];
            let v_h = |t: usize, c: usize| v[t * dm + hd * d + c];
            let p_h = |i: usize, c: usize| pe_proj[i * dm + hd * d + c];
            let u_h = |c: usize| self.pos_bias_u[hd * d + c];
            let v_bias_h = |c: usize| self.pos_bias_v[hd * d + c];

            // For each query frame t_q, compute scores against all t_k.
            for t_q in 0..time {
                // (Q + u) · Kᵀ : AC scores.
                // (Q + v) · Pᵀ : BD scores at offsets [-(T-1) .. T-1].
                // After rel_shift, BD' aligns offsets to time indices.
                // We build score[t_k] directly without an explicit BD buffer.

                // 1: AC[t_q, t_k] = Σ_c (Q[t_q,c] + u[c]) · K[t_k,c]
                for t_k in 0..time {
                    let mut s = 0.0_f32;
                    for c in 0..d {
                        s += (q_h(t_q, c) + u_h(c)) * k_h(t_k, c);
                    }
                    score[t_k] = s;
                }

                // 2: BD[t_q, off] = Σ_c (Q[t_q,c] + v[c]) · P[off, c]
                //    where off ∈ [0..2T-1] corresponds to relative
                //    position (off - (T-1)). The rel-shift trick maps
                //    (off, t_q) → t_k = t_q - (off - (T-1)) and we want
                //    score[t_k] += BD[t_q, off]. Equivalently, for each
                //    t_k we pick off = (T-1) + t_q - t_k.
                let mut bd_row = vec![0.0_f32; pos_len];
                for off in 0..pos_len {
                    let mut s = 0.0_f32;
                    for c in 0..d {
                        s += (q_h(t_q, c) + v_bias_h(c)) * p_h(off, c);
                    }
                    bd_row[off] = s;
                }
                for t_k in 0..time {
                    let off = (time as isize - 1) + t_q as isize - t_k as isize;
                    if (0..pos_len as isize).contains(&off) {
                        score[t_k] += bd_row[off as usize];
                    }
                }

                // 3: scale + softmax.
                for s in score.iter_mut() {
                    *s *= inv_sqrt_d;
                }
                softmax_inplace(&mut score);

                // 4: out_h[t_q] = Σ_t_k attn[t_k] · V[t_k]
                for c in 0..d {
                    let mut s = 0.0_f32;
                    for t_k in 0..time {
                        s += score[t_k] * v_h(t_k, c);
                    }
                    output[t_q * dm + hd * d + c] = s;
                }
            }
        }

        // 5) Output projection.
        let mut proj = vec![0.0_f32; time * dm];
        for t in 0..time {
            let src = &output[t * dm..(t + 1) * dm];
            let dst = &mut proj[t * dm..(t + 1) * dm];
            dst.copy_from_slice(&self.out_bias);
            mat_vec_add(&self.out_weight, src, dst, dm, dm);
        }

        // 6) Residual.
        for i in 0..time * dm {
            x[i] += proj[i];
        }
    }
}

/// Build a sinusoidal positional embedding table of shape `[length, dim]`
/// covering offsets centered on zero (length must be 2T-1; index 0
/// corresponds to offset `-(T-1)`, index `length-1` to `T-1`).
///
/// Standard Transformer-XL formula:
///   `pe[i, 2j]   = sin(off / 10000^(2j/dim))`
///   `pe[i, 2j+1] = cos(off / 10000^(2j/dim))`
/// where `off = i - (length / 2)`.
fn sinusoidal_pos_emb(length: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; length * dim];
    let half = (length / 2) as i32;
    for i in 0..length {
        let off = i as i32 - half;
        let off = off as f32;
        for j in 0..dim / 2 {
            let denom = 10000.0_f32.powf(2.0 * j as f32 / dim as f32);
            let arg = off / denom;
            out[i * dim + 2 * j] = arg.sin();
            if 2 * j + 1 < dim {
                out[i * dim + 2 * j + 1] = arg.cos();
            }
        }
    }
    out
}

/// `out += W · x` where W is `[rows, cols]` row-major.
fn mat_vec_add(w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut s = 0.0_f32;
        for c in 0..cols {
            s += row[c] * x[c];
        }
        out[r] += s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_mha(d_model: usize, n_heads: usize) -> MultiHeadAttention {
        let head_dim = d_model / n_heads;
        MultiHeadAttention {
            d_model,
            n_heads,
            head_dim,
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
            pos_bias_u: vec![0.0; n_heads * head_dim],
            pos_bias_v: vec![0.0; n_heads * head_dim],
        }
    }

    #[test]
    fn zero_weights_keep_input_unchanged() {
        let mha = zero_mha(4, 2);
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 × 4
        let orig = x.clone();
        mha.forward_residual(&mut x, 2);
        // V=0 → output projection of zero → 0. Residual: x += 0.
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn out_bias_adds_constant_through_residual() {
        let mut mha = zero_mha(4, 2);
        mha.out_bias = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = vec![10.0, 20.0, 30.0, 40.0];
        mha.forward_residual(&mut x, 1);
        assert!((x[0] - 11.0).abs() < 1e-4);
        assert!((x[1] - 22.0).abs() < 1e-4);
        assert!((x[2] - 33.0).abs() < 1e-4);
        assert!((x[3] - 44.0).abs() < 1e-4);
    }

    #[test]
    fn sinusoidal_pos_emb_at_offset_zero_is_special() {
        // At off=0: pe[..,2j]=sin(0)=0, pe[..,2j+1]=cos(0)=1.
        let pe = sinusoidal_pos_emb(5, 4); // half = 2
        // index 2 corresponds to off=0.
        assert!((pe[2 * 4 + 0] - 0.0).abs() < 1e-6);
        assert!((pe[2 * 4 + 1] - 1.0).abs() < 1e-6);
        assert!((pe[2 * 4 + 2] - 0.0).abs() < 1e-6);
        assert!((pe[2 * 4 + 3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sinusoidal_pos_emb_is_symmetric_in_cos() {
        // cos is even → pe[off, 2j+1] == pe[-off, 2j+1].
        let pe = sinusoidal_pos_emb(7, 4); // half = 3
        for j in 0..2 {
            assert!((pe[(3 - 1) * 4 + 2 * j + 1] - pe[(3 + 1) * 4 + 2 * j + 1]).abs() < 1e-6);
            assert!((pe[(3 - 2) * 4 + 2 * j + 1] - pe[(3 + 2) * 4 + 2 * j + 1]).abs() < 1e-6);
        }
    }
}
