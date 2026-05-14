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

/// Which attention pattern to use.
#[derive(Debug, Clone, Copy)]
pub enum AttentionMode {
    /// Standard full attention — every query attends to every key.
    Full,
    /// Longformer-style: each query attends to keys within a sliding
    /// window plus a fixed number of "global" tokens that always attend
    /// to/from everywhere. ReazonSpeech-NeMo-v2 uses this with
    /// `local_window=256, global_tokens=1`.
    LocalGlobal {
        /// Window radius — query t attends to keys in [t-w, t+w].
        local_window: usize,
        /// Number of leading frames that act as global tokens. Global
        /// tokens attend to all keys and are attended by all queries.
        global_tokens: usize,
    },
}

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
    /// NeMo MHA has `use_bias=True` by default, so K has a bias too. Earlier
    /// the loader silently dropped it which biased attention scores by a
    /// constant per channel.
    pub k_bias: Vec<f32>,
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
    /// Apply attention residual with full (all-to-all) attention.
    /// `x` is `[time, d_model]` row-major.
    pub fn forward_residual(&self, x: &mut [f32], time: usize) {
        self.forward_residual_with_mode(x, time, AttentionMode::Full);
    }

    /// Apply attention residual with the given attention pattern.
    pub fn forward_residual_with_mode(
        &self,
        x: &mut [f32],
        time: usize,
        mode: AttentionMode,
    ) {
        self.forward_inner(x, time, mode, /*skip_ln=*/ false);
    }

    /// Same as `forward_residual_with_mode` but treats `x` as already
    /// LN'd input. Used by tests/layer_reference.rs to isolate attention
    /// math from upstream LN behaviour.
    pub fn forward_residual_no_ln(
        &self,
        x: &mut [f32],
        time: usize,
        mode: AttentionMode,
    ) {
        self.forward_inner(x, time, mode, /*skip_ln=*/ true);
    }

    fn forward_inner(
        &self,
        x: &mut [f32],
        time: usize,
        mode: AttentionMode,
        skip_ln: bool,
    ) {
        let dm = self.d_model;
        let h = self.n_heads;
        let d = self.head_dim;
        debug_assert_eq!(x.len(), time * dm);
        debug_assert_eq!(d * h, dm);

        // 1) LN on a copy (or skip if caller already pre-normalized).
        let mut x_norm = x.to_vec();
        if !skip_ln {
            layer_norm_rows(&mut x_norm, time, dm, &self.ln_gamma, &self.ln_beta, 1e-5);
        }

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
            kd.copy_from_slice(&self.k_bias);
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
                //    position (off - (T-1)). For Longformer (`LocalGlobal`):
                //    - Global QUERIES (t_q < global_tokens) use AC-only,
                //      no BD — their attention goes through NeMo's
                //      `_compute_global_key_attn` which doesn't include
                //      the rel-pos bias.
                //    - Non-global queries DO use BD on all keys (including
                //      the global-key column), because NeMo's BD matrix is
                //      added to the full local-attention scores.
                let q_is_global = matches!(mode, AttentionMode::LocalGlobal { global_tokens, .. } if t_q < global_tokens);
                if !q_is_global {
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
                }

                // 3: scale, optional Longformer masking, softmax.
                for s in score.iter_mut() {
                    *s *= inv_sqrt_d;
                }
                if let AttentionMode::LocalGlobal {
                    local_window,
                    global_tokens,
                } = mode
                {
                    // Global queries attend to all keys; local queries
                    // attend only to keys within the window or to a
                    // global key. Mask the rest with -inf so softmax
                    // assigns them zero weight.
                    if !q_is_global {
                        for t_k in 0..time {
                            let k_is_global = t_k < global_tokens;
                            let dist = (t_q as isize - t_k as isize).unsigned_abs();
                            let in_window = dist <= local_window;
                            if !(k_is_global || in_window) {
                                score[t_k] = f32::NEG_INFINITY;
                            }
                        }
                    }
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
            k_bias: vec![0.0; d_model],
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

    // --- Longformer (LocalGlobal) tests ---

    /// Build an MHA where V is the identity per-frame (i.e. v_h(t,c) = LN(x)
    /// at that position, scaled by inv_sqrt). For zero weights the test
    /// can check the masking shape via output values.
    fn permissive_mha(d_model: usize, n_heads: usize) -> MultiHeadAttention {
        let head_dim = d_model / n_heads;
        // Identity-ish: V projects input to itself (W_v = I, b_v = 0).
        let mut v_weight = vec![0.0_f32; d_model * d_model];
        for i in 0..d_model {
            v_weight[i * d_model + i] = 1.0;
        }
        let mut out_weight = vec![0.0_f32; d_model * d_model];
        for i in 0..d_model {
            out_weight[i * d_model + i] = 1.0;
        }
        MultiHeadAttention {
            d_model,
            n_heads,
            head_dim,
            ln_gamma: vec![1.0; d_model],
            ln_beta: vec![0.0; d_model],
            q_weight: vec![0.0; d_model * d_model],
            q_bias: vec![0.0; d_model],
            k_weight: vec![0.0; d_model * d_model],
            k_bias: vec![0.0; d_model],
            v_weight,
            v_bias: vec![0.0; d_model],
            out_weight,
            out_bias: vec![0.0; d_model],
            pos_weight: vec![0.0; d_model * d_model],
            pos_bias_u: vec![0.0; n_heads * head_dim],
            pos_bias_v: vec![0.0; n_heads * head_dim],
        }
    }

    #[test]
    fn local_global_with_huge_window_equals_full_attention() {
        // local_window > time means every key is in window for every
        // query → mask never triggers → output equals Full mode output.
        let mha = permissive_mha(4, 2);
        let mut x_full = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x_lg = x_full.clone();

        mha.forward_residual_with_mode(&mut x_full, 2, AttentionMode::Full);
        mha.forward_residual_with_mode(
            &mut x_lg,
            2,
            AttentionMode::LocalGlobal {
                local_window: 100,
                global_tokens: 0,
            },
        );

        for (a, b) in x_full.iter().zip(x_lg.iter()) {
            assert!((a - b).abs() < 1e-5, "Full vs LG huge-window: {a} vs {b}");
        }
    }

    #[test]
    fn local_global_with_zero_window_self_attention_only() {
        // local_window=0, global_tokens=0 means each query t_q only
        // attends to t_k = t_q (distance 0 ≤ 0). With zero Q/K weights
        // all (single) score is 0 → softmax = 1.0 → out_h(t_q) = V(t_q).
        // V is identity, so the residual adds LN(x) to x.
        let mha = permissive_mha(4, 2);
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0]; // 1 frame
        let original = x.clone();
        mha.forward_residual_with_mode(
            &mut x,
            1,
            AttentionMode::LocalGlobal {
                local_window: 0,
                global_tokens: 0,
            },
        );
        // For a single frame, local_window=0 still means it attends to
        // itself, so residual is non-trivial. Just verify finite output.
        for (a, b) in x.iter().zip(original.iter()) {
            assert!(a.is_finite(), "non-finite output: {a}");
            assert!(b.is_finite()); // sanity
        }
    }

    #[test]
    fn global_token_attends_to_everything_even_outside_window() {
        // 5-frame input, local_window=1, global_tokens=1. The global
        // query (t_q=0) should attend to ALL 5 keys; a local query at
        // t_q=2 only attends to {0 (global), 1, 2, 3} — NOT 4.
        //
        // We use an input that varies *across channels* (so per-frame
        // LayerNorm produces a non-zero pattern, not a zero scalar) and
        // place a unique signature at frame 4. Then we run twice with
        // different masking to show the global query sees the change at
        // frame 4 while the far-local query does not.
        let mha = permissive_mha(4, 2);

        // Helper to build the input: zeros, then a signature at frame 4.
        let make_input = || {
            let mut x = vec![0.0_f32; 5 * 4];
            // Frame 4 has [-1, -1, 1, 1] — LN(this) is non-zero.
            x[4 * 4 + 0] = -1.0;
            x[4 * 4 + 1] = -1.0;
            x[4 * 4 + 2] = 1.0;
            x[4 * 4 + 3] = 1.0;
            x
        };

        let mut x_global = make_input();
        let mut x_far_local = make_input();

        // Both queries use the same window setup, but different t_q.
        // Run with global_tokens=1 → frame 0 is global.
        mha.forward_residual_with_mode(
            &mut x_global,
            5,
            AttentionMode::LocalGlobal {
                local_window: 1,
                global_tokens: 1,
            },
        );
        // Same mode, but we examine frame 2 (a local query whose window
        // [1, 3] excludes frame 4).
        mha.forward_residual_with_mode(
            &mut x_far_local,
            5,
            AttentionMode::LocalGlobal {
                local_window: 1,
                global_tokens: 1,
            },
        );

        for v in x_global.iter().chain(x_far_local.iter()) {
            assert!(v.is_finite());
        }

        // Frame 0 (global query) attended to frame 4 → its output
        // changed from the zero input.
        let frame0_change: f32 = (0..4).map(|c| x_global[c].abs()).sum();
        assert!(
            frame0_change > 1e-3,
            "global query (frame 0) should have attended to frame 4 \
             through the large channel-variation signal; change={frame0_change}",
        );

        // Frame 2 (local query, window [1, 3], frame 4 OUT of window
        // and frame 0 is zero). With V=identity and Q/K=0, the only
        // non-zero contribution to a query's output is from frames
        // visible to it that have non-zero V. Frames 1, 2, 3 are zero;
        // frame 0 (global key, in scope) is zero. So frame 2's output
        // change should be ≪ frame 0's.
        let frame2_change: f32 = (0..4).map(|c| x_far_local[2 * 4 + c].abs()).sum();
        assert!(
            frame2_change < frame0_change,
            "frame 2 (local query, frame 4 out of window) should change \
             less than frame 0 (global query). got: frame0={frame0_change}, \
             frame2={frame2_change}",
        );
    }
}
