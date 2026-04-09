//! NNAPI-based Whisper encoder graph builder.
//!
//! Splits each encoder layer into NPU-accelerated subgraphs with CPU attention:
//!   Graph A (NPU): LayerNorm → Q/K/V FC projections
//!   CPU:           Q@K^T → scale → softmax → attn@V
//!   Graph B (NPU): output FC → residual → LayerNorm → FC1 → GELU → FC2 → residual
//!
//! Uses FULLY_CONNECTED instead of BATCH_MATMUL (not supported on MediaTek NPU).
//! GELU is approximated as x * sigmoid(1.702 * x).

use crate::context::{NnapiCompiled, NnapiModelBuilder};
use crate::loader::NnapiLib;
use crate::types::*;

/// Compiled Whisper encoder ready for execution on NNAPI NPU + CPU attention.
pub struct WhisperEncoderNnapi {
    layers: Vec<EncoderLayer>,
    n_ctx: u32,
    n_state: u32,
    n_head: u32,
}

struct EncoderLayer {
    /// Graph A: input [n_ctx, n_state] → Q [n_ctx, n_state], K [...], V [...]
    qkv_graph: NnapiCompiled,
    /// Graph B: attn_out + residual_input → layer output [n_ctx, n_state]
    ffn_graph: NnapiCompiled,
}

unsafe impl Send for WhisperEncoderNnapi {}
unsafe impl Sync for WhisperEncoderNnapi {}

impl WhisperEncoderNnapi {
    /// Build and compile the encoder targeting a specific NNAPI device.
    pub fn build<F>(
        lib: &NnapiLib,
        device: *const ANeuralNetworksDevice,
        n_state: u32,
        n_head: u32,
        n_layer: u32,
        n_ctx: u32,
        mut weights: F,
    ) -> Result<Self, String>
    where
        F: FnMut(&str) -> Result<Vec<f32>, String>,
    {
        // Generate a cache token from model dimensions for NNAPI compilation caching
        let cache_dir = std::env::var("NNAPI_CACHE_DIR").ok();
        let mut layers = Vec::new();

        for layer_idx in 0..n_layer {
            eprintln!("NNAPI: building layer {layer_idx}/{n_layer}...");
            let is_last = layer_idx == n_layer - 1;

            // Per-layer cache token: hash of (n_state, n_head, n_ctx, layer_idx, is_last)
            let cache_token = cache_dir.as_ref().map(|_| {
                let mut token = [0u8; 32];
                let key = format!("whisper_enc_{n_state}_{n_head}_{n_ctx}_{layer_idx}_{is_last}");
                // Simple hash: copy key bytes cyclically
                for (i, b) in key.bytes().enumerate() {
                    token[i % 32] ^= b;
                }
                token
            });

            let layer = build_layer(
                lib, device, n_state, n_head, n_ctx, layer_idx, is_last, &mut weights,
                cache_dir.as_deref(), cache_token.as_ref(),
            )?;
            layers.push(layer);
        }

        Ok(Self { layers, n_ctx, n_state, n_head })
    }

    /// Execute: input [n_ctx * n_state] → output [n_ctx * n_state].
    pub fn execute(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let nc = self.n_ctx as usize;
        let ns = self.n_state as usize;
        let nh = self.n_head as usize;
        let head_dim = ns / nh;
        let expected = nc * ns;
        if input.len() != expected {
            return Err(format!("input length {}, expected {expected}", input.len()));
        }

        let out_bytes = expected * 4;
        let mut current = input.to_vec();

        for (i, layer) in self.layers.iter().enumerate() {
            // --- Graph A: LayerNorm + QKV projections (NPU) ---
            // Input: current [n_ctx, n_state]
            // Outputs: Q [n_ctx, n_state], K [...], V [...]
            let qkv_out_bytes = nc * ns * 4;
            let qkv_results = layer.qkv_graph.execute(
                &[(0, &current)],
                &[(0, qkv_out_bytes), (1, qkv_out_bytes), (2, qkv_out_bytes)],
            ).map_err(|e| format!("layer {i} qkv: {e}"))?;

            let q = &qkv_results[0];
            let k = &qkv_results[1];
            let v = &qkv_results[2];

            // --- CPU: Attention ---
            // QK^T → scale → softmax → @V
            let scale = 1.0 / (head_dim as f32).sqrt();
            let attn_out = cpu_attention(q, k, v, nc, ns, scale);

            // --- Graph B: output proj + residual + LN + FFN + residual (NPU) ---
            // Inputs: attn_out [n_ctx, n_state], residual (current) [n_ctx, n_state]
            let ffn_results = layer.ffn_graph.execute(
                &[(0, &attn_out), (1, &current)],
                &[(0, out_bytes)],
            ).map_err(|e| format!("layer {i} ffn: {e}"))?;

            current = ffn_results.into_iter().next()
                .ok_or_else(|| format!("layer {i}: no output"))?;
        }

        Ok(current)
    }
}

/// CPU attention: Q @ K^T * scale → softmax → @ V
///
/// Multi-threaded: parallelizes the outer loop of QK^T and attn@V matmuls
/// across available CPU cores. V is transposed before the attn@V matmul for
/// cache-friendly row access.
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], nc: usize, ns: usize, scale: f32) -> Vec<f32> {
    use std::sync::Arc;
    use std::thread;

    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // --- QK^T: [nc, ns] @ [ns, nc] → [nc, nc], scaled ---
    // K is stored row-major [nc, ns], so k[j] row = k[j*ns .. (j+1)*ns].
    // The inner product q_row[d] * k_row[d] is already cache-friendly for both.
    let mut qk = vec![0.0f32; nc * nc];
    {
        let q_arc = Arc::new(q.to_vec());
        let k_arc = Arc::new(k.to_vec());

        let rows_per_thread = (nc + n_threads - 1) / n_threads;
        let mut handles = Vec::with_capacity(n_threads);

        for t in 0..n_threads {
            let row_start = t * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(nc);
            if row_start >= nc {
                break;
            }
            let q_ref = Arc::clone(&q_arc);
            let k_ref = Arc::clone(&k_arc);
            let nc_ = nc;
            let ns_ = ns;
            let scale_ = scale;

            handles.push(thread::spawn(move || {
                let chunk_len = (row_end - row_start) * nc_;
                let mut chunk = vec![0.0f32; chunk_len];
                for i in row_start..row_end {
                    let q_row = &q_ref[i * ns_..(i + 1) * ns_];
                    for j in 0..nc_ {
                        let k_row = &k_ref[j * ns_..(j + 1) * ns_];
                        let mut sum = 0.0f32;
                        for d in 0..ns_ {
                            sum += q_row[d] * k_row[d];
                        }
                        chunk[(i - row_start) * nc_ + j] = sum * scale_;
                    }
                }
                (row_start, chunk)
            }));
        }

        for h in handles {
            let (row_start, chunk) = h.join().expect("QK^T thread panicked");
            let offset = row_start * nc;
            qk[offset..offset + chunk.len()].copy_from_slice(&chunk);
        }
    }

    // --- Softmax over last dim (already fast, no parallelism needed) ---
    for i in 0..nc {
        let row = &mut qk[i * nc..(i + 1) * nc];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in row.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        for x in row.iter_mut() {
            *x /= sum;
        }
    }

    // --- attn @ V: [nc, nc] @ [nc, ns] → [nc, ns] ---
    // Transpose V first: V is [nc, ns] row-major → V^T is [ns, nc] row-major.
    // This lets us access V^T row j (= V column j) contiguously when computing
    // out[i, j] = sum_t qk[i, t] * V^T[j, t].
    let mut vt = vec![0.0f32; ns * nc];
    for r in 0..nc {
        for c in 0..ns {
            vt[c * nc + r] = v[r * ns + c];
        }
    }

    let mut out = vec![0.0f32; nc * ns];
    {
        let qk_arc = Arc::new(qk);
        let vt_arc = Arc::new(vt);

        let rows_per_thread = (nc + n_threads - 1) / n_threads;
        let mut handles = Vec::with_capacity(n_threads);

        for t in 0..n_threads {
            let row_start = t * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(nc);
            if row_start >= nc {
                break;
            }
            let qk_ref = Arc::clone(&qk_arc);
            let vt_ref = Arc::clone(&vt_arc);
            let nc_ = nc;
            let ns_ = ns;

            handles.push(thread::spawn(move || {
                let chunk_len = (row_end - row_start) * ns_;
                let mut chunk = vec![0.0f32; chunk_len];
                for i in row_start..row_end {
                    let qk_row = &qk_ref[i * nc_..(i + 1) * nc_];
                    for j in 0..ns_ {
                        let vt_row = &vt_ref[j * nc_..(j + 1) * nc_];
                        let mut sum = 0.0f32;
                        for t in 0..nc_ {
                            sum += qk_row[t] * vt_row[t];
                        }
                        chunk[(i - row_start) * ns_ + j] = sum;
                    }
                }
                (row_start, chunk)
            }));
        }

        for h in handles {
            let (row_start, chunk) = h.join().expect("attn@V thread panicked");
            let offset = row_start * ns;
            out[offset..offset + chunk.len()].copy_from_slice(&chunk);
        }
    }

    out
}

// ======================================================================
// Graph builders
// ======================================================================

fn build_layer<F>(
    lib: &NnapiLib,
    device: *const ANeuralNetworksDevice,
    n_state: u32,
    _n_head: u32,
    n_ctx: u32,
    layer_idx: u32,
    is_last: bool,
    weights: &mut F,
    cache_dir: Option<&str>,
    cache_token: Option<&[u8; 32]>,
) -> Result<EncoderLayer, String>
where
    F: FnMut(&str) -> Result<Vec<f32>, String>,
{
    let prefix = format!("encoder.blocks.{layer_idx}");

    // QKV cache token variant
    let qkv_token = cache_token.map(|t| {
        let mut qt = *t;
        qt[31] ^= 0xAA; // differentiate from FFN
        qt
    });

    let qkv_graph = build_qkv_graph(lib, device, n_state, n_ctx, &prefix, weights,
        cache_dir, qkv_token.as_ref())?;

    let ffn_graph = build_ffn_graph(lib, device, n_state, n_ctx, &prefix, is_last, weights,
        cache_dir, cache_token)?;

    Ok(EncoderLayer { qkv_graph, ffn_graph })
}

/// Graph A: input [n_ctx, n_state] → LayerNorm → Q, K, V via FC
/// Model inputs: [input]
/// Model outputs: [Q, K, V]
fn build_qkv_graph<F>(
    lib: &NnapiLib,
    device: *const ANeuralNetworksDevice,
    n_state: u32,
    n_ctx: u32,
    prefix: &str,
    weights: &mut F,
    cache_dir: Option<&str>,
    cache_token: Option<&[u8; 32]>,
) -> Result<NnapiCompiled, String>
where
    F: FnMut(&str) -> Result<Vec<f32>, String>,
{
    let mut b = NnapiModelBuilder::new(lib)?;

    // Input: [n_ctx, n_state] (flattened from [1, n_ctx, n_state])
    let input = b.add_tensor_f32(&[n_ctx, n_state])?;

    // LayerNorm (operates on 2D: [n_ctx, n_state])
    let ln_out = add_layer_norm_2d(
        &mut b, input,
        &weights(&format!("{prefix}.attn_ln.weight"))?,
        &weights(&format!("{prefix}.attn_ln.bias"))?,
        n_ctx, n_state,
    )?;

    // Q projection: FC [n_ctx, n_state] → [n_ctx, n_state]
    let q_out = add_fc(
        &mut b, ln_out,
        &weights(&format!("{prefix}.attn.query.weight"))?,
        &weights(&format!("{prefix}.attn.query.bias"))?,
        n_ctx, n_state, n_state,
    )?;

    // K projection (no bias)
    let k_out = add_fc_no_bias(
        &mut b, ln_out,
        &weights(&format!("{prefix}.attn.key.weight"))?,
        n_ctx, n_state, n_state,
    )?;

    // V projection
    let v_out = add_fc(
        &mut b, ln_out,
        &weights(&format!("{prefix}.attn.value.weight"))?,
        &weights(&format!("{prefix}.attn.value.bias"))?,
        n_ctx, n_state, n_state,
    )?;

    b.finish_and_compile(device, &[input], &[q_out, k_out, v_out], cache_dir, cache_token)
}

/// Graph B: attn_out + residual → out_proj → add residual → LN → FFN → add residual
/// Model inputs: [attn_out, residual]
/// Model outputs: [output]
fn build_ffn_graph<F>(
    lib: &NnapiLib,
    device: *const ANeuralNetworksDevice,
    n_state: u32,
    n_ctx: u32,
    prefix: &str,
    is_last: bool,
    weights: &mut F,
    cache_dir: Option<&str>,
    cache_token: Option<&[u8; 32]>,
) -> Result<NnapiCompiled, String>
where
    F: FnMut(&str) -> Result<Vec<f32>, String>,
{
    let mut b = NnapiModelBuilder::new(lib)?;
    let n_ff = 4 * n_state;

    // Inputs
    let attn_out = b.add_tensor_f32(&[n_ctx, n_state])?;
    let residual = b.add_tensor_f32(&[n_ctx, n_state])?;

    // Output projection
    let proj_out = add_fc(
        &mut b, attn_out,
        &weights(&format!("{prefix}.attn.out.weight"))?,
        &weights(&format!("{prefix}.attn.out.bias"))?,
        n_ctx, n_state, n_state,
    )?;

    // Residual 1: residual + proj_out
    let res1 = add_add_2d(&mut b, residual, proj_out, n_ctx, n_state)?;

    // LayerNorm 2
    let ln2_out = add_layer_norm_2d(
        &mut b, res1,
        &weights(&format!("{prefix}.mlp_ln.weight"))?,
        &weights(&format!("{prefix}.mlp_ln.bias"))?,
        n_ctx, n_state,
    )?;

    // FC1: [n_ctx, n_state] → [n_ctx, 4*n_state]
    let fc1_out = add_fc(
        &mut b, ln2_out,
        &weights(&format!("{prefix}.mlp.0.weight"))?,
        &weights(&format!("{prefix}.mlp.0.bias"))?,
        n_ctx, n_state, n_ff,
    )?;

    // GELU ≈ x * sigmoid(1.702 * x)
    let gelu_out = add_gelu_approx(&mut b, fc1_out, n_ctx, n_ff)?;

    // FC2: [n_ctx, 4*n_state] → [n_ctx, n_state]
    let fc2_out = add_fc(
        &mut b, gelu_out,
        &weights(&format!("{prefix}.mlp.2.weight"))?,
        &weights(&format!("{prefix}.mlp.2.bias"))?,
        n_ctx, n_ff, n_state,
    )?;

    // Residual 2
    let mut output = add_add_2d(&mut b, res1, fc2_out, n_ctx, n_state)?;

    // Post-LayerNorm (last layer only)
    if is_last {
        output = add_layer_norm_2d(
            &mut b, output,
            &weights("encoder.ln_post.weight")?,
            &weights("encoder.ln_post.bias")?,
            n_ctx, n_state,
        )?;
    }

    b.finish_and_compile(device, &[attn_out, residual], &[output], cache_dir, cache_token)
}

// ======================================================================
// Operation helpers — all use 2D tensors [n_ctx, n_state] for FC compat
// ======================================================================

fn add_weight(b: &mut NnapiModelBuilder, data: &[f32], dims: &[u32]) -> Result<u32, String> {
    let idx = b.add_tensor_f32(dims)?;
    b.set_tensor_f32(idx, data)?;
    Ok(idx)
}

/// FULLY_CONNECTED: input [batch, in_size] @ weights [out_size, in_size]^T + bias → [batch, out_size]
fn add_fc(
    b: &mut NnapiModelBuilder,
    input: u32,
    w_data: &[f32],
    b_data: &[f32],
    batch: u32,
    in_size: u32,
    out_size: u32,
) -> Result<u32, String> {
    let w = add_weight(b, w_data, &[out_size, in_size])?;
    let bias = add_weight(b, b_data, &[out_size])?;
    let fuse = b.const_i32(ANEURALNETWORKS_FUSED_NONE)?;
    let output = b.add_tensor_f32(&[batch, out_size])?;
    b.add_op(ANEURALNETWORKS_FULLY_CONNECTED, &[input, w, bias, fuse], &[output])?;
    Ok(output)
}

/// FULLY_CONNECTED without bias (use zero bias).
fn add_fc_no_bias(
    b: &mut NnapiModelBuilder,
    input: u32,
    w_data: &[f32],
    batch: u32,
    in_size: u32,
    out_size: u32,
) -> Result<u32, String> {
    let zeros = vec![0.0f32; out_size as usize];
    add_fc(b, input, w_data, &zeros, batch, in_size, out_size)
}

/// ADD: a + b (2D, fuse=NONE)
fn add_add_2d(b: &mut NnapiModelBuilder, a: u32, b_t: u32, m: u32, n: u32) -> Result<u32, String> {
    let fuse = b.const_i32(ANEURALNETWORKS_FUSED_NONE)?;
    let output = b.add_tensor_f32(&[m, n])?;
    b.add_op(ANEURALNETWORKS_ADD, &[a, b_t, fuse], &[output])?;
    Ok(output)
}

/// MUL: a * b (2D, fuse=NONE)
fn add_mul_2d(b: &mut NnapiModelBuilder, a: u32, b_t: u32, m: u32, n: u32) -> Result<u32, String> {
    let fuse = b.const_i32(ANEURALNETWORKS_FUSED_NONE)?;
    let output = b.add_tensor_f32(&[m, n])?;
    b.add_op(ANEURALNETWORKS_MUL, &[a, b_t, fuse], &[output])?;
    Ok(output)
}

/// SUB: a - b (2D, fuse=NONE)
fn add_sub_2d(b: &mut NnapiModelBuilder, a: u32, b_t: u32, m: u32, n: u32) -> Result<u32, String> {
    let fuse = b.const_i32(ANEURALNETWORKS_FUSED_NONE)?;
    let output = b.add_tensor_f32(&[m, n])?;
    b.add_op(ANEURALNETWORKS_SUB, &[a, b_t, fuse], &[output])?;
    Ok(output)
}

/// MUL by scalar broadcast: input [m, n] * scalar [1] → [m, n]
fn add_mul_scalar(b: &mut NnapiModelBuilder, input: u32, scalar: f32, m: u32, n: u32) -> Result<u32, String> {
    let s = add_weight(b, &[scalar], &[1])?;
    add_mul_2d(b, input, s, m, n)
}

/// GELU ≈ x * sigmoid(1.702 * x)
fn add_gelu_approx(b: &mut NnapiModelBuilder, input: u32, m: u32, n: u32) -> Result<u32, String> {
    // temp = 1.702 * x
    let temp = add_mul_scalar(b, input, 1.702, m, n)?;
    // sig = sigmoid(temp)
    let sig = b.add_tensor_f32(&[m, n])?;
    b.add_op(ANEURALNETWORKS_LOGISTIC, &[temp], &[sig])?;
    // gelu = x * sig
    add_mul_2d(b, input, sig, m, n)
}

/// SOFTMAX (2D)
#[allow(dead_code)]
fn add_softmax_2d(b: &mut NnapiModelBuilder, input: u32, m: u32, n: u32) -> Result<u32, String> {
    let beta = b.const_f32(1.0)?;
    let output = b.add_tensor_f32(&[m, n])?;
    b.add_op(ANEURALNETWORKS_SOFTMAX, &[input, beta], &[output])?;
    Ok(output)
}

/// LayerNorm on 2D tensor [m, n]:
/// (x - mean) / sqrt(var + eps) * gamma + beta
fn add_layer_norm_2d(
    b: &mut NnapiModelBuilder,
    input: u32,
    gamma: &[f32],
    beta: &[f32],
    m: u32,
    n: u32,
) -> Result<u32, String> {
    let full = [m, n];
    let reduced = [m, 1u32];

    // axis = [1] (last dim of 2D)
    let axis = b.add_tensor_i32(&[1])?;
    b.set_tensor_i32(axis, &[1])?;
    let keepdims = b.const_i32(1)?;

    // mu = mean(x, axis=1, keepdims)
    let mu = b.add_tensor_f32(&reduced)?;
    b.add_op(ANEURALNETWORKS_MEAN, &[input, axis, keepdims], &[mu])?;

    // diff = x - mu (broadcast [m, n] - [m, 1])
    let diff = add_sub_2d(b, input, mu, m, n)?;

    // diff_sq = diff * diff
    let diff_sq = add_mul_2d(b, diff, diff, m, n)?;

    // axis2 for second mean
    let axis2 = b.add_tensor_i32(&[1])?;
    b.set_tensor_i32(axis2, &[1])?;
    let keepdims2 = b.const_i32(1)?;

    // var = mean(diff_sq, axis=1, keepdims)
    let var = b.add_tensor_f32(&reduced)?;
    b.add_op(ANEURALNETWORKS_MEAN, &[diff_sq, axis2, keepdims2], &[var])?;

    // var_eps = var + epsilon
    let eps = add_weight(b, &[1e-5], &[1])?;
    let fuse_none = b.const_i32(ANEURALNETWORKS_FUSED_NONE)?;
    let var_eps = b.add_tensor_f32(&reduced)?;
    b.add_op(ANEURALNETWORKS_ADD, &[var, eps, fuse_none], &[var_eps])?;

    // inv_std = rsqrt(var_eps)
    let inv_std = b.add_tensor_f32(&reduced)?;
    b.add_op(ANEURALNETWORKS_RSQRT, &[var_eps], &[inv_std])?;

    // normalized = diff * inv_std (broadcast)
    let normalized = add_mul_2d(b, diff, inv_std, m, n)?;

    // scaled = normalized * gamma (broadcast [m, n] * [n])
    let gamma_t = add_weight(b, gamma, &[n])?;
    let scaled = add_mul_2d(b, normalized, gamma_t, m, n)?;

    // output = scaled + beta (broadcast)
    let beta_t = add_weight(b, beta, &[n])?;
    let fuse_none2 = b.const_i32(ANEURALNETWORKS_FUSED_NONE)?;
    let output = b.add_tensor_f32(&full)?;
    b.add_op(ANEURALNETWORKS_ADD, &[scaled, beta_t, fuse_none2], &[output])?;
    Ok(output)
}
