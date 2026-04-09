//! NNAPI-based Whisper encoder graph builder.
//!
//! Splits each encoder layer into NPU-accelerated subgraphs with CPU attention:
//!   Graph A (NPU): LayerNorm → Q/K/V FC projections
//!   CPU:           Q@K^T → scale → softmax → attn@V
//!   Graph B (NPU): output FC → residual → LayerNorm → FC1 → GELU → FC2 → residual
//!
//! Uses FULLY_CONNECTED instead of BATCH_MATMUL (not supported on MediaTek NPU).
//! GELU is approximated as x * sigmoid(1.702 * x).
//!
//! Memory strategy: layers are compiled in chunks and freed after execution to
//! allow 32-layer large models to run within 3GB RAM.

use crate::context::{NnapiCompiled, NnapiModelBuilder};
use crate::loader::NnapiLib;
use crate::types::*;

/// Max layers to keep compiled simultaneously (memory limit).
/// Each compiled layer ≈ 30-60MB for large models (2 graphs × 15-30MB).
/// 8 layers × 60MB = ~480MB — fits alongside whisper.cpp model (537MB).
const CHUNK_SIZE: u32 = 8;

/// Whisper encoder that runs on NNAPI NPU + CPU attention.
///
/// Instead of pre-compiling all layers, builds and executes in chunks to
/// avoid OOM on memory-constrained devices.
pub struct WhisperEncoderNnapi {
    /// Reference to the NNAPI library (must outlive this struct).
    lib: *const NnapiLib,
    device: *const ANeuralNetworksDevice,
    n_ctx: u32,
    n_state: u32,
    n_head: u32,
    n_layer: u32,
    /// Pre-compiled layers (if they fit in memory).
    precompiled: Vec<EncoderLayer>,
    /// Weight provider — stored for on-demand compilation.
    /// None if all layers are precompiled.
    weight_fn: Option<Box<dyn FnMut(&str) -> Result<Vec<f32>, String>>>,
    cache_dir: Option<String>,
}

struct EncoderLayer {
    qkv_graph: NnapiCompiled,
    ffn_graph: NnapiCompiled,
}

unsafe impl Send for WhisperEncoderNnapi {}
unsafe impl Sync for WhisperEncoderNnapi {}

impl WhisperEncoderNnapi {
    /// Build the encoder. Compiles as many layers as fit in memory upfront.
    /// Remaining layers are compiled on-demand during execution.
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
        F: FnMut(&str) -> Result<Vec<f32>, String> + 'static,
    {
        let cache_dir = std::env::var("NNAPI_CACHE_DIR").ok();

        // Try to precompile first chunk to verify NNAPI works
        let first_chunk = CHUNK_SIZE.min(n_layer);
        let mut precompiled = Vec::new();

        for layer_idx in 0..first_chunk {
            eprintln!("NNAPI: building layer {layer_idx}/{n_layer}...");
            let is_last = layer_idx == n_layer - 1;
            let layer = build_layer_with_cache(
                lib, device, n_state, n_head, n_ctx, layer_idx, is_last,
                &mut weights, cache_dir.as_deref(),
            )?;
            precompiled.push(layer);
        }

        Ok(Self {
            lib: lib as *const NnapiLib,
            device,
            n_ctx,
            n_state,
            n_head,
            n_layer,
            precompiled,
            weight_fn: if first_chunk < n_layer {
                Some(Box::new(weights))
            } else {
                None
            },
            cache_dir,
        })
    }

    fn lib(&self) -> &NnapiLib {
        unsafe { &*self.lib }
    }

    /// Debug: Execute Layer 0 QKV graph only, return Q output for comparison.
    pub fn debug_layer0_q(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let nc = self.n_ctx as usize;
        let ns = self.n_state as usize;
        let out_bytes = nc * ns * 4;
        if self.precompiled.is_empty() {
            return Err("no precompiled layers".into());
        }
        let qkv = self.precompiled[0].qkv_graph.execute(
            &[(0, input)],
            &[(0, out_bytes), (1, out_bytes), (2, out_bytes)],
        )?;
        Ok(qkv[0].clone())
    }

    /// Execute: input [n_ctx * n_state] → output [n_ctx * n_state].
    pub fn execute(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let nc = self.n_ctx as usize;
        let ns = self.n_state as usize;
        let nh = self.n_head as usize;
        let head_dim = ns / nh;
        let expected = nc * ns;
        if input.len() != expected {
            return Err(format!("input length {}, expected {expected}", input.len()));
        }

        let mut current = input.to_vec();

        // Execute precompiled layers first
        let precompiled_count = self.precompiled.len() as u32;
        for (i, layer) in self.precompiled.iter().enumerate() {
            current = execute_layer(layer, &current, nc, ns, head_dim, i)?;
        }

        // Execute remaining layers in chunks (on-demand compile → execute → free)
        if precompiled_count < self.n_layer {
            let remaining_start = precompiled_count;
            let remaining_end = self.n_layer;

            // Take weight_fn out of self to avoid borrow conflicts
            let mut weight_fn = self.weight_fn.take()
                .ok_or("weight function unavailable")?;
            let lib = unsafe { &*self.lib };
            let device = self.device;
            let n_state = self.n_state;
            let n_head = self.n_head;
            let n_ctx = self.n_ctx;
            let n_layer = self.n_layer;
            let cache_dir_ref = self.cache_dir.as_deref();

            let mut chunk_start = remaining_start;
            while chunk_start < remaining_end {
                let chunk_end = (chunk_start + CHUNK_SIZE).min(remaining_end);
                eprintln!("NNAPI: compiling layers {chunk_start}..{chunk_end}");
                let mut chunk_layers = Vec::new();

                for layer_idx in chunk_start..chunk_end {
                    let is_last = layer_idx == n_layer - 1;
                    let wf: &mut dyn FnMut(&str) -> Result<Vec<f32>, String> = &mut *weight_fn;
                    let layer = build_layer_with_cache(
                        lib, device, n_state, n_head, n_ctx,
                        layer_idx, is_last, wf, cache_dir_ref,
                    )?;
                    chunk_layers.push(layer);
                }

                for (j, layer) in chunk_layers.iter().enumerate() {
                    let abs_idx = (chunk_start + j as u32) as usize;
                    current = execute_layer(layer, &current, nc, ns, head_dim, abs_idx)?;
                }

                drop(chunk_layers);
                chunk_start = chunk_end;
            }

            // Put weight_fn back
            self.weight_fn = Some(weight_fn);
        }

        Ok(current)
    }
}

/// Execute a single layer: NPU QKV → CPU attention → NPU FFN.
fn execute_layer(
    layer: &EncoderLayer,
    current: &[f32],
    nc: usize,
    ns: usize,
    head_dim: usize,
    layer_idx: usize,
) -> Result<Vec<f32>, String> {
    let out_bytes = nc * ns * 4;

    // Graph A: LayerNorm + QKV projections (NPU)
    let qkv_results = layer.qkv_graph.execute(
        &[(0, current)],
        &[(0, out_bytes), (1, out_bytes), (2, out_bytes)],
    ).map_err(|e| format!("layer {layer_idx} qkv: {e}"))?;

    let q = &qkv_results[0];
    let k = &qkv_results[1];
    let v = &qkv_results[2];

    // CPU: Multi-head attention
    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn_out = cpu_attention(q, k, v, nc, ns, scale);

    // Graph B: output proj + residual + LN + FFN (NPU)
    // Inputs: (0: attn_out, 1: residual=current)
    let ffn_results = layer.ffn_graph.execute(
        &[(0, &attn_out), (1, current)],
        &[(0, out_bytes)],
    ).map_err(|e| format!("layer {layer_idx} ffn: {e}"))?;

    let output = ffn_results.into_iter().next()
        .ok_or_else(|| format!("layer {layer_idx}: no output"))?;

    if layer_idx < 4 {
        eprintln!("  L{layer_idx} out[0..4]: {:.4?}", &output[..4.min(output.len())]);
    }
    // Dump layer outputs for offline comparison
    if layer_idx < 4 {
        let path = format!("/data/local/tmp/any-stt/layer{layer_idx}_output.bin");
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(output.as_ptr() as *const u8, output.len() * 4)
        };
        let _ = std::fs::write(&path, bytes);
    }

    Ok(output)
}

/// Multi-head CPU attention: split Q/K/V into heads, attend per head, concatenate.
///
/// Q, K, V: [nc, ns] where ns = n_head * head_dim
/// Output: [nc, ns]
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], nc: usize, ns: usize, scale: f32) -> Vec<f32> {
    // Determine number of heads from scale: scale = 1/sqrt(head_dim)
    // head_dim = 1/scale^2, n_head = ns / head_dim
    let head_dim = (1.0 / (scale * scale)).round() as usize;
    let n_head = ns / head_dim;

    if n_head <= 1 || head_dim * n_head != ns {
        // Fallback to single-head
        return cpu_attention_single(q, k, v, nc, ns, scale);
    }

    // Multi-head: Q[nc, ns] → reshape to [nc, n_head, head_dim] → per-head attention
    let mut out = vec![0.0f32; nc * ns];

    for h in 0..n_head {
        let offset = h * head_dim;
        // Extract head slices: q_h[i] = q[i*ns + offset .. +head_dim]
        let mut q_h = vec![0.0f32; nc * head_dim];
        let mut k_h = vec![0.0f32; nc * head_dim];
        let mut v_h = vec![0.0f32; nc * head_dim];
        for i in 0..nc {
            q_h[i * head_dim..(i + 1) * head_dim].copy_from_slice(&q[i * ns + offset..i * ns + offset + head_dim]);
            k_h[i * head_dim..(i + 1) * head_dim].copy_from_slice(&k[i * ns + offset..i * ns + offset + head_dim]);
            v_h[i * head_dim..(i + 1) * head_dim].copy_from_slice(&v[i * ns + offset..i * ns + offset + head_dim]);
        }

        // Single-head attention on this head
        let head_out = cpu_attention_single(&q_h, &k_h, &v_h, nc, head_dim, scale);

        // Write back to output
        for i in 0..nc {
            out[i * ns + offset..i * ns + offset + head_dim].copy_from_slice(&head_out[i * head_dim..(i + 1) * head_dim]);
        }
    }

    out
}

/// Single-head attention: Q @ K^T * scale → softmax → @ V
fn cpu_attention_single(q: &[f32], k: &[f32], v: &[f32], nc: usize, ns: usize, scale: f32) -> Vec<f32> {
    use std::sync::Arc;
    use std::thread;

    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // --- QK^T: [nc, ns] @ [ns, nc] → [nc, nc], scaled ---
    let mut qk = vec![0.0f32; nc * nc];
    {
        let q_arc = Arc::new(q.to_vec());
        let k_arc = Arc::new(k.to_vec());

        let rows_per_thread = (nc + n_threads - 1) / n_threads;
        let mut handles = Vec::with_capacity(n_threads);

        for t in 0..n_threads {
            let row_start = t * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(nc);
            if row_start >= nc { break; }
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

    // --- Softmax over last dim ---
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
    // Transpose V for cache-friendly access
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
            if row_start >= nc { break; }
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

fn build_layer_with_cache(
    lib: &NnapiLib,
    device: *const ANeuralNetworksDevice,
    n_state: u32,
    _n_head: u32,
    n_ctx: u32,
    layer_idx: u32,
    is_last: bool,
    weights: &mut dyn FnMut(&str) -> Result<Vec<f32>, String>,
    cache_dir: Option<&str>,
) -> Result<EncoderLayer, String>
{
    let prefix = format!("encoder.blocks.{layer_idx}");

    let cache_token = cache_dir.map(|_| {
        let mut token = [0u8; 32];
        let key = format!("whisper_enc_{n_state}_{_n_head}_{n_ctx}_{layer_idx}_{is_last}");
        for (i, b) in key.bytes().enumerate() {
            token[i % 32] ^= b;
        }
        token
    });

    let qkv_token = cache_token.map(|mut t| { t[31] ^= 0xAA; t });

    let qkv_graph = build_qkv_graph(
        lib, device, n_state, n_ctx, &prefix, weights,
        cache_dir, qkv_token.as_ref(),
    )?;

    let ffn_graph = build_ffn_graph(
        lib, device, n_state, n_ctx, &prefix, is_last, weights,
        cache_dir, cache_token.as_ref(),
    )?;

    Ok(EncoderLayer { qkv_graph, ffn_graph })
}

/// Graph A: input [n_ctx, n_state] → LayerNorm → Q, K, V via FC
fn build_qkv_graph(
    lib: &NnapiLib,
    device: *const ANeuralNetworksDevice,
    n_state: u32,
    n_ctx: u32,
    prefix: &str,
    weights: &mut dyn FnMut(&str) -> Result<Vec<f32>, String>,
    cache_dir: Option<&str>,
    cache_token: Option<&[u8; 32]>,
) -> Result<NnapiCompiled, String>
{
    let mut b = NnapiModelBuilder::new(lib)?;

    let input = b.add_tensor_f32(&[n_ctx, n_state])?;

    let ln_out = add_layer_norm_2d(
        &mut b, input,
        &weights(&format!("{prefix}.attn_ln.weight"))?,
        &weights(&format!("{prefix}.attn_ln.bias"))?,
        n_ctx, n_state,
    )?;

    let q_out = add_fc(
        &mut b, ln_out,
        &weights(&format!("{prefix}.attn.query.weight"))?,
        &weights(&format!("{prefix}.attn.query.bias"))?,
        n_ctx, n_state, n_state,
    )?;

    let k_out = add_fc_no_bias(
        &mut b, ln_out,
        &weights(&format!("{prefix}.attn.key.weight"))?,
        n_ctx, n_state, n_state,
    )?;

    let v_out = add_fc(
        &mut b, ln_out,
        &weights(&format!("{prefix}.attn.value.weight"))?,
        &weights(&format!("{prefix}.attn.value.bias"))?,
        n_ctx, n_state, n_state,
    )?;

    b.finish_and_compile(device, &[input], &[q_out, k_out, v_out], cache_dir, cache_token)
}

/// Graph B: attn_out + residual → out_proj → add → LN → FFN → add
fn build_ffn_graph(
    lib: &NnapiLib,
    device: *const ANeuralNetworksDevice,
    n_state: u32,
    n_ctx: u32,
    prefix: &str,
    is_last: bool,
    weights: &mut dyn FnMut(&str) -> Result<Vec<f32>, String>,
    cache_dir: Option<&str>,
    cache_token: Option<&[u8; 32]>,
) -> Result<NnapiCompiled, String>
{
    let mut b = NnapiModelBuilder::new(lib)?;
    let n_ff = 4 * n_state;

    let attn_out = b.add_tensor_f32(&[n_ctx, n_state])?;
    let residual = b.add_tensor_f32(&[n_ctx, n_state])?;

    let proj_out = add_fc(
        &mut b, attn_out,
        &weights(&format!("{prefix}.attn.out.weight"))?,
        &weights(&format!("{prefix}.attn.out.bias"))?,
        n_ctx, n_state, n_state,
    )?;

    let res1 = add_add_2d(&mut b, residual, proj_out, n_ctx, n_state)?;

    let ln2_out = add_layer_norm_2d(
        &mut b, res1,
        &weights(&format!("{prefix}.mlp_ln.weight"))?,
        &weights(&format!("{prefix}.mlp_ln.bias"))?,
        n_ctx, n_state,
    )?;

    let fc1_out = add_fc(
        &mut b, ln2_out,
        &weights(&format!("{prefix}.mlp.0.weight"))?,
        &weights(&format!("{prefix}.mlp.0.bias"))?,
        n_ctx, n_state, n_ff,
    )?;

    let gelu_out = add_gelu_approx(&mut b, fc1_out, n_ctx, n_ff)?;

    let fc2_out = add_fc(
        &mut b, gelu_out,
        &weights(&format!("{prefix}.mlp.2.weight"))?,
        &weights(&format!("{prefix}.mlp.2.bias"))?,
        n_ctx, n_ff, n_state,
    )?;

    let mut output = add_add_2d(&mut b, res1, fc2_out, n_ctx, n_state)?;

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

fn add_add_2d(b: &mut NnapiModelBuilder, a: u32, b_t: u32, m: u32, n: u32) -> Result<u32, String> {
    let fuse = b.const_i32(ANEURALNETWORKS_FUSED_NONE)?;
    let output = b.add_tensor_f32(&[m, n])?;
    b.add_op(ANEURALNETWORKS_ADD, &[a, b_t, fuse], &[output])?;
    Ok(output)
}

fn add_mul_2d(b: &mut NnapiModelBuilder, a: u32, b_t: u32, m: u32, n: u32) -> Result<u32, String> {
    let fuse = b.const_i32(ANEURALNETWORKS_FUSED_NONE)?;
    let output = b.add_tensor_f32(&[m, n])?;
    b.add_op(ANEURALNETWORKS_MUL, &[a, b_t, fuse], &[output])?;
    Ok(output)
}

fn add_sub_2d(b: &mut NnapiModelBuilder, a: u32, b_t: u32, m: u32, n: u32) -> Result<u32, String> {
    let fuse = b.const_i32(ANEURALNETWORKS_FUSED_NONE)?;
    let output = b.add_tensor_f32(&[m, n])?;
    b.add_op(ANEURALNETWORKS_SUB, &[a, b_t, fuse], &[output])?;
    Ok(output)
}

fn add_mul_scalar(b: &mut NnapiModelBuilder, input: u32, scalar: f32, m: u32, n: u32) -> Result<u32, String> {
    let s = add_weight(b, &[scalar], &[1])?;
    add_mul_2d(b, input, s, m, n)
}

/// GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// GPT-2 tanh approximation — more accurate than sigmoid approximation.
fn add_gelu_approx(b: &mut NnapiModelBuilder, input: u32, m: u32, n: u32) -> Result<u32, String> {
    let dims = [m, n];

    // x² = x * x
    let x2 = add_mul_2d(b, input, input, m, n)?;
    // x³ = x² * x
    let x3 = add_mul_2d(b, x2, input, m, n)?;
    // 0.044715 * x³
    let coeff_x3 = add_mul_scalar(b, x3, 0.044715, m, n)?;
    // inner = x + 0.044715 * x³
    let inner = add_add_2d(b, input, coeff_x3, m, n)?;
    // scaled = sqrt(2/π) * inner ≈ 0.7978846 * inner
    let scaled = add_mul_scalar(b, inner, 0.7978846, m, n)?;
    // tanh_out = tanh(scaled)
    let tanh_out = b.add_tensor_f32(&dims)?;
    b.add_op(ANEURALNETWORKS_TANH, &[scaled], &[tanh_out])?;
    // 1 + tanh_out
    let one = add_weight(b, &[1.0], &[1])?;
    let one_plus = add_add_2d(b, one, tanh_out, m, n)?;
    // 0.5 * x
    let half_x = add_mul_scalar(b, input, 0.5, m, n)?;
    // gelu = 0.5 * x * (1 + tanh_out)
    add_mul_2d(b, half_x, one_plus, m, n)
}

#[allow(dead_code)]
fn add_softmax_2d(b: &mut NnapiModelBuilder, input: u32, m: u32, n: u32) -> Result<u32, String> {
    let beta = b.const_f32(1.0)?;
    let output = b.add_tensor_f32(&[m, n])?;
    b.add_op(ANEURALNETWORKS_SOFTMAX, &[input, beta], &[output])?;
    Ok(output)
}

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

    let axis = b.add_tensor_i32(&[1])?;
    b.set_tensor_i32(axis, &[1])?;
    let keepdims = b.const_i32(1)?;

    let mu = b.add_tensor_f32(&reduced)?;
    b.add_op(ANEURALNETWORKS_MEAN, &[input, axis, keepdims], &[mu])?;

    let diff = add_sub_2d(b, input, mu, m, n)?;
    let diff_sq = add_mul_2d(b, diff, diff, m, n)?;

    let axis2 = b.add_tensor_i32(&[1])?;
    b.set_tensor_i32(axis2, &[1])?;
    let keepdims2 = b.const_i32(1)?;

    let var = b.add_tensor_f32(&reduced)?;
    b.add_op(ANEURALNETWORKS_MEAN, &[diff_sq, axis2, keepdims2], &[var])?;

    let eps = add_weight(b, &[1e-5], &[1])?;
    let fuse_none = b.const_i32(ANEURALNETWORKS_FUSED_NONE)?;
    let var_eps = b.add_tensor_f32(&reduced)?;
    b.add_op(ANEURALNETWORKS_ADD, &[var, eps, fuse_none], &[var_eps])?;

    let inv_std = b.add_tensor_f32(&reduced)?;
    b.add_op(ANEURALNETWORKS_RSQRT, &[var_eps], &[inv_std])?;

    let normalized = add_mul_2d(b, diff, inv_std, m, n)?;

    let gamma_t = add_weight(b, gamma, &[n])?;
    let scaled = add_mul_2d(b, normalized, gamma_t, m, n)?;

    let beta_t = add_weight(b, beta, &[n])?;
    let fuse_none2 = b.const_i32(ANEURALNETWORKS_FUSED_NONE)?;
    let output = b.add_tensor_f32(&full)?;
    b.add_op(ANEURALNETWORKS_ADD, &[scaled, beta_t, fuse_none2], &[output])?;
    Ok(output)
}
