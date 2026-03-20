//! QNN-based Whisper encoder graph builder.
//!
//! Builds a QNN graph implementing the Whisper encoder transformer blocks:
//!   LayerNorm → Multi-Head Attention → Residual → LayerNorm → FFN → Residual
//!
//! The preprocessor (Conv1d + GELU + pos_embed) runs on CPU because
//! Conv1d is not supported on Hexagon HTP.

use crate::context::QnnContext;
use crate::ops::*;
use crate::types::*;

/// Configuration for the Whisper encoder.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Hidden state dimension: 384 (tiny), 512 (base), 768 (small), 1024 (medium), 1280 (large)
    pub n_state: u32,
    /// Number of attention heads: 6 (tiny), 8 (base), 12 (small), 16 (medium), 20 (large)
    pub n_head: u32,
    /// Number of encoder layers: 4 (tiny), 6 (base), 12 (small), 24 (medium), 32 (large)
    pub n_layer: u32,
    /// Audio context length (always 1500 for standard Whisper)
    pub n_ctx: u32,
    /// Graph splitting strategy for large models
    pub split_strategy: SplitStrategy,
}

impl EncoderConfig {
    /// Tiny model configuration.
    pub fn tiny() -> Self {
        Self {
            n_state: 384,
            n_head: 6,
            n_layer: 4,
            n_ctx: 1500,
            split_strategy: SplitStrategy::SingleGraph,
        }
    }

    /// Base model configuration.
    pub fn base() -> Self {
        Self {
            n_state: 512,
            n_head: 8,
            n_layer: 6,
            n_ctx: 1500,
            split_strategy: SplitStrategy::SingleGraph,
        }
    }

    /// Small model configuration.
    pub fn small() -> Self {
        Self {
            n_state: 768,
            n_head: 12,
            n_layer: 12,
            n_ctx: 1500,
            split_strategy: SplitStrategy::SingleGraph,
        }
    }

    /// Medium model configuration.
    pub fn medium() -> Self {
        Self {
            n_state: 1024,
            n_head: 16,
            n_layer: 24,
            n_ctx: 1500,
            split_strategy: SplitStrategy::PerLayer,
        }
    }

    /// Large model configuration.
    pub fn large() -> Self {
        Self {
            n_state: 1280,
            n_head: 20,
            n_layer: 32,
            n_ctx: 1500,
            split_strategy: SplitStrategy::PerLayer,
        }
    }

    /// Head dimension (n_state / n_head).
    pub fn head_dim(&self) -> u32 {
        self.n_state / self.n_head
    }
}

/// Strategy for splitting the encoder graph.
///
/// Large models may exceed HTP memory limits when compiled as a single graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitStrategy {
    /// Compile all layers into one graph (tiny/base/small).
    SingleGraph,
    /// One graph per encoder layer (medium+).
    PerLayer,
    /// Split attention heads across multiple graphs (large).
    HeadSplit { heads_per_chunk: u32 },
}

/// A compiled Whisper encoder ready for execution on QNN HTP.
pub struct WhisperEncoderGraph {
    ctx: QnnContext,
    /// One graph handle per split chunk.
    graphs: Vec<Qnn_GraphHandle_t>,
    config: EncoderConfig,
    /// Input tensors for execute (one per graph).
    input_tensors: Vec<Vec<Qnn_Tensor_t>>,
    /// Output tensors for execute (one per graph).
    output_tensors: Vec<Vec<Qnn_Tensor_t>>,
}

// SAFETY: QNN handles are internally thread-safe for non-concurrent access.
// WhisperEncoderGraph enforces non-concurrent use via &mut self on execute.
unsafe impl Send for WhisperEncoderGraph {}
unsafe impl Sync for WhisperEncoderGraph {}

impl WhisperEncoderGraph {
    /// Build the encoder graph from weights.
    ///
    /// `weights` is a closure that returns the dequantized f32 weight for a given name.
    /// Weight names follow the whisper.cpp convention:
    ///   `encoder.blocks.{layer}.{component}.{param}`
    pub fn build<F>(
        ctx: QnnContext,
        config: EncoderConfig,
        mut weights: F,
    ) -> Result<Self, String>
    where
        F: FnMut(&str) -> Result<Vec<f32>, String>,
    {
        match config.split_strategy {
            SplitStrategy::SingleGraph => {
                Self::build_single_graph(ctx, config, &mut weights)
            }
            SplitStrategy::PerLayer => {
                Self::build_per_layer(ctx, config, &mut weights)
            }
            SplitStrategy::HeadSplit { .. } => {
                Err("HeadSplit not yet implemented".into())
            }
        }
    }

    /// Build all encoder layers as a single graph.
    fn build_single_graph<F>(
        ctx: QnnContext,
        config: EncoderConfig,
        weights: &mut F,
    ) -> Result<Self, String>
    where
        F: FnMut(&str) -> Result<Vec<f32>, String>,
    {
        let graph = ctx.create_graph("whisper_encoder")?;
        let mut names = NameGen::new("enc");
        let n_ctx = config.n_ctx;
        let n_state = config.n_state;

        // Input tensor: encoder input [1, n_ctx, n_state]
        // (after CPU preprocessor: conv1d + gelu + conv1d + gelu + pos_embed)
        let input_name = names.next("_input");
        let mut input_dims = [1u32, n_ctx, n_state];
        let input = register_app_write_tensor(&ctx, graph, &input_name, &mut input_dims)?;

        let mut current = input.clone_desc();

        // Build each encoder layer
        for layer in 0..config.n_layer {
            current = Self::build_encoder_block(
                &ctx, graph, &config, &mut names, weights, &current, layer,
            )?;
        }

        // Final layer norm
        let ln_post_w_name = names.next("_ln_post_w");
        let ln_post_b_name = names.next("_ln_post_b");
        let mut ln_post_w_data = weights("encoder.ln_post.weight")?;
        let mut ln_post_b_data = weights("encoder.ln_post.bias")?;
        let mut ln_dims = [n_state];
        let ln_post_w = register_static_tensor(
            &ctx, graph, &ln_post_w_name, &mut ln_dims, &mut ln_post_w_data,
        )?;
        let mut ln_dims2 = [n_state];
        let ln_post_b = register_static_tensor(
            &ctx, graph, &ln_post_b_name, &mut ln_dims2, &mut ln_post_b_data,
        )?;

        let output_name = names.next("_output");
        let mut output_dims = [1u32, n_ctx, n_state];
        let output = register_app_read_tensor(&ctx, graph, &output_name, &mut output_dims)?;

        let ln_name = names.next("_ln_post");
        add_layer_norm(
            &ctx, graph, &ln_name, &current, &ln_post_w, &ln_post_b, &output, 1e-5,
        )?;

        ctx.finalize_graph(graph)?;

        Ok(Self {
            ctx,
            graphs: vec![graph],
            config,
            input_tensors: vec![vec![input]],
            output_tensors: vec![vec![output]],
        })
    }

    /// Build one graph per encoder layer.
    fn build_per_layer<F>(
        ctx: QnnContext,
        config: EncoderConfig,
        weights: &mut F,
    ) -> Result<Self, String>
    where
        F: FnMut(&str) -> Result<Vec<f32>, String>,
    {
        let n_ctx = config.n_ctx;
        let n_state = config.n_state;
        let mut graphs = Vec::new();
        let mut all_inputs = Vec::new();
        let mut all_outputs = Vec::new();

        for layer in 0..config.n_layer {
            let graph_name = format!("enc_layer_{layer}");
            let graph = ctx.create_graph(&graph_name)?;
            let mut names = NameGen::new(&format!("l{layer}"));

            // Input
            let input_name = names.next("_in");
            let mut input_dims = [1u32, n_ctx, n_state];
            let input = register_app_write_tensor(&ctx, graph, &input_name, &mut input_dims)?;

            let current = Self::build_encoder_block(
                &ctx, graph, &config, &mut names, weights, &input, layer,
            )?;

            // If last layer, add post-LN
            if layer == config.n_layer - 1 {
                let ln_post_w_name = names.next("_ln_post_w");
                let ln_post_b_name = names.next("_ln_post_b");
                let mut ln_post_w_data = weights("encoder.ln_post.weight")?;
                let mut ln_post_b_data = weights("encoder.ln_post.bias")?;
                let mut ln_dims = [n_state];
                let ln_post_w = register_static_tensor(
                    &ctx, graph, &ln_post_w_name, &mut ln_dims, &mut ln_post_w_data,
                )?;
                let mut ln_dims2 = [n_state];
                let ln_post_b = register_static_tensor(
                    &ctx, graph, &ln_post_b_name, &mut ln_dims2, &mut ln_post_b_data,
                )?;

                let output_name = names.next("_out");
                let mut output_dims = [1u32, n_ctx, n_state];
                let output =
                    register_app_read_tensor(&ctx, graph, &output_name, &mut output_dims)?;

                let ln_name = names.next("_ln_post");
                add_layer_norm(
                    &ctx, graph, &ln_name, &current, &ln_post_w, &ln_post_b, &output, 1e-5,
                )?;

                all_outputs.push(vec![output]);
            } else {
                let output_name = names.next("_out");
                let mut output_dims = [1u32, n_ctx, n_state];
                let output =
                    register_app_read_tensor(&ctx, graph, &output_name, &mut output_dims)?;

                // Identity pass-through (reshape as no-op)
                let reshape_name = names.next("_passthru");
                add_reshape(&ctx, graph, &reshape_name, &current, &output)?;

                all_outputs.push(vec![output]);
            }

            ctx.finalize_graph(graph)?;
            graphs.push(graph);
            all_inputs.push(vec![input]);
        }

        Ok(Self {
            ctx,
            graphs,
            config,
            input_tensors: all_inputs,
            output_tensors: all_outputs,
        })
    }

    /// Build a single encoder transformer block.
    ///
    /// Returns the output tensor (native, to be chained to the next block).
    fn build_encoder_block<F>(
        ctx: &QnnContext,
        graph: Qnn_GraphHandle_t,
        config: &EncoderConfig,
        names: &mut NameGen,
        weights: &mut F,
        input: &Qnn_Tensor_t,
        layer: u32,
    ) -> Result<Qnn_Tensor_t, String>
    where
        F: FnMut(&str) -> Result<Vec<f32>, String>,
    {
        let n_ctx = config.n_ctx;
        let n_state = config.n_state;
        let prefix = format!("encoder.blocks.{layer}");

        // ===== Self-Attention =====
        // 1. LayerNorm
        let ln1_w_name = names.next("_attn_ln_w");
        let ln1_b_name = names.next("_attn_ln_b");
        let mut ln1_w_data = weights(&format!("{prefix}.attn_ln.weight"))?;
        let mut ln1_b_data = weights(&format!("{prefix}.attn_ln.bias"))?;
        let mut ln_dims = [n_state];
        let ln1_w = register_static_tensor(ctx, graph, &ln1_w_name, &mut ln_dims, &mut ln1_w_data)?;
        let mut ln_dims2 = [n_state];
        let ln1_b = register_static_tensor(ctx, graph, &ln1_b_name, &mut ln_dims2, &mut ln1_b_data)?;

        let ln1_out_name = names.next("_attn_ln_out");
        let mut ln1_out_dims = [1u32, n_ctx, n_state];
        let ln1_out = register_native_tensor(ctx, graph, &ln1_out_name, &mut ln1_out_dims)?;

        let ln1_op_name = names.next("_attn_ln");
        add_layer_norm(ctx, graph, &ln1_op_name, input, &ln1_w, &ln1_b, &ln1_out, 1e-5)?;

        // 2. Q, K, V projections (fused as single matmul with [n_state, n_state] weight)
        let q_w_name = names.next("_q_w");
        let q_b_name = names.next("_q_b");
        let mut q_w_data = weights(&format!("{prefix}.attn.query.weight"))?;
        let mut q_b_data = weights(&format!("{prefix}.attn.query.bias"))?;
        let mut qkv_w_dims = [n_state, n_state];
        let q_w = register_static_tensor(ctx, graph, &q_w_name, &mut qkv_w_dims, &mut q_w_data)?;
        let mut bias_dims = [n_state];
        let _q_b = register_static_tensor(ctx, graph, &q_b_name, &mut bias_dims, &mut q_b_data)?;

        let k_w_name = names.next("_k_w");
        let mut k_w_data = weights(&format!("{prefix}.attn.key.weight"))?;
        let mut kw_dims = [n_state, n_state];
        let k_w = register_static_tensor(ctx, graph, &k_w_name, &mut kw_dims, &mut k_w_data)?;
        // Note: key has no bias in whisper

        let v_w_name = names.next("_v_w");
        let v_b_name = names.next("_v_b");
        let mut v_w_data = weights(&format!("{prefix}.attn.value.weight"))?;
        let mut v_b_data = weights(&format!("{prefix}.attn.value.bias"))?;
        let mut vw_dims = [n_state, n_state];
        let v_w = register_static_tensor(ctx, graph, &v_w_name, &mut vw_dims, &mut v_w_data)?;
        let mut vb_dims = [n_state];
        let _v_b = register_static_tensor(ctx, graph, &v_b_name, &mut vb_dims, &mut v_b_data)?;

        // Q = ln1_out @ Q_w^T + Q_b  -> [1, n_ctx, n_state]
        let q_out_name = names.next("_q");
        let mut q_out_dims = [1u32, n_ctx, n_state];
        let q_out = register_native_tensor(ctx, graph, &q_out_name, &mut q_out_dims)?;
        let q_mm_name = names.next("_q_mm");
        add_matmul(ctx, graph, &q_mm_name, &ln1_out, &q_w, &q_out, false, true)?;
        // TODO: add bias (ElementWiseAdd with broadcast)

        let k_out_name = names.next("_k");
        let mut k_out_dims = [1u32, n_ctx, n_state];
        let k_out = register_native_tensor(ctx, graph, &k_out_name, &mut k_out_dims)?;
        let k_mm_name = names.next("_k_mm");
        add_matmul(ctx, graph, &k_mm_name, &ln1_out, &k_w, &k_out, false, true)?;

        let v_out_name = names.next("_v");
        let mut v_out_dims = [1u32, n_ctx, n_state];
        let v_out = register_native_tensor(ctx, graph, &v_out_name, &mut v_out_dims)?;
        let v_mm_name = names.next("_v_mm");
        add_matmul(ctx, graph, &v_mm_name, &ln1_out, &v_w, &v_out, false, true)?;

        // 3. Attention: softmax(Q @ K^T / sqrt(d_k)) @ V
        //    For simplicity, we compute the full attention without head splitting.
        //    QK: [1, n_ctx, n_ctx]
        let qk_name = names.next("_qk");
        let mut qk_dims = [1u32, n_ctx, n_ctx];
        let qk = register_native_tensor(ctx, graph, &qk_name, &mut qk_dims)?;
        let qk_mm_name = names.next("_qk_mm");
        add_matmul(ctx, graph, &qk_mm_name, &q_out, &k_out, &qk, false, true)?;

        // Scale by 1/sqrt(d_k) — done via ElementWiseMultiply with scalar
        let scale_val = 1.0 / (config.head_dim() as f32).sqrt();
        let scale_name = names.next("_scale");
        let mut scale_data = [scale_val];
        let mut scale_dims = [1u32];
        let scale_tensor = register_static_tensor(
            ctx, graph, &scale_name, &mut scale_dims, &mut scale_data,
        )?;

        let qk_scaled_name = names.next("_qk_scaled");
        let mut qk_scaled_dims = [1u32, n_ctx, n_ctx];
        let qk_scaled =
            register_native_tensor(ctx, graph, &qk_scaled_name, &mut qk_scaled_dims)?;
        let scale_op_name = names.next("_scale_op");
        add_element_wise_multiply(ctx, graph, &scale_op_name, &qk, &scale_tensor, &qk_scaled)?;

        // Softmax
        let sm_out_name = names.next("_sm");
        let mut sm_dims = [1u32, n_ctx, n_ctx];
        let sm_out = register_native_tensor(ctx, graph, &sm_out_name, &mut sm_dims)?;
        let sm_op_name = names.next("_softmax");
        add_softmax(ctx, graph, &sm_op_name, &qk_scaled, &sm_out)?;

        // Attention @ V -> [1, n_ctx, n_state]
        let attn_out_name = names.next("_attn_v");
        let mut attn_out_dims = [1u32, n_ctx, n_state];
        let attn_out = register_native_tensor(ctx, graph, &attn_out_name, &mut attn_out_dims)?;
        let attn_mm_name = names.next("_attn_v_mm");
        add_matmul(ctx, graph, &attn_mm_name, &sm_out, &v_out, &attn_out, false, false)?;

        // 4. Output projection
        let out_w_name = names.next("_attn_out_w");
        let out_b_name = names.next("_attn_out_b");
        let mut out_w_data = weights(&format!("{prefix}.attn.out.weight"))?;
        let mut out_b_data = weights(&format!("{prefix}.attn.out.bias"))?;
        let mut outw_dims = [n_state, n_state];
        let out_w = register_static_tensor(ctx, graph, &out_w_name, &mut outw_dims, &mut out_w_data)?;
        let mut outb_dims = [n_state];
        let out_b = register_static_tensor(ctx, graph, &out_b_name, &mut outb_dims, &mut out_b_data)?;
        let _ = out_b; // TODO: add bias

        let proj_out_name = names.next("_attn_proj");
        let mut proj_out_dims = [1u32, n_ctx, n_state];
        let proj_out = register_native_tensor(ctx, graph, &proj_out_name, &mut proj_out_dims)?;
        let proj_mm_name = names.next("_attn_proj_mm");
        add_matmul(ctx, graph, &proj_mm_name, &attn_out, &out_w, &proj_out, false, true)?;

        // 5. Residual connection: input + attn_output
        let res1_name = names.next("_res1");
        let mut res1_dims = [1u32, n_ctx, n_state];
        let res1 = register_native_tensor(ctx, graph, &res1_name, &mut res1_dims)?;
        let res1_op_name = names.next("_res1_add");
        add_element_wise_add(ctx, graph, &res1_op_name, input, &proj_out, &res1)?;

        // ===== Feed-Forward Network =====
        // 6. LayerNorm
        let ln2_w_name = names.next("_mlp_ln_w");
        let ln2_b_name = names.next("_mlp_ln_b");
        let mut ln2_w_data = weights(&format!("{prefix}.mlp_ln.weight"))?;
        let mut ln2_b_data = weights(&format!("{prefix}.mlp_ln.bias"))?;
        let mut ln2w_dims = [n_state];
        let ln2_w = register_static_tensor(ctx, graph, &ln2_w_name, &mut ln2w_dims, &mut ln2_w_data)?;
        let mut ln2b_dims = [n_state];
        let ln2_b = register_static_tensor(ctx, graph, &ln2_b_name, &mut ln2b_dims, &mut ln2_b_data)?;

        let ln2_out_name = names.next("_mlp_ln_out");
        let mut ln2_out_dims = [1u32, n_ctx, n_state];
        let ln2_out = register_native_tensor(ctx, graph, &ln2_out_name, &mut ln2_out_dims)?;
        let ln2_op_name = names.next("_mlp_ln");
        add_layer_norm(ctx, graph, &ln2_op_name, &res1, &ln2_w, &ln2_b, &ln2_out, 1e-5)?;

        // 7. FC1: [n_ctx, n_state] @ [n_state, 4*n_state] -> [n_ctx, 4*n_state]
        let n_ff = 4 * n_state;
        let fc1_w_name = names.next("_mlp_0_w");
        let fc1_b_name = names.next("_mlp_0_b");
        let mut fc1_w_data = weights(&format!("{prefix}.mlp.0.weight"))?;
        let mut fc1_b_data = weights(&format!("{prefix}.mlp.0.bias"))?;
        let mut fc1w_dims = [n_ff, n_state];
        let fc1_w = register_static_tensor(ctx, graph, &fc1_w_name, &mut fc1w_dims, &mut fc1_w_data)?;
        let mut fc1b_dims = [n_ff];
        let fc1_b = register_static_tensor(ctx, graph, &fc1_b_name, &mut fc1b_dims, &mut fc1_b_data)?;
        let _ = fc1_b; // TODO: add bias

        let fc1_out_name = names.next("_fc1");
        let mut fc1_out_dims = [1u32, n_ctx, n_ff];
        let fc1_out = register_native_tensor(ctx, graph, &fc1_out_name, &mut fc1_out_dims)?;
        let fc1_mm_name = names.next("_fc1_mm");
        add_matmul(ctx, graph, &fc1_mm_name, &ln2_out, &fc1_w, &fc1_out, false, true)?;

        // 8. GELU activation
        let gelu_out_name = names.next("_gelu");
        let mut gelu_out_dims = [1u32, n_ctx, n_ff];
        let gelu_out = register_native_tensor(ctx, graph, &gelu_out_name, &mut gelu_out_dims)?;
        let gelu_op_name = names.next("_gelu_op");
        add_gelu(ctx, graph, &gelu_op_name, &fc1_out, &gelu_out)?;

        // 9. FC2: [n_ctx, 4*n_state] @ [4*n_state, n_state] -> [n_ctx, n_state]
        let fc2_w_name = names.next("_mlp_2_w");
        let fc2_b_name = names.next("_mlp_2_b");
        let mut fc2_w_data = weights(&format!("{prefix}.mlp.2.weight"))?;
        let mut fc2_b_data = weights(&format!("{prefix}.mlp.2.bias"))?;
        let mut fc2w_dims = [n_state, n_ff];
        let fc2_w = register_static_tensor(ctx, graph, &fc2_w_name, &mut fc2w_dims, &mut fc2_w_data)?;
        let mut fc2b_dims = [n_state];
        let fc2_b = register_static_tensor(ctx, graph, &fc2_b_name, &mut fc2b_dims, &mut fc2_b_data)?;
        let _ = fc2_b; // TODO: add bias

        let fc2_out_name = names.next("_fc2");
        let mut fc2_out_dims = [1u32, n_ctx, n_state];
        let fc2_out = register_native_tensor(ctx, graph, &fc2_out_name, &mut fc2_out_dims)?;
        let fc2_mm_name = names.next("_fc2_mm");
        add_matmul(ctx, graph, &fc2_mm_name, &gelu_out, &fc2_w, &fc2_out, false, true)?;

        // 10. Residual connection: res1 + ffn_output
        let res2_name = names.next("_res2");
        let mut res2_dims = [1u32, n_ctx, n_state];
        let res2 = register_native_tensor(ctx, graph, &res2_name, &mut res2_dims)?;
        let res2_op_name = names.next("_res2_add");
        add_element_wise_add(ctx, graph, &res2_op_name, &res1, &fc2_out, &res2)?;

        Ok(res2)
    }

    /// Execute the encoder: input [1, n_ctx, n_state] -> output [1, n_ctx, n_state].
    pub fn execute(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let n_ctx = self.config.n_ctx as usize;
        let n_state = self.config.n_state as usize;
        let expected_len = n_ctx * n_state;
        if input.len() != expected_len {
            return Err(format!(
                "input length mismatch: got {}, expected {expected_len}",
                input.len()
            ));
        }

        match self.config.split_strategy {
            SplitStrategy::SingleGraph => {
                let mut input_data = input.to_vec();
                let mut output_data = vec![0.0f32; expected_len];

                self.input_tensors[0][0].set_data(&mut input_data);
                self.output_tensors[0][0].set_data(&mut output_data);

                self.ctx.execute(
                    self.graphs[0],
                    &mut self.input_tensors[0],
                    &mut self.output_tensors[0],
                )?;

                Ok(output_data)
            }
            SplitStrategy::PerLayer => {
                let mut current = input.to_vec();
                for i in 0..self.graphs.len() {
                    let mut output_data = vec![0.0f32; expected_len];
                    self.input_tensors[i][0].set_data(&mut current);
                    self.output_tensors[i][0].set_data(&mut output_data);
                    self.ctx.execute(
                        self.graphs[i],
                        &mut self.input_tensors[i],
                        &mut self.output_tensors[i],
                    )?;
                    current = output_data;
                }
                Ok(current)
            }
            SplitStrategy::HeadSplit { .. } => {
                Err("HeadSplit execute not yet implemented".into())
            }
        }
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

/// Extension trait for Qnn_Tensor_t to clone the descriptor.
trait TensorDescClone {
    fn clone_desc(&self) -> Self;
}

impl TensorDescClone for Qnn_Tensor_t {
    fn clone_desc(&self) -> Self {
        unsafe { std::ptr::read(self) }
    }
}
