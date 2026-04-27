//! FastConformer encoder building blocks.
//!
//! This module hosts the f32 reference implementation of the encoder.
//! It will eventually be backed by ggml graphs (CPU/CUDA/Metal/Vulkan)
//! and QNN/NNAPI graphs for the NPU hot path. Today, the goal is to
//! have a correct CPU forward that matches NeMo numerics within ~1e-3
//! per layer, validated by `tests/layer_reference.rs`.
//!
//! ## Module map
//! ```text
//! ops.rs        — primitive ops (matvec, linear, layer_norm, swish, GLU,
//!                 softmax, batch_norm)
//! subsample.rs  — striding conv2d ×8 (TODO)
//! attention.rs  — multi-head rel-pos with optional Longformer (TODO)
//! conv_module.rs — pointwise / depthwise / GLU / batch_norm (TODO)
//! ff.rs         — feed-forward macaron half (TODO)
//! block.rs      — full Conformer block (TODO)
//! ```
//!
//! ## Status
//! ops.rs is complete and unit-tested. The remaining modules are
//! scaffolded as stubs — they have the type signatures the rest of the
//! codebase will rely on, but their forward methods return
//! `NotImplemented`. Each carries a TODO list of what's left to wire.

pub mod ops;

pub use ops::{
    batch_norm_1d_inference, glu_channels, layer_norm, layer_norm_rows, linear, matvec_add,
    softmax_inplace, swish_inplace,
};
