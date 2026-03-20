//! QNN operation builder helpers.
//!
//! Provides high-level functions for constructing QNN operations (MatMul,
//! LayerNorm, Softmax, GELU, etc.) used in the Whisper encoder graph.
//! These wrap the low-level QNN types from types.rs.

use std::ffi::CString;

use crate::context::QnnContext;
use crate::types::*;

/// Counter for generating unique tensor/op names within a graph.
pub struct NameGen {
    prefix: String,
    counter: u32,
}

impl NameGen {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            counter: 0,
        }
    }

    pub fn next(&mut self, suffix: &str) -> CString {
        let name = format!("{}_{}{}", self.prefix, self.counter, suffix);
        self.counter += 1;
        CString::new(name).unwrap()
    }
}

/// Package name for QNN built-in ops.
pub fn qti_aisw() -> CString {
    CString::new("qti.aisw").unwrap()
}

/// Register a static tensor (weights embedded in the graph).
/// For tensors < 4MB, this allows HTP to pre-load them to DSP.
pub fn register_static_tensor(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    dims: &mut [u32],
    data: &mut [f32],
) -> Result<Qnn_Tensor_t, String> {
    let mut tensor = Qnn_Tensor_t::new(
        name.as_ptr(),
        QNN_TENSOR_TYPE_STATIC,
        dims.len() as u32,
        dims.as_mut_ptr(),
    );
    tensor.set_data(data);
    ctx.register_tensor(graph, &mut tensor)?;
    Ok(tensor)
}

/// Register an APP_WRITE tensor (dynamic input/intermediate).
pub fn register_app_write_tensor(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    dims: &mut [u32],
) -> Result<Qnn_Tensor_t, String> {
    let mut tensor = Qnn_Tensor_t::app_write(name.as_ptr(), dims.len() as u32, dims.as_mut_ptr());
    ctx.register_tensor(graph, &mut tensor)?;
    Ok(tensor)
}

/// Register a NATIVE tensor (intermediate, managed by backend).
pub fn register_native_tensor(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    dims: &mut [u32],
) -> Result<Qnn_Tensor_t, String> {
    let mut tensor = Qnn_Tensor_t::new(
        name.as_ptr(),
        QNN_TENSOR_TYPE_NATIVE,
        dims.len() as u32,
        dims.as_mut_ptr(),
    );
    ctx.register_tensor(graph, &mut tensor)?;
    Ok(tensor)
}

/// Register an APP_READ tensor (output read back to CPU).
pub fn register_app_read_tensor(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    dims: &mut [u32],
) -> Result<Qnn_Tensor_t, String> {
    let mut tensor = Qnn_Tensor_t::app_read(name.as_ptr(), dims.len() as u32, dims.as_mut_ptr());
    ctx.register_tensor(graph, &mut tensor)?;
    Ok(tensor)
}

/// Add a MatMul operation: C = A @ B
/// A: [M, K], B: [K, N] -> C: [M, N]
pub fn add_matmul(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    input_a: &Qnn_Tensor_t,
    input_b: &Qnn_Tensor_t,
    output: &Qnn_Tensor_t,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<(), String> {
    let pkg = qti_aisw();
    let type_name = CString::new("MatMul").unwrap();
    let tp0_name = CString::new("transpose_in0").unwrap();
    let tp1_name = CString::new("transpose_in1").unwrap();

    let mut params = [
        Qnn_Param_t::scalar(tp0_name.as_ptr(), Qnn_Scalar_t::bool8(transpose_a)),
        Qnn_Param_t::scalar(tp1_name.as_ptr(), Qnn_Scalar_t::bool8(transpose_b)),
    ];

    // Clone tensors for the op (addNode takes ownership of the array)
    let mut inputs = [clone_tensor(input_a), clone_tensor(input_b)];
    let mut outputs = [clone_tensor(output)];

    let op = Qnn_OpConfig_t::new(
        name.as_ptr(),
        pkg.as_ptr(),
        type_name.as_ptr(),
        &mut params,
        &mut inputs,
        &mut outputs,
    );
    ctx.add_node(graph, op)
}

/// Add a LayerNorm operation.
/// input: [*, n_state], weight: [n_state], bias: [n_state] -> output: [*, n_state]
pub fn add_layer_norm(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    input: &Qnn_Tensor_t,
    weight: &Qnn_Tensor_t,
    bias: &Qnn_Tensor_t,
    output: &Qnn_Tensor_t,
    epsilon: f32,
) -> Result<(), String> {
    let pkg = qti_aisw();
    let type_name = CString::new("LayerNorm").unwrap();
    let eps_name = CString::new("epsilon").unwrap();
    let axes_name = CString::new("axes").unwrap();

    let mut params = [
        Qnn_Param_t::scalar(eps_name.as_ptr(), Qnn_Scalar_t::float32(epsilon)),
        // axes param: normalize over last dimension
        // For LayerNorm, QNN expects axes=[last_dim_index]
        // We use scalar param for the axis
        Qnn_Param_t::scalar(axes_name.as_ptr(), Qnn_Scalar_t::bool8(false)), // placeholder
    ];

    let mut inputs = [
        clone_tensor(input),
        clone_tensor(weight),
        clone_tensor(bias),
    ];
    let mut outputs = [clone_tensor(output)];

    let op = Qnn_OpConfig_t::new(
        name.as_ptr(),
        pkg.as_ptr(),
        type_name.as_ptr(),
        &mut params[..1], // only epsilon for now
        &mut inputs,
        &mut outputs,
    );
    ctx.add_node(graph, op)
}

/// Add a Softmax operation over the last dimension.
pub fn add_softmax(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    input: &Qnn_Tensor_t,
    output: &Qnn_Tensor_t,
) -> Result<(), String> {
    let pkg = qti_aisw();
    let type_name = CString::new("Softmax").unwrap();

    let mut inputs = [clone_tensor(input)];
    let mut outputs = [clone_tensor(output)];

    let op = Qnn_OpConfig_t::new(
        name.as_ptr(),
        pkg.as_ptr(),
        type_name.as_ptr(),
        &mut [],
        &mut inputs,
        &mut outputs,
    );
    ctx.add_node(graph, op)
}

/// Add a GELU activation.
pub fn add_gelu(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    input: &Qnn_Tensor_t,
    output: &Qnn_Tensor_t,
) -> Result<(), String> {
    let pkg = qti_aisw();
    let type_name = CString::new("Gelu").unwrap();

    let mut inputs = [clone_tensor(input)];
    let mut outputs = [clone_tensor(output)];

    let op = Qnn_OpConfig_t::new(
        name.as_ptr(),
        pkg.as_ptr(),
        type_name.as_ptr(),
        &mut [],
        &mut inputs,
        &mut outputs,
    );
    ctx.add_node(graph, op)
}

/// Add element-wise addition (residual connection).
pub fn add_element_wise_add(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    input_a: &Qnn_Tensor_t,
    input_b: &Qnn_Tensor_t,
    output: &Qnn_Tensor_t,
) -> Result<(), String> {
    let pkg = qti_aisw();
    let type_name = CString::new("ElementWiseAdd").unwrap();

    let mut inputs = [clone_tensor(input_a), clone_tensor(input_b)];
    let mut outputs = [clone_tensor(output)];

    let op = Qnn_OpConfig_t::new(
        name.as_ptr(),
        pkg.as_ptr(),
        type_name.as_ptr(),
        &mut [],
        &mut inputs,
        &mut outputs,
    );
    ctx.add_node(graph, op)
}

/// Add a Reshape operation.
pub fn add_reshape(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    input: &Qnn_Tensor_t,
    output: &Qnn_Tensor_t,
) -> Result<(), String> {
    let pkg = qti_aisw();
    let type_name = CString::new("Reshape").unwrap();

    let mut inputs = [clone_tensor(input)];
    let mut outputs = [clone_tensor(output)];

    let op = Qnn_OpConfig_t::new(
        name.as_ptr(),
        pkg.as_ptr(),
        type_name.as_ptr(),
        &mut [],
        &mut inputs,
        &mut outputs,
    );
    ctx.add_node(graph, op)
}

/// Add a Transpose/Permute operation.
pub fn add_transpose(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    input: &Qnn_Tensor_t,
    output: &Qnn_Tensor_t,
    perm: &Qnn_Tensor_t,
) -> Result<(), String> {
    let pkg = qti_aisw();
    let type_name = CString::new("Transpose").unwrap();

    let mut inputs = [clone_tensor(input), clone_tensor(perm)];
    let mut outputs = [clone_tensor(output)];

    let op = Qnn_OpConfig_t::new(
        name.as_ptr(),
        pkg.as_ptr(),
        type_name.as_ptr(),
        &mut [],
        &mut inputs,
        &mut outputs,
    );
    ctx.add_node(graph, op)
}

/// Add element-wise multiply (for scaling in attention).
pub fn add_element_wise_multiply(
    ctx: &QnnContext,
    graph: Qnn_GraphHandle_t,
    name: &CString,
    input_a: &Qnn_Tensor_t,
    input_b: &Qnn_Tensor_t,
    output: &Qnn_Tensor_t,
) -> Result<(), String> {
    let pkg = qti_aisw();
    let type_name = CString::new("ElementWiseMultiply").unwrap();

    let mut inputs = [clone_tensor(input_a), clone_tensor(input_b)];
    let mut outputs = [clone_tensor(output)];

    let op = Qnn_OpConfig_t::new(
        name.as_ptr(),
        pkg.as_ptr(),
        type_name.as_ptr(),
        &mut [],
        &mut inputs,
        &mut outputs,
    );
    ctx.add_node(graph, op)
}

/// Shallow clone a tensor descriptor (copies the struct, not the data).
fn clone_tensor(t: &Qnn_Tensor_t) -> Qnn_Tensor_t {
    // SAFETY: Qnn_Tensor_t is repr(C) with no Drop — memcpy is fine
    unsafe { std::ptr::read(t) }
}
