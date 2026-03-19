//! QNN MatMul test — runs a matrix multiplication on Hexagon HTP NPU.
//!
//! Requires QNN SDK libraries. Set QNN_LIB_DIR and ADSP_LIBRARY_PATH.
//! Skips gracefully when QNN is not available.

use std::ffi::CString;

use qnn_backend::types::*;
use qnn_backend::{QnnContext, QnnLibrary};

#[test]
fn qnn_matmul_on_htp() {
    let lib = match QnnLibrary::load_htp() {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("SKIPPED: {e}");
            return;
        }
    };

    let ctx = match QnnContext::new(lib) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("SKIPPED (context): {e}");
            return;
        }
    };

    let graph = ctx.create_graph("matmul_test").expect("graph_create");

    // MatMul: A[2,3] * B[3,2] = C[2,2]
    let a_name = CString::new("a").unwrap();
    let b_name = CString::new("b").unwrap();
    let c_name = CString::new("c").unwrap();

    let mut a_dims = [2u32, 3];
    let mut b_dims = [3u32, 2];
    let mut c_dims = [2u32, 2];

    // Register tensors (data=null at registration time)
    let mut ta = Qnn_Tensor_t::app_write(a_name.as_ptr(), 2, a_dims.as_mut_ptr());
    let mut tb = Qnn_Tensor_t::app_write(b_name.as_ptr(), 2, b_dims.as_mut_ptr());
    let mut tc = Qnn_Tensor_t::app_read(c_name.as_ptr(), 2, c_dims.as_mut_ptr());

    let a_id = ctx.register_tensor(graph, &mut ta).expect("register a");
    let b_id = ctx.register_tensor(graph, &mut tb).expect("register b");
    let c_id = ctx.register_tensor(graph, &mut tc).expect("register c");

    eprintln!("Tensor IDs: a={a_id}, b={b_id}, c={c_id}");

    // MatMul op with transpose params
    let op_name = CString::new("mm").unwrap();
    let pkg_name = CString::new("qti.aisw").unwrap();
    let type_name = CString::new("MatMul").unwrap();
    let tp0_name = CString::new("transpose_in0").unwrap();
    let tp1_name = CString::new("transpose_in1").unwrap();

    let mut params = [
        Qnn_Param_t::scalar(tp0_name.as_ptr(), Qnn_Scalar_t::bool8(false)),
        Qnn_Param_t::scalar(tp1_name.as_ptr(), Qnn_Scalar_t::bool8(false)),
    ];

    let mut inputs = [ta, tb];
    let mut outputs = [tc];

    let op = Qnn_OpConfig_t::new(
        op_name.as_ptr(),
        pkg_name.as_ptr(),
        type_name.as_ptr(),
        &mut params,
        &mut inputs,
        &mut outputs,
    );

    ctx.add_node(graph, op).expect("add_node MatMul");
    eprintln!("addNode OK");

    ctx.finalize_graph(graph).expect("finalize");
    eprintln!("finalize OK");

    // Set data for execution
    let mut a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c_data: Vec<f32> = vec![0.0; 4];

    inputs[0].set_data(&mut a_data);
    inputs[1].set_data(&mut b_data);
    outputs[0].set_data(&mut c_data);

    ctx.execute(graph, &mut inputs, &mut outputs)
        .expect("execute");

    eprintln!("Result:   {:?}", c_data);
    eprintln!("Expected: [58.0, 64.0, 139.0, 154.0]");

    // Verify with tolerance (HTP may use FP16 internally)
    let expected = [58.0f32, 64.0, 139.0, 154.0];
    for (i, (got, exp)) in c_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1.0,
            "element {i}: got {got}, expected {exp}"
        );
    }

    eprintln!("SUCCESS — MatMul on Hexagon HTP verified from Rust!");
}
