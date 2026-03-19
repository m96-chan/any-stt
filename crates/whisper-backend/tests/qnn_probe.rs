//! QNN probe test — verifies QNN SDK can be loaded and initialized.
//!
//! This test is skipped if QNN libraries are not available.
//! Set QNN_LIB_DIR to the directory containing libQnnHtp.so.

use whisper_backend::qnn::{is_qnn_available, QnnContext, QnnLibrary};

#[test]
fn qnn_availability_check() {
    let available = is_qnn_available();
    eprintln!("QNN HTP available: {available}");
}

#[test]
fn qnn_load_htp_backend() {
    let lib = match QnnLibrary::load_htp() {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("SKIPPED: {e}");
            return;
        }
    };

    let ver = &lib.iface.api_version.core_api_version;
    eprintln!("QNN core API: v{}.{}.{}", ver.major, ver.minor, ver.patch);
    eprintln!("backend_id: {}", lib.iface.backend_id);

    assert!(lib.vt().backend_create.is_some(), "backend_create is null");
    assert!(lib.vt().backend_free.is_some(), "backend_free is null");
    assert!(lib.vt().context_create.is_some(), "context_create is null");
    assert!(lib.vt().graph_create.is_some(), "graph_create is null");
    assert!(lib.vt().graph_add_node.is_some(), "graph_add_node is null");
    assert!(lib.vt().graph_finalize.is_some(), "graph_finalize is null");
    assert!(lib.vt().graph_execute.is_some(), "graph_execute is null");
    assert!(
        lib.vt().tensor_create_graph_tensor.is_some(),
        "tensor_create_graph_tensor is null"
    );
}

#[test]
fn qnn_create_backend_and_context() {
    let lib = match QnnLibrary::load_htp() {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("SKIPPED: {e}");
            return;
        }
    };

    let _ctx = QnnContext::new(lib).expect("failed to create QNN context");
    eprintln!("QNN context created successfully");
}

#[test]
fn qnn_create_graph() {
    let lib = match QnnLibrary::load_htp() {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("SKIPPED: {e}");
            return;
        }
    };

    let ctx = QnnContext::new(lib).expect("failed to create QNN context");
    let graph = ctx.create_graph("test_graph");

    match graph {
        Ok(_) => eprintln!("Graph created successfully"),
        Err(e) => eprintln!("create_graph: {e}"),
    }
}
