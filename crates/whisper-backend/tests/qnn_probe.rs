//! QNN probe test — verifies QNN SDK can be loaded and initialized.
//!
//! This test is skipped if QNN libraries are not available.
//! Set QNN_LIB_DIR to the directory containing libQnnHtp.so.

use whisper_backend::qnn::{is_qnn_available, QnnContext, QnnLibrary};

#[test]
fn qnn_availability_check() {
    let available = is_qnn_available();
    eprintln!("QNN HTP available: {available}");
    // This test always passes — it just reports availability.
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

    let backend_ver = &lib.iface.api_version.backend_api_version;
    eprintln!(
        "QNN backend API: v{}.{}.{}",
        backend_ver.major, backend_ver.minor, backend_ver.patch
    );

    eprintln!("backend_id: {}", lib.iface.backend_id);

    // Verify key function pointers are populated.
    assert!(lib.vt().backend_create.is_some(), "backend_create is null");
    assert!(lib.vt().backend_free.is_some(), "backend_free is null");
    assert!(lib.vt().context_create.is_some(), "context_create is null");
    assert!(lib.vt().graph_create.is_some(), "graph_create is null");
    assert!(lib.vt().graph_finalize.is_some(), "graph_finalize is null");
    assert!(lib.vt().graph_execute.is_some(), "graph_execute is null");
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

    let ctx = QnnContext::new(lib).expect("failed to create QNN context");

    // Get API version through the context.
    match ctx.api_version() {
        Ok(ver) => {
            eprintln!(
                "Backend reports API v{}.{}.{}",
                ver.core_api_version.major,
                ver.core_api_version.minor,
                ver.core_api_version.patch
            );
        }
        Err(e) => eprintln!("api_version: {e}"),
    }

    // Get build ID.
    match ctx.build_id() {
        Ok(id) => eprintln!("Build ID: {id}"),
        Err(e) => eprintln!("build_id: {e}"),
    }
}

#[test]
fn qnn_create_empty_graph() {
    let lib = match QnnLibrary::load_htp() {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("SKIPPED: {e}");
            return;
        }
    };

    let ctx = QnnContext::new(lib).expect("failed to create QNN context");
    let graph = ctx.create_graph("test_empty_graph");

    match graph {
        Ok(_) => eprintln!("Empty graph created successfully"),
        Err(e) => eprintln!("create_graph: {e} (may be expected for HTP without ops)"),
    }
}
