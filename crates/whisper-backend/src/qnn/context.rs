//! QNN context management — backend + context lifecycle.

use std::ffi::CString;
use std::ptr;

use super::loader::QnnLibrary;
use super::types::*;

/// A live QNN execution context.
pub struct QnnContext {
    pub lib: QnnLibrary,
    backend: Qnn_BackendHandle_t,
    context: Qnn_ContextHandle_t,
}

impl QnnContext {
    /// Initialize QNN: create backend and context.
    pub fn new(lib: QnnLibrary) -> Result<Self, String> {
        let vt = lib.vt();

        // Create backend (no logger for now).
        let backend_create = vt.backend_create.ok_or("backend_create null")?;
        let mut backend: Qnn_BackendHandle_t = ptr::null_mut();
        let err = unsafe { backend_create(ptr::null_mut(), ptr::null(), &mut backend) };
        if err != QNN_SUCCESS {
            return Err(format!("backend_create: error {err}"));
        }

        // Create context.
        let context_create = vt.context_create.ok_or("context_create null")?;
        let mut context: Qnn_ContextHandle_t = ptr::null_mut();
        let err =
            unsafe { context_create(backend, ptr::null_mut(), ptr::null(), &mut context) };
        if err != QNN_SUCCESS {
            // Clean up backend on failure.
            if let Some(f) = vt.backend_free {
                unsafe { f(backend) };
            }
            return Err(format!("context_create: error {err}"));
        }

        Ok(Self {
            lib,
            backend,
            context,
        })
    }

    /// Create a named graph in this context.
    pub fn create_graph(&self, name: &str) -> Result<Qnn_GraphHandle_t, String> {
        let graph_create = self.lib.vt().graph_create.ok_or("graph_create null")?;
        let name_c = CString::new(name).map_err(|e| e.to_string())?;
        let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
        let err =
            unsafe { graph_create(self.context, name_c.as_ptr(), ptr::null(), &mut graph) };
        if err != QNN_SUCCESS {
            return Err(format!("graph_create: error {err}"));
        }
        Ok(graph)
    }

    /// Finalize (compile) a graph for HTP execution.
    pub fn finalize_graph(&self, graph: Qnn_GraphHandle_t) -> Result<(), String> {
        let graph_finalize = self.lib.vt().graph_finalize.ok_or("graph_finalize null")?;
        let err = unsafe { graph_finalize(graph, ptr::null_mut(), ptr::null_mut()) };
        if err != QNN_SUCCESS {
            return Err(format!("graph_finalize: error {err}"));
        }
        Ok(())
    }

    /// Get the API version reported by the backend.
    pub fn api_version(&self) -> Result<Qnn_ApiVersion_t, String> {
        let get_ver = self
            .lib
            .vt()
            .backend_get_api_version
            .ok_or("backend_get_api_version null")?;
        let mut ver = Qnn_ApiVersion_t {
            core_api_version: Qnn_Version_t {
                major: 0,
                minor: 0,
                patch: 0,
            },
            backend_api_version: Qnn_Version_t {
                major: 0,
                minor: 0,
                patch: 0,
            },
        };
        let err = unsafe { get_ver(&mut ver) };
        if err != QNN_SUCCESS {
            return Err(format!("backend_get_api_version: error {err}"));
        }
        Ok(ver)
    }

    /// Get the build ID string from the backend.
    pub fn build_id(&self) -> Result<String, String> {
        let get_id = self
            .lib
            .vt()
            .backend_get_build_id
            .ok_or("backend_get_build_id null")?;
        let mut id_ptr: *const std::os::raw::c_char = ptr::null();
        let err = unsafe { get_id(&mut id_ptr) };
        if err != QNN_SUCCESS || id_ptr.is_null() {
            return Err(format!("backend_get_build_id: error {err}"));
        }
        let s = unsafe { std::ffi::CStr::from_ptr(id_ptr) }
            .to_string_lossy()
            .into_owned();
        Ok(s)
    }
}

impl Drop for QnnContext {
    fn drop(&mut self) {
        let vt = self.lib.vt();
        if let Some(f) = vt.context_free {
            unsafe { f(self.context, ptr::null_mut()) };
        }
        if let Some(f) = vt.backend_free {
            unsafe { f(self.backend) };
        }
    }
}
