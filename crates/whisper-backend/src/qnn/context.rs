//! QNN context management — backend + context + graph lifecycle.

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

        let backend_create = vt.backend_create.ok_or("backend_create null")?;
        let mut backend: Qnn_BackendHandle_t = ptr::null_mut();
        let err = unsafe { backend_create(ptr::null_mut(), ptr::null(), &mut backend) };
        if err != QNN_SUCCESS {
            return Err(format!("backend_create: error {err}"));
        }

        let context_create = vt.context_create.ok_or("context_create null")?;
        let mut context: Qnn_ContextHandle_t = ptr::null_mut();
        let err =
            unsafe { context_create(backend, ptr::null_mut(), ptr::null(), &mut context) };
        if err != QNN_SUCCESS {
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

    /// Create a named graph.
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

    /// Register a tensor with the graph. Backend assigns `tensor.v1.id`.
    pub fn register_tensor(
        &self,
        graph: Qnn_GraphHandle_t,
        tensor: &mut Qnn_Tensor_t,
    ) -> Result<u32, String> {
        let create_tensor = self
            .lib
            .vt()
            .tensor_create_graph_tensor
            .ok_or("tensor_create_graph_tensor null")?;
        let err = unsafe { create_tensor(graph, tensor) };
        if err != QNN_SUCCESS {
            return Err(format!("tensorCreateGraphTensor: error {err}"));
        }
        Ok(tensor.v1.id)
    }

    /// Add a node (operation) to a graph.
    pub fn add_node(
        &self,
        graph: Qnn_GraphHandle_t,
        op: Qnn_OpConfig_t,
    ) -> Result<(), String> {
        let graph_add_node = self.lib.vt().graph_add_node.ok_or("graph_add_node null")?;
        let err = unsafe { graph_add_node(graph, op) };
        if err != QNN_SUCCESS {
            return Err(format!("graph_add_node: error {err}"));
        }
        Ok(())
    }

    /// Finalize (compile) a graph.
    pub fn finalize_graph(&self, graph: Qnn_GraphHandle_t) -> Result<(), String> {
        let graph_finalize = self.lib.vt().graph_finalize.ok_or("graph_finalize null")?;
        let err = unsafe { graph_finalize(graph, ptr::null_mut(), ptr::null_mut()) };
        if err != QNN_SUCCESS {
            return Err(format!("graph_finalize: error {err}"));
        }
        Ok(())
    }

    /// Execute a finalized graph.
    pub fn execute(
        &self,
        graph: Qnn_GraphHandle_t,
        inputs: &mut [Qnn_Tensor_t],
        outputs: &mut [Qnn_Tensor_t],
    ) -> Result<(), String> {
        let graph_execute = self.lib.vt().graph_execute.ok_or("graph_execute null")?;
        let err = unsafe {
            graph_execute(
                graph,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if err != QNN_SUCCESS {
            return Err(format!("graph_execute: error {err}"));
        }
        Ok(())
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
