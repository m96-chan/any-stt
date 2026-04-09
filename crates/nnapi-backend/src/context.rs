//! NNAPI model building, compilation, and execution.

use std::os::raw::c_void;
use std::ptr;

use crate::loader::NnapiLib;
use crate::types::*;

/// An NNAPI model under construction.
pub struct NnapiModelBuilder {
    pub lib: *const NnapiLib,
    model: *mut ANeuralNetworksModel,
    next_operand: u32,
    /// Keeps owned data alive until model_finish is called.
    _owned_data: Vec<Vec<u8>>,
}

impl NnapiModelBuilder {
    /// Create a new model builder.
    ///
    /// SAFETY: `lib` must outlive this builder.
    pub fn new(lib: &NnapiLib) -> Result<Self, String> {
        let mut model: *mut ANeuralNetworksModel = ptr::null_mut();
        let err = unsafe { (lib.model_create)(&mut model) };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("model_create: error {err}"));
        }
        Ok(Self {
            lib: lib as *const NnapiLib,
            model,
            next_operand: 0,
            _owned_data: Vec::new(),
        })
    }

    fn lib(&self) -> &NnapiLib {
        unsafe { &*self.lib }
    }

    /// Add a float32 tensor operand. Returns operand index.
    pub fn add_tensor_f32(&mut self, dims: &[u32]) -> Result<u32, String> {
        // Heap-allocate dims to ensure stable pointer
        let dims_vec: Vec<u32> = dims.to_vec();
        let op_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: dims_vec.len() as u32,
            dimensions: dims_vec.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };
        let result = self.add_operand(&op_type);
        // Keep dims_vec alive through the call (it's consumed here)
        drop(dims_vec);
        result
    }

    /// Add an int32 tensor operand.
    pub fn add_tensor_i32(&mut self, dims: &[u32]) -> Result<u32, String> {
        let dims_vec: Vec<u32> = dims.to_vec();
        let op_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_INT32,
            dimension_count: dims_vec.len() as u32,
            dimensions: dims_vec.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };
        let result = self.add_operand(&op_type);
        drop(dims_vec);
        result
    }

    /// Add a scalar float32 operand.
    pub fn add_scalar_f32(&mut self) -> Result<u32, String> {
        let op_type = ANeuralNetworksOperandType::scalar_f32();
        self.add_operand(&op_type)
    }

    /// Add a scalar int32 operand.
    pub fn add_scalar_i32(&mut self) -> Result<u32, String> {
        let op_type = ANeuralNetworksOperandType::scalar_i32();
        self.add_operand(&op_type)
    }

    /// Add a scalar bool operand.
    pub fn add_scalar_bool(&mut self) -> Result<u32, String> {
        let op_type = ANeuralNetworksOperandType::scalar_bool();
        self.add_operand(&op_type)
    }

    fn add_operand(&mut self, op_type: &ANeuralNetworksOperandType) -> Result<u32, String> {
        let idx = self.next_operand;
        let err = unsafe { (self.lib().model_add_operand)(self.model, op_type) };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!(
                "model_addOperand[{idx}](type={}, ndim={}): error {err}",
                op_type.type_, op_type.dimension_count,
            ));
        }
        self.next_operand += 1;
        Ok(idx)
    }

    /// Set a constant float32 tensor value.
    pub fn set_tensor_f32(&mut self, idx: u32, data: &[f32]) -> Result<(), String> {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let len = bytes.len();
        self._owned_data.push(bytes);
        let ptr = self._owned_data.last().unwrap().as_ptr() as *const c_void;
        let err =
            unsafe { (self.lib().model_set_operand_value)(self.model, idx as i32, ptr, len) };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("setOperandValue[{idx}]: error {err}"));
        }
        Ok(())
    }

    /// Set a constant int32 tensor value.
    pub fn set_tensor_i32(&mut self, idx: u32, data: &[i32]) -> Result<(), String> {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_ne_bytes()).collect();
        let len = bytes.len();
        self._owned_data.push(bytes);
        let ptr = self._owned_data.last().unwrap().as_ptr() as *const c_void;
        let err =
            unsafe { (self.lib().model_set_operand_value)(self.model, idx as i32, ptr, len) };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("setOperandValue[{idx}]: error {err}"));
        }
        Ok(())
    }

    /// Set a constant scalar float32 value.
    pub fn set_scalar_f32(&mut self, idx: u32, val: f32) -> Result<(), String> {
        let bytes = val.to_ne_bytes().to_vec();
        let len = bytes.len();
        self._owned_data.push(bytes);
        let ptr = self._owned_data.last().unwrap().as_ptr() as *const c_void;
        let err =
            unsafe { (self.lib().model_set_operand_value)(self.model, idx as i32, ptr, len) };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("setScalarF32[{idx}]: error {err}"));
        }
        Ok(())
    }

    /// Set a constant scalar int32 value.
    pub fn set_scalar_i32(&mut self, idx: u32, val: i32) -> Result<(), String> {
        let bytes = val.to_ne_bytes().to_vec();
        let len = bytes.len();
        self._owned_data.push(bytes);
        let ptr = self._owned_data.last().unwrap().as_ptr() as *const c_void;
        let err =
            unsafe { (self.lib().model_set_operand_value)(self.model, idx as i32, ptr, len) };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("setScalarI32[{idx}]: error {err}"));
        }
        Ok(())
    }

    /// Set a constant scalar bool value (NNAPI bool = 1 byte).
    pub fn set_scalar_bool(&mut self, idx: u32, val: bool) -> Result<(), String> {
        let bytes = vec![if val { 1u8 } else { 0u8 }];
        let len = bytes.len();
        self._owned_data.push(bytes);
        let ptr = self._owned_data.last().unwrap().as_ptr() as *const c_void;
        let err =
            unsafe { (self.lib().model_set_operand_value)(self.model, idx as i32, ptr, len) };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("setScalarBool[{idx}]: error {err}"));
        }
        Ok(())
    }

    /// Add a constant scalar i32 operand with value set inline.
    pub fn const_i32(&mut self, val: i32) -> Result<u32, String> {
        let idx = self.add_scalar_i32()?;
        self.set_scalar_i32(idx, val)?;
        Ok(idx)
    }

    /// Add a constant scalar f32 operand with value set inline.
    pub fn const_f32(&mut self, val: f32) -> Result<u32, String> {
        let idx = self.add_scalar_f32()?;
        self.set_scalar_f32(idx, val)?;
        Ok(idx)
    }

    /// Add a constant scalar bool operand with value set inline.
    pub fn const_bool(&mut self, val: bool) -> Result<u32, String> {
        let idx = self.add_scalar_bool()?;
        self.set_scalar_bool(idx, val)?;
        Ok(idx)
    }

    /// Add an operation.
    pub fn add_op(
        &self,
        op_type: i32,
        inputs: &[u32],
        outputs: &[u32],
    ) -> Result<(), String> {
        let err = unsafe {
            (self.lib().model_add_operation)(
                self.model,
                op_type,
                inputs.len() as u32,
                inputs.as_ptr(),
                outputs.len() as u32,
                outputs.as_ptr(),
            )
        };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("addOperation(op={op_type}): error {err}"));
        }
        Ok(())
    }

    /// Finalize and compile the model targeting a specific device.
    pub fn finish_and_compile(
        self,
        device: *const ANeuralNetworksDevice,
        model_inputs: &[u32],
        model_outputs: &[u32],
        cache_dir: Option<&str>,
        cache_token: Option<&[u8; 32]>,
    ) -> Result<NnapiCompiled, String> {
        let lib = self.lib();

        // Identify I/O
        let err = unsafe {
            (lib.model_identify_inputs_and_outputs)(
                self.model,
                model_inputs.len() as u32,
                model_inputs.as_ptr(),
                model_outputs.len() as u32,
                model_outputs.as_ptr(),
            )
        };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("identifyInputsAndOutputs: error {err}"));
        }

        // Finish model
        let err = unsafe { (lib.model_finish)(self.model) };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("model_finish: error {err}"));
        }

        // Create compilation targeting the device
        let devices = [device];
        let mut compilation: *mut ANeuralNetworksCompilation = ptr::null_mut();
        let err = unsafe {
            (lib.compilation_create_for_devices)(
                self.model,
                devices.as_ptr(),
                1,
                &mut compilation,
            )
        };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("compilation_createForDevices: error {err}"));
        }

        // Set preference to sustained speed
        unsafe {
            (lib.compilation_set_preference)(
                compilation,
                ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
            );
        }

        // Set caching if available
        if let (Some(set_caching), Some(dir), Some(token)) =
            (lib.compilation_set_caching, cache_dir, cache_token)
        {
            let dir_c = std::ffi::CString::new(dir).unwrap();
            unsafe {
                set_caching(compilation, dir_c.as_ptr() as *const _, token.as_ptr());
            }
        }

        // Compile (this is where NPU compilation happens — may take seconds)
        eprintln!("NNAPI: compiling model for NPU...");
        let err = unsafe { (lib.compilation_finish)(compilation) };
        if err != ANEURALNETWORKS_NO_ERROR {
            unsafe { (lib.compilation_free)(compilation) };
            return Err(format!("compilation_finish: error {err}"));
        }
        eprintln!("NNAPI: compilation done");

        // Free weight data after compilation to save memory.
        // The compiled model is self-contained (verified on mtk-neuron_shim).
        drop(self._owned_data);

        Ok(NnapiCompiled {
            lib: self.lib,
            _model: self.model,
            compilation,
        })
    }
}

/// A compiled NNAPI model ready for execution.
pub struct NnapiCompiled {
    lib: *const NnapiLib,
    _model: *mut ANeuralNetworksModel,
    compilation: *mut ANeuralNetworksCompilation,
}

// SAFETY: NNAPI handles are thread-safe for non-concurrent execution.
unsafe impl Send for NnapiCompiled {}
unsafe impl Sync for NnapiCompiled {}

impl NnapiCompiled {
    fn lib(&self) -> &NnapiLib {
        unsafe { &*self.lib }
    }

    /// Execute the compiled model.
    ///
    /// `inputs`: list of (input_index, data) pairs
    /// `outputs`: list of (output_index, buffer_size_bytes) pairs
    /// Returns output buffers in order.
    pub fn execute(
        &self,
        inputs: &[(i32, &[f32])],
        output_sizes: &[(i32, usize)],
    ) -> Result<Vec<Vec<f32>>, String> {
        let lib = self.lib();

        let mut execution: *mut ANeuralNetworksExecution = ptr::null_mut();
        let err = unsafe { (lib.execution_create)(self.compilation, &mut execution) };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("execution_create: error {err}"));
        }

        // Set inputs
        for &(idx, data) in inputs {
            let err = unsafe {
                (lib.execution_set_input)(
                    execution,
                    idx,
                    ptr::null(),
                    data.as_ptr() as *const c_void,
                    data.len() * std::mem::size_of::<f32>(),
                )
            };
            if err != ANEURALNETWORKS_NO_ERROR {
                unsafe { (lib.execution_free)(execution) };
                return Err(format!("execution_setInput[{idx}]: error {err}"));
            }
        }

        // Allocate and set outputs
        let mut output_bufs: Vec<Vec<f32>> = output_sizes
            .iter()
            .map(|&(_, size)| vec![0.0f32; size / std::mem::size_of::<f32>()])
            .collect();

        for (i, &(idx, _)) in output_sizes.iter().enumerate() {
            let buf = &mut output_bufs[i];
            let err = unsafe {
                (lib.execution_set_output)(
                    execution,
                    idx,
                    ptr::null(),
                    buf.as_mut_ptr() as *mut c_void,
                    buf.len() * std::mem::size_of::<f32>(),
                )
            };
            if err != ANEURALNETWORKS_NO_ERROR {
                unsafe { (lib.execution_free)(execution) };
                return Err(format!("execution_setOutput[{idx}]: error {err}"));
            }
        }

        // Compute
        let err = unsafe { (lib.execution_compute)(execution) };
        unsafe { (lib.execution_free)(execution) };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("execution_compute: error {err}"));
        }

        Ok(output_bufs)
    }
}

impl Drop for NnapiCompiled {
    fn drop(&mut self) {
        let lib = self.lib();
        unsafe {
            (lib.compilation_free)(self.compilation);
            (lib.model_free)(self._model);
        }
    }
}
