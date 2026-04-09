//! NNAPI C FFI type definitions.
//!
//! Matches Android NDK NeuralNetworks.h for API level 34+ (Android 14+).

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::c_void;

// ======================================================================
// Opaque handle types
// ======================================================================

pub type ANeuralNetworksModel = c_void;
pub type ANeuralNetworksCompilation = c_void;
pub type ANeuralNetworksExecution = c_void;
pub type ANeuralNetworksDevice = c_void;
pub type ANeuralNetworksMemory = c_void;

// ======================================================================
// Result codes
// ======================================================================

pub const ANEURALNETWORKS_NO_ERROR: i32 = 0;
pub const ANEURALNETWORKS_BAD_DATA: i32 = 2;
pub const ANEURALNETWORKS_OP_FAILED: i32 = 4;

// ======================================================================
// Operand types
// ======================================================================

pub const ANEURALNETWORKS_FLOAT32: i32 = 0;
pub const ANEURALNETWORKS_INT32: i32 = 1;
pub const ANEURALNETWORKS_UINT32: i32 = 2;
pub const ANEURALNETWORKS_TENSOR_FLOAT32: i32 = 3;
pub const ANEURALNETWORKS_TENSOR_INT32: i32 = 4;
pub const ANEURALNETWORKS_TENSOR_QUANT8_ASYMM: i32 = 5;
pub const ANEURALNETWORKS_BOOL: i32 = 6;
pub const ANEURALNETWORKS_TENSOR_FLOAT16: i32 = 8;
pub const ANEURALNETWORKS_TENSOR_BOOL8: i32 = 9;

// ======================================================================
// Operation codes
// ======================================================================

pub const ANEURALNETWORKS_ADD: i32 = 0;
pub const ANEURALNETWORKS_FULLY_CONNECTED: i32 = 9;
pub const ANEURALNETWORKS_LOGISTIC: i32 = 14;
pub const ANEURALNETWORKS_MUL: i32 = 18;
pub const ANEURALNETWORKS_RESHAPE: i32 = 22;
pub const ANEURALNETWORKS_SOFTMAX: i32 = 25;
pub const ANEURALNETWORKS_TANH: i32 = 28;
pub const ANEURALNETWORKS_MEAN: i32 = 31;
pub const ANEURALNETWORKS_SUB: i32 = 36;
pub const ANEURALNETWORKS_TRANSPOSE: i32 = 37;
pub const ANEURALNETWORKS_RSQRT: i32 = 83;
pub const ANEURALNETWORKS_BATCH_MATMUL: i32 = 102;

// ======================================================================
// Fuse codes (for ADD, MUL, SUB)
// ======================================================================

pub const ANEURALNETWORKS_FUSED_NONE: i32 = 0;
pub const ANEURALNETWORKS_FUSED_RELU: i32 = 1;

// ======================================================================
// Device types
// ======================================================================

pub const ANEURALNETWORKS_DEVICE_UNKNOWN: i32 = 0;
pub const ANEURALNETWORKS_DEVICE_OTHER: i32 = 1;
pub const ANEURALNETWORKS_DEVICE_CPU: i32 = 2;
pub const ANEURALNETWORKS_DEVICE_GPU: i32 = 3;
pub const ANEURALNETWORKS_DEVICE_ACCELERATOR: i32 = 4; // NPU

// ======================================================================
// Preference codes (for compilation)
// ======================================================================

pub const ANEURALNETWORKS_PREFER_LOW_POWER: i32 = 0;
pub const ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER: i32 = 1;
pub const ANEURALNETWORKS_PREFER_SUSTAINED_SPEED: i32 = 2;

// ======================================================================
// Operand type struct — repr(C) matching NeuralNetworks.h
// ======================================================================

#[repr(C)]
pub struct ANeuralNetworksOperandType {
    pub type_: i32,
    pub dimension_count: u32,
    pub dimensions: *const u32,
    pub scale: f32,
    pub zero_point: i32,
}

impl ANeuralNetworksOperandType {
    pub fn tensor_f32(dims: &[u32]) -> Self {
        Self {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: dims.len() as u32,
            dimensions: dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        }
    }

    pub fn tensor_i32(dims: &[u32]) -> Self {
        Self {
            type_: ANEURALNETWORKS_TENSOR_INT32,
            dimension_count: dims.len() as u32,
            dimensions: dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        }
    }

    pub fn scalar_f32() -> Self {
        Self {
            type_: ANEURALNETWORKS_FLOAT32,
            dimension_count: 0,
            dimensions: std::ptr::null(),
            scale: 0.0,
            zero_point: 0,
        }
    }

    pub fn scalar_i32() -> Self {
        Self {
            type_: ANEURALNETWORKS_INT32,
            dimension_count: 0,
            dimensions: std::ptr::null(),
            scale: 0.0,
            zero_point: 0,
        }
    }

    pub fn scalar_bool() -> Self {
        Self {
            type_: ANEURALNETWORKS_BOOL,
            dimension_count: 0,
            dimensions: std::ptr::null(),
            scale: 0.0,
            zero_point: 0,
        }
    }
}

// ======================================================================
// Function pointer types for dlopen
// ======================================================================

pub type FnGetDeviceCount = unsafe extern "C" fn(num_devices: *mut u32) -> i32;

pub type FnGetDevice =
    unsafe extern "C" fn(dev_index: u32, device: *mut *const ANeuralNetworksDevice) -> i32;

pub type FnDeviceGetName =
    unsafe extern "C" fn(device: *const ANeuralNetworksDevice, name: *mut *const i8) -> i32;

pub type FnDeviceGetType =
    unsafe extern "C" fn(device: *const ANeuralNetworksDevice, type_: *mut i32) -> i32;

pub type FnModelCreate = unsafe extern "C" fn(model: *mut *mut ANeuralNetworksModel) -> i32;

pub type FnModelFree = unsafe extern "C" fn(model: *mut ANeuralNetworksModel);

pub type FnModelAddOperand = unsafe extern "C" fn(
    model: *mut ANeuralNetworksModel,
    type_: *const ANeuralNetworksOperandType,
) -> i32;

pub type FnModelSetOperandValue = unsafe extern "C" fn(
    model: *mut ANeuralNetworksModel,
    index: i32,
    buffer: *const c_void,
    length: usize,
) -> i32;

pub type FnModelAddOperation = unsafe extern "C" fn(
    model: *mut ANeuralNetworksModel,
    type_: i32,
    input_count: u32,
    inputs: *const u32,
    output_count: u32,
    outputs: *const u32,
) -> i32;

pub type FnModelIdentifyInputsAndOutputs = unsafe extern "C" fn(
    model: *mut ANeuralNetworksModel,
    input_count: u32,
    inputs: *const u32,
    output_count: u32,
    outputs: *const u32,
) -> i32;

pub type FnModelFinish = unsafe extern "C" fn(model: *mut ANeuralNetworksModel) -> i32;

pub type FnCompilationCreateForDevices = unsafe extern "C" fn(
    model: *const ANeuralNetworksModel,
    devices: *const *const ANeuralNetworksDevice,
    num_devices: u32,
    compilation: *mut *mut ANeuralNetworksCompilation,
) -> i32;

pub type FnCompilationSetPreference = unsafe extern "C" fn(
    compilation: *mut ANeuralNetworksCompilation,
    preference: i32,
) -> i32;

pub type FnCompilationSetCaching = unsafe extern "C" fn(
    compilation: *mut ANeuralNetworksCompilation,
    cache_dir: *const i8,
    token: *const u8,
) -> i32;

pub type FnCompilationFinish =
    unsafe extern "C" fn(compilation: *mut ANeuralNetworksCompilation) -> i32;

pub type FnCompilationFree = unsafe extern "C" fn(compilation: *mut ANeuralNetworksCompilation);

pub type FnExecutionCreate = unsafe extern "C" fn(
    compilation: *const ANeuralNetworksCompilation,
    execution: *mut *mut ANeuralNetworksExecution,
) -> i32;

pub type FnExecutionSetInput = unsafe extern "C" fn(
    execution: *mut ANeuralNetworksExecution,
    index: i32,
    type_: *const ANeuralNetworksOperandType,
    buffer: *const c_void,
    length: usize,
) -> i32;

pub type FnExecutionSetOutput = unsafe extern "C" fn(
    execution: *mut ANeuralNetworksExecution,
    index: i32,
    type_: *const ANeuralNetworksOperandType,
    buffer: *mut c_void,
    length: usize,
) -> i32;

pub type FnExecutionCompute =
    unsafe extern "C" fn(execution: *mut ANeuralNetworksExecution) -> i32;

pub type FnExecutionFree = unsafe extern "C" fn(execution: *mut ANeuralNetworksExecution);
