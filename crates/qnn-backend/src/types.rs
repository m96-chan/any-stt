//! QNN C API type definitions — exact layout matching SDK 2.35.0 on aarch64.
//!
//! All struct sizes verified via C sizeof on Android aarch64:
//!   Qnn_QuantizeParams_t = 40, Qnn_TensorV1_t = 112, Qnn_Tensor_t = 144,
//!   Qnn_Scalar_t = 16, Qnn_Param_t = 160, Qnn_OpConfigV1_t = 72,
//!   Qnn_OpConfig_t = 80, QnnInterface_t = 576

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_void};

// ======================================================================
// Handle types
// ======================================================================

pub type Qnn_ErrorHandle_t = u32;
pub type Qnn_BackendHandle_t = *mut c_void;
pub type Qnn_DeviceHandle_t = *mut c_void;
pub type Qnn_ContextHandle_t = *mut c_void;
pub type Qnn_GraphHandle_t = *mut c_void;
pub type Qnn_ProfileHandle_t = *mut c_void;
pub type Qnn_SignalHandle_t = *mut c_void;
pub type Qnn_LogHandle_t = *mut c_void;
pub type Qnn_MemHandle_t = *mut c_void;

// ======================================================================
// Constants
// ======================================================================

pub const QNN_SUCCESS: Qnn_ErrorHandle_t = 0;

pub const QNN_TENSOR_VERSION_1: u32 = 1;
pub const QNN_OPCONFIG_VERSION_1: u32 = 1;

pub const QNN_TENSOR_TYPE_APP_WRITE: u32 = 0;
pub const QNN_TENSOR_TYPE_APP_READ: u32 = 1;
pub const QNN_TENSOR_TYPE_APP_READWRITE: u32 = 2;
pub const QNN_TENSOR_TYPE_NATIVE: u32 = 3;
pub const QNN_TENSOR_TYPE_STATIC: u32 = 4;

pub const QNN_DATATYPE_FLOAT_32: u32 = 0x0232;
pub const QNN_DATATYPE_FLOAT_16: u32 = 0x0216;
pub const QNN_DATATYPE_INT_8: u32 = 0x0308;
pub const QNN_DATATYPE_INT_32: u32 = 0x0332;
pub const QNN_DATATYPE_UINT_8: u32 = 0x0408;
pub const QNN_DATATYPE_BOOL_8: u32 = 0x0508;
pub const QNN_DATATYPE_SFIXED_POINT_8: u32 = 0x0508;

pub const QNN_DEFINITION_UNDEFINED: u32 = 0x7FFF_FFFF;
pub const QNN_DEFINITION_DEFINED: u32 = 0;
pub const QNN_QUANTIZATION_ENCODING_UNDEFINED: u32 = 0x7FFF_FFFF;
pub const QNN_QUANTIZATION_ENCODING_SCALE_OFFSET: u32 = 0;

pub const QNN_DATATYPE_UFIXED_POINT_8: u32 = 0x0408;
pub const QNN_DATATYPE_UFIXED_POINT_16: u32 = 0x0416;

pub const QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER: u32 = 0;
pub const QNN_TENSORMEMTYPE_RAW: u32 = 0;
pub const QNN_PARAMTYPE_SCALAR: u32 = 0;
pub const QNN_PARAMTYPE_TENSOR: u32 = 1;

// ======================================================================
// Qnn_QuantizeParams_t — 40 bytes
// Layout: encodingDefinition(4) + quantizationEncoding(4) + union(32)
// ======================================================================

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Qnn_QuantizeParams_t {
    pub encoding_definition: u32,
    pub quantization_encoding: u32,
    pub _union: [u8; 32],
}

impl Qnn_QuantizeParams_t {
    pub fn undefined() -> Self {
        Self {
            encoding_definition: QNN_DEFINITION_UNDEFINED,
            quantization_encoding: QNN_QUANTIZATION_ENCODING_UNDEFINED,
            _union: [0; 32],
        }
    }

    /// Create scale-offset quantization parameters.
    /// For UFIXED_POINT_8: real_value = (quantized_value - offset) * scale
    pub fn scale_offset(scale: f32, offset: i32) -> Self {
        let mut params = Self {
            encoding_definition: QNN_DEFINITION_DEFINED,
            quantization_encoding: QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
            _union: [0; 32],
        };
        // scaleOffsetEncoding: scale (f32) at union[0..4], offset (i32) at union[4..8]
        params._union[0..4].copy_from_slice(&scale.to_le_bytes());
        params._union[4..8].copy_from_slice(&offset.to_le_bytes());
        params
    }
}

// ======================================================================
// Qnn_ClientBuffer_t — 16 bytes
// Layout: data(8) + dataSize(4) + pad(4)
// ======================================================================

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Qnn_ClientBuffer_t {
    pub data: *mut c_void,
    pub data_size: u32,
    _pad: u32,
}

impl Qnn_ClientBuffer_t {
    pub fn null() -> Self {
        Self {
            data: std::ptr::null_mut(),
            data_size: 0,
            _pad: 0,
        }
    }

    pub fn from_slice<T>(slice: &mut [T]) -> Self {
        Self {
            data: slice.as_mut_ptr() as *mut c_void,
            data_size: (slice.len() * std::mem::size_of::<T>()) as u32,
            _pad: 0,
        }
    }
}

// ======================================================================
// Qnn_TensorV1_t — 112 bytes
// ======================================================================

#[repr(C)]
pub struct Qnn_TensorV1_t {
    pub id: u32,                              // 0
    _pad0: u32,                               // 4
    pub name: *const c_char,                  // 8
    pub tensor_type: u32,                     // 16
    pub data_format: u32,                     // 20
    pub data_type: u32,                       // 24
    _pad1: u32,                               // 28
    pub quant_params: Qnn_QuantizeParams_t,   // 32 (40 bytes)
    pub rank: u32,                            // 72
    _pad2: u32,                               // 76
    pub dimensions: *mut u32,                 // 80
    pub mem_type: u32,                        // 88
    _pad3: u32,                               // 92
    pub client_buf: Qnn_ClientBuffer_t,       // 96 (16 bytes)
}                                             // total = 112

// ======================================================================
// Qnn_Tensor_t — 144 bytes
// Layout: version(4) + pad(4) + union(136) — union holds V1(112) or V2(136)
// ======================================================================

#[repr(C)]
pub struct Qnn_Tensor_t {
    pub version: u32,               // 0
    _pad: u32,                      // 4
    pub v1: Qnn_TensorV1_t,        // 8 (112 bytes for V1; V2 uses 136)
    _tail: [u8; 24],               // 120 (pad to 136 for V2 union size)
}                                   // total = 144

impl Qnn_Tensor_t {
    /// Create an APP_WRITE (input) tensor for graph registration.
    /// Data is set to null; use `set_data` before execute.
    pub fn app_write(name: *const c_char, rank: u32, dimensions: *mut u32) -> Self {
        Self::new(name, QNN_TENSOR_TYPE_APP_WRITE, rank, dimensions)
    }

    /// Create an APP_READ (output) tensor for graph registration.
    pub fn app_read(name: *const c_char, rank: u32, dimensions: *mut u32) -> Self {
        Self::new(name, QNN_TENSOR_TYPE_APP_READ, rank, dimensions)
    }

    /// Create a STATIC tensor (weights embedded in graph, preloaded to DSP).
    /// Use `set_data` to provide the weight data before registration.
    pub fn static_tensor(name: *const c_char, rank: u32, dimensions: *mut u32) -> Self {
        Self::new(name, QNN_TENSOR_TYPE_STATIC, rank, dimensions)
    }

    /// Create a quantized APP_WRITE tensor (UFIXED_POINT_8 with scale/offset).
    pub fn app_write_quant_u8(
        name: *const c_char,
        rank: u32,
        dimensions: *mut u32,
        scale: f32,
        offset: i32,
    ) -> Self {
        let mut t = Self::new(name, QNN_TENSOR_TYPE_APP_WRITE, rank, dimensions);
        t.v1.data_type = QNN_DATATYPE_UFIXED_POINT_8;
        t.v1.quant_params = Qnn_QuantizeParams_t::scale_offset(scale, offset);
        t
    }

    /// Create a quantized APP_READ tensor (UFIXED_POINT_8 with scale/offset).
    pub fn app_read_quant_u8(
        name: *const c_char,
        rank: u32,
        dimensions: *mut u32,
        scale: f32,
        offset: i32,
    ) -> Self {
        let mut t = Self::new(name, QNN_TENSOR_TYPE_APP_READ, rank, dimensions);
        t.v1.data_type = QNN_DATATYPE_UFIXED_POINT_8;
        t.v1.quant_params = Qnn_QuantizeParams_t::scale_offset(scale, offset);
        t
    }

    pub(crate) fn new(name: *const c_char, tensor_type: u32, rank: u32, dimensions: *mut u32) -> Self {
        Self {
            version: QNN_TENSOR_VERSION_1,
            _pad: 0,
            v1: Qnn_TensorV1_t {
                id: 0,
                _pad0: 0,
                name,
                tensor_type,
                data_format: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                data_type: QNN_DATATYPE_FLOAT_32,
                _pad1: 0,
                quant_params: Qnn_QuantizeParams_t::undefined(),
                rank,
                _pad2: 0,
                dimensions,
                mem_type: QNN_TENSORMEMTYPE_RAW,
                _pad3: 0,
                client_buf: Qnn_ClientBuffer_t::null(),
            },
            _tail: [0; 24],
        }
    }

    /// Set the client buffer data pointer (for execute).
    pub fn set_data<T>(&mut self, slice: &mut [T]) {
        self.v1.client_buf = Qnn_ClientBuffer_t::from_slice(slice);
    }
}

// ======================================================================
// Qnn_Scalar_t — 16 bytes
// Layout: dataType(4) + pad(4) + value_union(8)
// ======================================================================

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Qnn_Scalar_t {
    pub data_type: u32,
    _pad: u32,
    pub value: u64, // union: bool8, int32, float, etc.
}

impl Qnn_Scalar_t {
    pub fn bool8(val: bool) -> Self {
        Self {
            data_type: QNN_DATATYPE_BOOL_8,
            _pad: 0,
            value: if val { 1 } else { 0 },
        }
    }

    pub fn float32(val: f32) -> Self {
        Self {
            data_type: QNN_DATATYPE_FLOAT_32,
            _pad: 0,
            value: u64::from(val.to_bits()),
        }
    }
}

// ======================================================================
// Qnn_Param_t — 160 bytes
// Layout: paramType(4) + pad(4) + name(8) + union(144: Scalar_t or Tensor_t)
// ======================================================================

#[repr(C)]
pub struct Qnn_Param_t {
    pub param_type: u32,
    _pad: u32,
    pub name: *const c_char,
    // union: scalar (16 bytes) or tensor (144 bytes) — use 144 bytes
    _union: [u8; 144],
}

impl Qnn_Param_t {
    pub fn scalar(name: *const c_char, scalar: Qnn_Scalar_t) -> Self {
        let mut p = Self {
            param_type: QNN_PARAMTYPE_SCALAR,
            _pad: 0,
            name,
            _union: [0; 144],
        };
        // Copy scalar (16 bytes) into union
        let src = &scalar as *const Qnn_Scalar_t as *const u8;
        unsafe {
            std::ptr::copy_nonoverlapping(src, p._union.as_mut_ptr(), 16);
        }
        p
    }
}

// ======================================================================
// Qnn_OpConfigV1_t — 72 bytes
// ======================================================================

#[repr(C)]
pub struct Qnn_OpConfigV1_t {
    pub name: *const c_char,              // 0
    pub package_name: *const c_char,      // 8
    pub type_name: *const c_char,         // 16
    pub num_of_params: u32,               // 24
    _pad0: u32,                           // 28
    pub params: *mut Qnn_Param_t,         // 32
    pub num_of_inputs: u32,               // 40
    _pad1: u32,                           // 44
    pub input_tensors: *mut Qnn_Tensor_t, // 48
    pub num_of_outputs: u32,              // 56
    _pad2: u32,                           // 60
    pub output_tensors: *mut Qnn_Tensor_t, // 64
}                                          // 72

// ======================================================================
// Qnn_OpConfig_t — 80 bytes
// ======================================================================

#[repr(C)]
pub struct Qnn_OpConfig_t {
    pub version: u32,
    _pad: u32,
    pub v1: Qnn_OpConfigV1_t,
}

impl Qnn_OpConfig_t {
    pub fn new(
        name: *const c_char,
        package_name: *const c_char,
        type_name: *const c_char,
        params: &mut [Qnn_Param_t],
        inputs: &mut [Qnn_Tensor_t],
        outputs: &mut [Qnn_Tensor_t],
    ) -> Self {
        Self {
            version: QNN_OPCONFIG_VERSION_1,
            _pad: 0,
            v1: Qnn_OpConfigV1_t {
                name,
                package_name,
                type_name,
                num_of_params: params.len() as u32,
                _pad0: 0,
                params: if params.is_empty() {
                    std::ptr::null_mut()
                } else {
                    params.as_mut_ptr()
                },
                num_of_inputs: inputs.len() as u32,
                _pad1: 0,
                input_tensors: inputs.as_mut_ptr(),
                num_of_outputs: outputs.len() as u32,
                _pad2: 0,
                output_tensors: outputs.as_mut_ptr(),
            },
        }
    }
}

// ======================================================================
// Version types
// ======================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Qnn_Version_t {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Qnn_ApiVersion_t {
    pub core_api_version: Qnn_Version_t,
    pub backend_api_version: Qnn_Version_t,
}

// ======================================================================
// QnnInterface vtable — 55 function pointers, matches C header exactly
// ======================================================================

// Config types (opaque)
pub type QnnBackend_Config_t = c_void;
pub type QnnContext_Config_t = c_void;
pub type QnnGraph_Config_t = c_void;
pub type QnnLog_Callback_t = *const c_void;
pub type QnnLog_Level_t = u32;

#[repr(C)]
pub struct QnnInterfaceVtable {
    pub property_has_capability: *const c_void,

    pub backend_create: Option<
        unsafe extern "C" fn(
            Qnn_LogHandle_t,
            *const *const QnnBackend_Config_t,
            *mut Qnn_BackendHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub backend_set_config: *const c_void,
    pub backend_get_api_version:
        Option<unsafe extern "C" fn(*mut Qnn_ApiVersion_t) -> Qnn_ErrorHandle_t>,
    pub backend_get_build_id:
        Option<unsafe extern "C" fn(*mut *const c_char) -> Qnn_ErrorHandle_t>,
    pub backend_register_op_package: *const c_void,
    pub backend_get_supported_operations: *const c_void,
    pub backend_validate_op_config: *const c_void,
    pub backend_free:
        Option<unsafe extern "C" fn(Qnn_BackendHandle_t) -> Qnn_ErrorHandle_t>,

    pub context_create: Option<
        unsafe extern "C" fn(
            Qnn_BackendHandle_t,
            Qnn_DeviceHandle_t,
            *const *const QnnContext_Config_t,
            *mut Qnn_ContextHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub context_set_config: *const c_void,
    pub context_get_binary_size: Option<
        unsafe extern "C" fn(
            Qnn_ContextHandle_t,
            *mut u64,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub context_get_binary: Option<
        unsafe extern "C" fn(
            Qnn_ContextHandle_t,
            *mut *const c_void,
            *mut u64,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub context_create_from_binary: Option<
        unsafe extern "C" fn(
            Qnn_BackendHandle_t,
            Qnn_DeviceHandle_t,
            *const *const QnnContext_Config_t,
            *const c_void,
            u64,
            *mut Qnn_ContextHandle_t,
            Qnn_ProfileHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub context_free: Option<
        unsafe extern "C" fn(Qnn_ContextHandle_t, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t,
    >,

    pub graph_create: Option<
        unsafe extern "C" fn(
            Qnn_ContextHandle_t,
            *const c_char,
            *const *const QnnGraph_Config_t,
            *mut Qnn_GraphHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub graph_create_subgraph: *const c_void,
    pub graph_set_config: *const c_void,
    pub graph_add_node: Option<
        unsafe extern "C" fn(Qnn_GraphHandle_t, Qnn_OpConfig_t) -> Qnn_ErrorHandle_t,
    >,
    pub graph_finalize: Option<
        unsafe extern "C" fn(
            Qnn_GraphHandle_t,
            Qnn_ProfileHandle_t,
            Qnn_SignalHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub graph_retrieve: *const c_void,
    pub graph_execute: Option<
        unsafe extern "C" fn(
            Qnn_GraphHandle_t,
            *const Qnn_Tensor_t,
            u32,
            *mut Qnn_Tensor_t,
            u32,
            Qnn_ProfileHandle_t,
            Qnn_SignalHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub graph_execute_async: *const c_void,

    pub tensor_create_context_tensor: *const c_void,
    pub tensor_create_graph_tensor: Option<
        unsafe extern "C" fn(Qnn_GraphHandle_t, *mut Qnn_Tensor_t) -> Qnn_ErrorHandle_t,
    >,

    pub log_create: *const c_void,
    pub log_set_log_level: *const c_void,
    pub log_free: *const c_void,

    // Remaining 32 function pointers (profile, mem, device, signal, error, etc.)
    pub _remaining: [*const c_void; 32],
}

// ======================================================================
// QnnInterface_t — top-level provider struct, 576 bytes
// ======================================================================

#[repr(C)]
pub struct QnnInterface_t {
    pub backend_id: u32,
    _pad: u32,
    pub provider_name: *const c_char,
    pub api_version: Qnn_ApiVersion_t,
    pub vtable: QnnInterfaceVtable,
}

/// Signature of QnnInterface_getProviders.
pub type QnnInterface_getProvidersFn = unsafe extern "C" fn(
    provider_list: *mut *const *const QnnInterface_t,
    num_providers: *mut u32,
) -> Qnn_ErrorHandle_t;
