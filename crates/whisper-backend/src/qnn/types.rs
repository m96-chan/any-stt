//! QNN C API type definitions.
//!
//! Derived from QNN SDK 2.35.0 headers. Only the subset needed for
//! backend init, graph build, and execution is included.

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_void};

// ---- Handle types (all opaque pointers) ----

pub type Qnn_ErrorHandle_t = u32;
pub type Qnn_BackendHandle_t = *mut c_void;
pub type Qnn_DeviceHandle_t = *mut c_void;
pub type Qnn_ContextHandle_t = *mut c_void;
pub type Qnn_GraphHandle_t = *mut c_void;
pub type Qnn_ProfileHandle_t = *mut c_void;
pub type Qnn_SignalHandle_t = *mut c_void;
pub type Qnn_LogHandle_t = *mut c_void;
pub type Qnn_MemHandle_t = *mut c_void;
pub type Qnn_NotifyFn_t = *const c_void; // callback fn ptr, opaque for now

// ---- Error codes ----

pub const QNN_SUCCESS: Qnn_ErrorHandle_t = 0;

// ---- Version ----

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

// ---- Tensor (Qnn_Tensor_t) ----
// Opaque for now — we pass raw pointers to QNN.
// Full struct definition will be added when building graphs.
pub type Qnn_Tensor_t = c_void;

// ---- OpConfig (Qnn_OpConfig_t) ----
// Opaque for now.
pub type Qnn_OpConfig_t = c_void;

// ---- Config types (opaque pointers to config arrays) ----
pub type QnnBackend_Config_t = c_void;
pub type QnnContext_Config_t = c_void;
pub type QnnGraph_Config_t = c_void;

// ---- Log callback ----
pub type QnnLog_Callback_t = Option<
    unsafe extern "C" fn(
        fmt: *const c_char,
        level: u32,
        timestamp: u64,
        // varargs omitted — QNN uses printf-style
    ),
>;
pub type QnnLog_Level_t = u32;
pub const QNN_LOG_LEVEL_ERROR: QnnLog_Level_t = 1;
pub const QNN_LOG_LEVEL_WARN: QnnLog_Level_t = 2;
pub const QNN_LOG_LEVEL_INFO: QnnLog_Level_t = 3;
pub const QNN_LOG_LEVEL_DEBUG: QnnLog_Level_t = 5;

// ---- Misc opaque types needed by function signatures ----
pub type QnnProfile_Level_t = u32;
pub type QnnProfile_EventId_t = u64;
pub type QnnProfile_Config_t = c_void;
pub type QnnProfile_EventData_t = c_void;
pub type QnnProfile_ExtendedEventData_t = c_void;
pub type QnnProperty_Key_t = u32;
pub type QnnBackend_OperationName_t = c_char;
pub type Qnn_ContextBinarySize_t = u64;
pub type QnnDevice_PlatformInfo_t = c_void;
pub type QnnDevice_Infrastructure_t = c_void;
pub type QnnDevice_Config_t = c_void;
pub type QnnSignal_Config_t = c_void;
pub type QnnGraph_ExecuteEnvironment_t = c_void;
pub type Qnn_MemDescriptor_t = c_void;
pub type QnnBackend_Property_t = c_void;
pub type QnnContext_Property_t = c_void;
pub type QnnGraph_Property_t = c_void;
pub type QnnContext_SectionType_t = u32;
pub type QnnContext_Buffer_t = c_void;
pub type QnnContext_Params_t = c_void;

// ============================================================================
// QnnInterface vtable — exact layout from QnnInterface.h
// ============================================================================
//
// This struct must match QNN_INTERFACE_VER_TYPE exactly (field order matters).
// Each field is an Option<fn ptr> (nullable function pointer).

/// The implementation vtable. Matches `QnnInterface_ImplementationV2_X_t`.
#[repr(C)]
pub struct QnnInterfaceVtable {
    // QnnProperty
    pub property_has_capability:
        Option<unsafe extern "C" fn(key: QnnProperty_Key_t) -> Qnn_ErrorHandle_t>,

    // QnnBackend
    pub backend_create: Option<
        unsafe extern "C" fn(
            logger: Qnn_LogHandle_t,
            config: *const *const QnnBackend_Config_t,
            backend: *mut Qnn_BackendHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub backend_set_config: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub backend_get_api_version: Option<
        unsafe extern "C" fn(version: *mut Qnn_ApiVersion_t) -> Qnn_ErrorHandle_t,
    >,
    pub backend_get_build_id:
        Option<unsafe extern "C" fn(id: *mut *const c_char) -> Qnn_ErrorHandle_t>,
    pub backend_register_op_package: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub backend_get_supported_operations: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub backend_validate_op_config: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub backend_free:
        Option<unsafe extern "C" fn(backend: Qnn_BackendHandle_t) -> Qnn_ErrorHandle_t>,

    // QnnContext
    pub context_create: Option<
        unsafe extern "C" fn(
            backend: Qnn_BackendHandle_t,
            device: Qnn_DeviceHandle_t,
            config: *const *const QnnContext_Config_t,
            context: *mut Qnn_ContextHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub context_set_config: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_get_binary_size: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_get_binary: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_create_from_binary: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_free: Option<
        unsafe extern "C" fn(
            context: Qnn_ContextHandle_t,
            profile: Qnn_ProfileHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,

    // QnnGraph
    pub graph_create: Option<
        unsafe extern "C" fn(
            context: Qnn_ContextHandle_t,
            name: *const c_char,
            config: *const *const QnnGraph_Config_t,
            graph: *mut Qnn_GraphHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub graph_create_subgraph: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub graph_set_config: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub graph_add_node: Option<
        unsafe extern "C" fn(
            graph: Qnn_GraphHandle_t,
            op_config: *const Qnn_OpConfig_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub graph_finalize: Option<
        unsafe extern "C" fn(
            graph: Qnn_GraphHandle_t,
            profile: Qnn_ProfileHandle_t,
            signal: Qnn_SignalHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub graph_retrieve: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub graph_execute: Option<
        unsafe extern "C" fn(
            graph: Qnn_GraphHandle_t,
            inputs: *const Qnn_Tensor_t,
            num_inputs: u32,
            outputs: *mut Qnn_Tensor_t,
            num_outputs: u32,
            profile: Qnn_ProfileHandle_t,
            signal: Qnn_SignalHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub graph_execute_async: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,

    // QnnTensor
    pub tensor_create_context_tensor: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub tensor_create_graph_tensor: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,

    // QnnLog
    pub log_create: Option<
        unsafe extern "C" fn(
            callback: QnnLog_Callback_t,
            level: QnnLog_Level_t,
            logger: *mut Qnn_LogHandle_t,
        ) -> Qnn_ErrorHandle_t,
    >,
    pub log_set_log_level: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub log_free:
        Option<unsafe extern "C" fn(logger: Qnn_LogHandle_t) -> Qnn_ErrorHandle_t>,

    // QnnProfile
    pub profile_create: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub profile_set_config: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub profile_get_events: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub profile_get_sub_events: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub profile_get_event_data: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub profile_get_extended_event_data: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub profile_free: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,

    // QnnMem
    pub mem_register: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub mem_de_register: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,

    // QnnDevice
    pub device_get_platform_info: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub device_free_platform_info: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub device_get_infrastructure: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub device_create: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub device_set_config: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub device_get_info: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub device_free: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,

    // QnnSignal
    pub signal_create: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub signal_set_config: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub signal_trigger: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub signal_free: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,

    // QnnError
    pub error_get_message: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub error_get_verbose_message: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub error_free_verbose_message: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,

    // Extended (added in later versions)
    pub graph_prepare_execution_environment: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub graph_release_execution_environment: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub graph_get_property: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_validate_binary: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_create_from_binary_with_signal: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_create_from_binary_list_async: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub tensor_update_graph_tensors: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub tensor_update_context_tensors: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_get_binary_section_size: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_get_binary_section: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_apply_binary_section: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub backend_get_property: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_get_property: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_get_incremental_binary: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_release_incremental_binary: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
    pub context_finalize: Option<unsafe extern "C" fn() -> Qnn_ErrorHandle_t>,
}

// ============================================================================
// QnnInterface_t — the top-level provider struct
// ============================================================================

/// Matches `QnnInterface_t` from QnnInterface.h.
#[repr(C)]
pub struct QnnInterface_t {
    pub backend_id: u32,
    pub provider_name: *const c_char,
    pub api_version: Qnn_ApiVersion_t,
    pub vtable: QnnInterfaceVtable,
}

/// Signature of `QnnInterface_getProviders`.
///
/// Note: `providerList` is `const QnnInterface_t***` in C.
/// The function sets `*providerList` to point to an array of `*const QnnInterface_t`.
pub type QnnInterface_getProvidersFn = unsafe extern "C" fn(
    provider_list: *mut *const *const QnnInterface_t,
    num_providers: *mut u32,
) -> Qnn_ErrorHandle_t;
