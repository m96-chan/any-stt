//! Runtime NNAPI library loader via dlopen.
//!
//! Loads `libneuralnetworks.so` and resolves all required function pointers.
//! Also provides NPU device discovery.

use std::ffi::CStr;
use std::ptr;

use crate::types::*;

/// Loaded NNAPI library with resolved function pointers.
pub struct NnapiLib {
    _lib: libloading::Library,

    // Device enumeration
    pub get_device_count: FnGetDeviceCount,
    pub get_device: FnGetDevice,
    pub device_get_name: FnDeviceGetName,
    pub device_get_type: FnDeviceGetType,

    // Model
    pub model_create: FnModelCreate,
    pub model_free: FnModelFree,
    pub model_add_operand: FnModelAddOperand,
    pub model_set_operand_value: FnModelSetOperandValue,
    pub model_add_operation: FnModelAddOperation,
    pub model_identify_inputs_and_outputs: FnModelIdentifyInputsAndOutputs,
    pub model_finish: FnModelFinish,

    // Compilation
    pub compilation_create_for_devices: FnCompilationCreateForDevices,
    pub compilation_set_preference: FnCompilationSetPreference,
    pub compilation_set_caching: Option<FnCompilationSetCaching>,
    pub compilation_finish: FnCompilationFinish,
    pub compilation_free: FnCompilationFree,

    // Execution
    pub execution_create: FnExecutionCreate,
    pub execution_set_input: FnExecutionSetInput,
    pub execution_set_output: FnExecutionSetOutput,
    pub execution_compute: FnExecutionCompute,
    pub execution_free: FnExecutionFree,
}

macro_rules! load_sym {
    ($lib:expr, $name:literal) => {
        *$lib
            .get($name)
            .map_err(|e| format!("symbol {}: {e}", std::str::from_utf8($name).unwrap_or("?")))?
    };
}

macro_rules! load_sym_opt {
    ($lib:expr, $name:literal) => {
        $lib.get($name).ok().map(|s| *s)
    };
}

impl NnapiLib {
    /// Load NNAPI from the system library.
    pub fn load() -> Result<Self, String> {
        let lib = unsafe { libloading::Library::new("libneuralnetworks.so") }
            .map_err(|e| format!("dlopen libneuralnetworks.so: {e}"))?;

        unsafe {
            Ok(Self {
                get_device_count: load_sym!(lib, b"ANeuralNetworks_getDeviceCount\0"),
                get_device: load_sym!(lib, b"ANeuralNetworks_getDevice\0"),
                device_get_name: load_sym!(lib, b"ANeuralNetworksDevice_getName\0"),
                device_get_type: load_sym!(lib, b"ANeuralNetworksDevice_getType\0"),
                model_create: load_sym!(lib, b"ANeuralNetworksModel_create\0"),
                model_free: load_sym!(lib, b"ANeuralNetworksModel_free\0"),
                model_add_operand: load_sym!(lib, b"ANeuralNetworksModel_addOperand\0"),
                model_set_operand_value: load_sym!(
                    lib,
                    b"ANeuralNetworksModel_setOperandValue\0"
                ),
                model_add_operation: load_sym!(lib, b"ANeuralNetworksModel_addOperation\0"),
                model_identify_inputs_and_outputs: load_sym!(
                    lib,
                    b"ANeuralNetworksModel_identifyInputsAndOutputs\0"
                ),
                model_finish: load_sym!(lib, b"ANeuralNetworksModel_finish\0"),
                compilation_create_for_devices: load_sym!(
                    lib,
                    b"ANeuralNetworksCompilation_createForDevices\0"
                ),
                compilation_set_preference: load_sym!(
                    lib,
                    b"ANeuralNetworksCompilation_setPreference\0"
                ),
                compilation_set_caching: load_sym_opt!(
                    lib,
                    b"ANeuralNetworksCompilation_setCaching\0"
                ),
                compilation_finish: load_sym!(lib, b"ANeuralNetworksCompilation_finish\0"),
                compilation_free: load_sym!(lib, b"ANeuralNetworksCompilation_free\0"),
                execution_create: load_sym!(lib, b"ANeuralNetworksExecution_create\0"),
                execution_set_input: load_sym!(lib, b"ANeuralNetworksExecution_setInput\0"),
                execution_set_output: load_sym!(lib, b"ANeuralNetworksExecution_setOutput\0"),
                execution_compute: load_sym!(lib, b"ANeuralNetworksExecution_compute\0"),
                execution_free: load_sym!(lib, b"ANeuralNetworksExecution_free\0"),
                _lib: lib,
            })
        }
    }

    /// Find the best ACCELERATOR (NPU) device.
    ///
    /// Prefers MDLA (Deep Learning Accelerator) > neuron > dsp for MediaTek,
    /// then falls back to any non-CPU accelerator.
    pub fn find_npu_device(&self) -> Result<*const ANeuralNetworksDevice, String> {
        let devices = self.enumerate_devices()?;
        // Priority: neuron (NeuroPilot, broadest FP32 op support) > mdla > any accelerator
        for pref in &["neuron", "mdla"] {
            for &(dev, ref name, dev_type) in &devices {
                if dev_type == ANEURALNETWORKS_DEVICE_ACCELERATOR && name.contains(pref) {
                    eprintln!("NNAPI: selected device: {name}");
                    return Ok(dev);
                }
            }
        }
        // Fallback: any accelerator
        for &(dev, ref name, dev_type) in &devices {
            if dev_type >= ANEURALNETWORKS_DEVICE_GPU {
                eprintln!("NNAPI: selected device: {name}");
                return Ok(dev);
            }
        }
        Err("no NPU/accelerator device found via NNAPI".into())
    }

    /// Find the CPU reference device.
    pub fn find_cpu_device(&self) -> Result<*const ANeuralNetworksDevice, String> {
        let devices = self.enumerate_devices()?;
        for &(dev, ref name, dev_type) in &devices {
            if dev_type == ANEURALNETWORKS_DEVICE_CPU || name.contains("reference") {
                eprintln!("NNAPI: selected CPU device: {name}");
                return Ok(dev);
            }
        }
        Err("no CPU device found".into())
    }

    /// Find a device by name substring.
    pub fn find_device_by_name(&self, pattern: &str) -> Result<*const ANeuralNetworksDevice, String> {
        let devices = self.enumerate_devices()?;
        for &(dev, ref name, _) in &devices {
            if name.contains(pattern) {
                eprintln!("NNAPI: selected device: {name}");
                return Ok(dev);
            }
        }
        Err(format!("no device matching '{pattern}' found"))
    }

    /// Enumerate all NNAPI devices with their names and types.
    pub fn enumerate_devices(&self) -> Result<Vec<(*const ANeuralNetworksDevice, String, i32)>, String> {
        let mut count: u32 = 0;
        let err = unsafe { (self.get_device_count)(&mut count) };
        if err != ANEURALNETWORKS_NO_ERROR {
            return Err(format!("getDeviceCount: error {err}"));
        }
        eprintln!("NNAPI: {count} devices found");

        let mut devices = Vec::new();
        for i in 0..count {
            let mut device: *const ANeuralNetworksDevice = ptr::null();
            let err = unsafe { (self.get_device)(i, &mut device) };
            if err != ANEURALNETWORKS_NO_ERROR { continue; }

            let mut name_ptr: *const i8 = ptr::null();
            let mut dev_type: i32 = 0;
            unsafe {
                (self.device_get_name)(device, &mut name_ptr);
                (self.device_get_type)(device, &mut dev_type);
            }
            let name = if name_ptr.is_null() {
                "unknown".to_string()
            } else {
                unsafe { CStr::from_ptr(name_ptr as *const _) }.to_str().unwrap_or("?").to_string()
            };
            let type_str = match dev_type {
                ANEURALNETWORKS_DEVICE_CPU => "CPU",
                ANEURALNETWORKS_DEVICE_GPU => "GPU",
                ANEURALNETWORKS_DEVICE_ACCELERATOR => "NPU",
                _ => "OTHER",
            };
            eprintln!("  device[{i}]: {name} ({type_str})");
            devices.push((device, name, dev_type));
        }
        Ok(devices)
    }
}

/// Check if NNAPI with an NPU device is available.
pub fn is_nnapi_available() -> bool {
    NnapiLib::load()
        .and_then(|lib| lib.find_npu_device().map(|_| ()))
        .is_ok()
}

/// Probe NNAPI with a minimal model to verify basic functionality.
pub fn probe_nnapi(lib: &NnapiLib) -> Result<String, String> {
    use crate::context::NnapiModelBuilder;
    use crate::types::*;

    let device = lib.find_npu_device()?;

    // Test 1: Create model and add a simple scalar operand
    let mut b = NnapiModelBuilder::new(lib)?;
    eprintln!("  probe: model created OK");

    // Try scalar i32
    let s = b.add_scalar_i32();
    eprintln!("  probe: add_scalar_i32 = {:?}", s);

    // Try 1D tensor
    let t1 = b.add_tensor_f32(&[4]);
    eprintln!("  probe: add_tensor_f32([4]) = {:?}", t1);

    // Try 2D tensor
    let t2 = b.add_tensor_f32(&[2, 3]);
    eprintln!("  probe: add_tensor_f32([2,3]) = {:?}", t2);

    // Try 3D tensor
    let t3 = b.add_tensor_f32(&[1, 4, 3]);
    eprintln!("  probe: add_tensor_f32([1,4,3]) = {:?}", t3);

    // Try a simple ADD op if all succeeded
    if let (Ok(a), Ok(b_idx), Ok(fuse)) = (t1.as_ref(), t2.as_ref(), s.as_ref()) {
        // Set fuse code
        let _ = b.set_scalar_i32(*fuse, ANEURALNETWORKS_FUSED_NONE);
        let out = b.add_tensor_f32(&[4]);
        if let Ok(out_idx) = out {
            let add_result = b.add_op(ANEURALNETWORKS_ADD, &[*a, *a, *fuse], &[out_idx]);
            eprintln!("  probe: ADD op = {:?}", add_result);
        }
    }

    Ok("probe complete".into())
}
