//! Runtime QNN library loader via dlopen.

use std::path::Path;
use std::ptr;

use super::types::*;

/// A loaded QNN backend with resolved interface vtable.
pub struct QnnLibrary {
    _lib: libloading::Library,
    pub iface: &'static QnnInterface_t,
}

impl QnnLibrary {
    /// Load a QNN backend library (e.g., `libQnnHtp.so`).
    pub fn load(lib_path: &Path) -> Result<Self, String> {
        let lib = unsafe { libloading::Library::new(lib_path) }
            .map_err(|e| format!("dlopen {}: {e}", lib_path.display()))?;

        let get_providers: libloading::Symbol<QnnInterface_getProvidersFn> = unsafe {
            lib.get(b"QnnInterface_getProviders\0")
                .map_err(|e| format!("symbol QnnInterface_getProviders: {e}"))?
        };

        let mut provider_list: *const *const QnnInterface_t = ptr::null();
        let mut num_providers: u32 = 0;

        let err = unsafe { get_providers(&mut provider_list, &mut num_providers) };
        if err != QNN_SUCCESS {
            return Err(format!("QnnInterface_getProviders: error {err}"));
        }
        if num_providers == 0 || provider_list.is_null() {
            return Err("no QNN providers".into());
        }

        // Get the first provider.
        let iface_ptr = unsafe { *provider_list };
        if iface_ptr.is_null() {
            return Err("null QNN provider".into());
        }
        let iface: &'static QnnInterface_t = unsafe { &*iface_ptr };

        let ver = &iface.api_version.core_api_version;
        eprintln!(
            "QNN loaded: backend_id={}, API v{}.{}.{}",
            iface.backend_id, ver.major, ver.minor, ver.patch
        );

        Ok(Self { _lib: lib, iface })
    }

    /// Try common paths to find libQnnHtp.so.
    pub fn load_htp() -> Result<Self, String> {
        let candidates: Vec<String> = [
            std::env::var("QNN_SDK_ROOT")
                .ok()
                .map(|r| format!("{r}/lib/aarch64-android/libQnnHtp.so")),
            std::env::var("QNN_SDK_ROOT")
                .ok()
                .map(|r| format!("{r}/lib/aarch64-linux/libQnnHtp.so")),
            std::env::var("QNN_LIB_DIR")
                .ok()
                .map(|d| format!("{d}/libQnnHtp.so")),
            Some("/usr/lib/libQnnHtp.so".into()),
            Some("/usr/local/lib/libQnnHtp.so".into()),
            Some("/vendor/lib64/libQnnHtp.so".into()),
            Some("./libQnnHtp.so".into()),
        ]
        .into_iter()
        .flatten()
        .collect();

        let mut last_err = String::from("no candidate paths");
        for path in &candidates {
            let p = Path::new(path);
            if p.exists() {
                match Self::load(p) {
                    Ok(lib) => return Ok(lib),
                    Err(e) => last_err = e,
                }
            }
        }

        Err(format!(
            "QNN HTP not found. Set QNN_SDK_ROOT or QNN_LIB_DIR. Last: {last_err}"
        ))
    }

    /// Convenience: vtable access.
    pub fn vt(&self) -> &QnnInterfaceVtable {
        &self.iface.vtable
    }
}

/// Check if QNN HTP is available (non-blocking).
pub fn is_qnn_available() -> bool {
    QnnLibrary::load_htp().is_ok()
}
