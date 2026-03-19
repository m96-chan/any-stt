//! Minimal QNN dlopen test — just loads the library and gets providers.
//! No vtable access, no context creation.

#[test]
fn qnn_dlopen_only() {
    let lib_dir = std::env::var("QNN_LIB_DIR").unwrap_or_default();
    let path = format!("{lib_dir}/libQnnHtp.so");
    let p = std::path::Path::new(&path);

    if !p.exists() {
        eprintln!("SKIPPED: {path} not found");
        return;
    }

    eprintln!("Loading: {path}");

    let lib = unsafe { libloading::Library::new(p) };
    match lib {
        Ok(lib) => {
            eprintln!("dlopen OK");

            // Just resolve the symbol, don't call it yet
            let sym: Result<
                libloading::Symbol<
                    unsafe extern "C" fn(
                        *mut *const *const std::os::raw::c_void,
                        *mut u32,
                    ) -> u32,
                >,
                _,
            > = unsafe { lib.get(b"QnnInterface_getProviders\0") };

            match sym {
                Ok(_) => eprintln!("QnnInterface_getProviders symbol found"),
                Err(e) => eprintln!("Symbol not found: {e}"),
            }

            // Now call it with raw pointers
            let get_providers: libloading::Symbol<
                unsafe extern "C" fn(
                    *mut *const *const std::os::raw::c_void,
                    *mut u32,
                ) -> u32,
            > = unsafe { lib.get(b"QnnInterface_getProviders\0").unwrap() };

            let mut providers: *const *const std::os::raw::c_void = std::ptr::null();
            let mut num: u32 = 0;

            let err = unsafe { get_providers(&mut providers, &mut num) };
            eprintln!("getProviders returned: {err}, num_providers: {num}");

            if err == 0 && num > 0 && !providers.is_null() {
                let first = unsafe { *providers };
                eprintln!("First provider ptr: {first:?}");
                if !first.is_null() {
                    // Read first 4 bytes (backend_id)
                    let backend_id = unsafe { *(first as *const u32) };
                    eprintln!("backend_id (raw u32): {backend_id}");

                    // Read provider_name ptr (offset 8 on 64-bit)
                    let name_ptr = unsafe { *((first as *const u8).add(8) as *const *const std::os::raw::c_char) };
                    if !name_ptr.is_null() {
                        let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) };
                        eprintln!("provider_name: {:?}", name);
                    } else {
                        eprintln!("provider_name: null");
                    }

                    // Read api_version (offset 16: 6x u32 = core major/minor/patch + backend major/minor/patch)
                    let ver_ptr = unsafe { (first as *const u8).add(16) as *const u32 };
                    let core_major = unsafe { *ver_ptr };
                    let core_minor = unsafe { *ver_ptr.add(1) };
                    let core_patch = unsafe { *ver_ptr.add(2) };
                    let be_major = unsafe { *ver_ptr.add(3) };
                    let be_minor = unsafe { *ver_ptr.add(4) };
                    let be_patch = unsafe { *ver_ptr.add(5) };
                    eprintln!("core API: v{core_major}.{core_minor}.{core_patch}");
                    eprintln!("backend API: v{be_major}.{be_minor}.{be_patch}");

                    // Read first function pointer in vtable (offset 40 = 16 + 24)
                    let vtable_start = unsafe { (first as *const u8).add(40) as *const *const std::os::raw::c_void };
                    let first_fn = unsafe { *vtable_start };
                    eprintln!("First vtable entry (propertyHasCapability): {first_fn:?}");
                }
            }
        }
        Err(e) => {
            eprintln!("dlopen failed: {e}");
        }
    }
}
