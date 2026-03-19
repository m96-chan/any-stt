use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let whisper_dir = manifest_dir
        .join("../../third-party/whisper.cpp")
        .canonicalize()
        .expect("third-party/whisper.cpp not found — did you init the submodule?");

    // --- Step 1: Build whisper.cpp via cmake ---

    let mut cfg = cmake::Config::new(&whisper_dir);

    let target = env::var("TARGET").unwrap_or_default();
    let is_android = target.contains("android");
    let is_cross = env::var("HOST").unwrap_or_default() != target;

    // Build shared library on Android (static init in .so avoids linker issues),
    // static library on other platforms.
    cfg.define("BUILD_SHARED_LIBS", if is_android { "ON" } else { "OFF" })
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_BUILD_SERVER", "OFF")
        .define("WHISPER_COREML", "OFF")
        .define("WHISPER_OPENVINO", "OFF")
        .define("WHISPER_CURL", "OFF")
        .define("WHISPER_SDL2", "OFF")
        // GGML_NATIVE uses -mcpu=native which breaks cross-compilation.
        .define("GGML_NATIVE", if is_cross { "OFF" } else { "ON" })
        // Disable all GPU backends for the base CPU build.
        .define("GGML_CUDA", "OFF")
        .define("GGML_METAL", "OFF")
        .define("GGML_VULKAN", "OFF")
        .define("GGML_OPENCL", "OFF")
        .define("GGML_SYCL", "OFF")
        .define("GGML_RPC", "OFF");

    // Android: use NDK cmake toolchain and disable OpenMP.
    if is_android {
        cfg.define("GGML_OPENMP", "OFF");

        if let Ok(ndk) = env::var("ANDROID_NDK_HOME") {
            let toolchain = format!("{ndk}/build/cmake/android.toolchain.cmake");
            cfg.define("CMAKE_TOOLCHAIN_FILE", &toolchain)
                .define("ANDROID_ABI", "arm64-v8a")
                .define("ANDROID_PLATFORM", "android-35")
                .define("ANDROID_STL", "c++_shared");
        }
    }

    let dst = cfg.build();

    // Link search paths — cmake installs libs into lib/ or lib64/.
    let lib_dir = dst.join("lib");
    let lib64_dir = dst.join("lib64");
    if lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    if lib64_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib64_dir.display());
    }

    // --- Step 2: Build our C shim ---

    let include_dir = dst.join("include");
    cc::Build::new()
        .file(manifest_dir.join("csrc/shim.c"))
        .include(&include_dir)
        .include(whisper_dir.join("include"))
        .include(whisper_dir.join("ggml/include"))
        .warnings(false)
        .compile("whisper_shim");

    // --- Step 3: Link ---

    if is_android {
        // Android: link whisper as shared library.
        // TODO: whisper.so's static init triggers getauxval crash on Android 15
        // due to NDK's static libc. Need to dlopen whisper.so at runtime instead.
        println!("cargo:rustc-link-lib=dylib=whisper");
        println!("cargo:rustc-link-lib=dylib=ggml");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
        println!("cargo:rustc-link-lib=dylib=ggml-cpu");
        println!("cargo:rustc-link-lib=c++_shared");
    } else {
        // Other platforms: static link for simplicity.
        println!("cargo:rustc-link-lib=static=whisper");
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-base");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
    }

    // System dependencies.
    if is_android {
        // Add NDK sysroot lib path for C++ standard library.
        if let Ok(ndk) = env::var("ANDROID_NDK_HOME") {
            let ndk_lib = format!(
                "{ndk}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android"
            );
            println!("cargo:rustc-link-search=native={ndk_lib}");
        }
        // Don't link c++ at all from Rust side — whisper.so already links it.
        // Android bionic has pthreads built-in; no -lpthread needed.
    } else {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=pthread");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    }

    // Re-run if source changes.
    println!("cargo:rerun-if-changed={}", whisper_dir.display());
    println!("cargo:rerun-if-changed=csrc/shim.c");
}
