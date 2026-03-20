use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let whisper_dir = manifest_dir
        .join("../../third-party/whisper.cpp")
        .canonicalize()
        .expect("third-party/whisper.cpp not found — did you init the submodule?");

    let target = env::var("TARGET").unwrap_or_default();
    let is_android = target.contains("android");
    let is_cross = env::var("HOST").unwrap_or_default() != target;
    let use_vulkan = env::var("CARGO_FEATURE_VULKAN").is_ok();

    // --- Step 1: Build whisper.cpp via cmake ---

    let mut cfg = cmake::Config::new(&whisper_dir);

    // Android: shared libs (loaded via dlopen to avoid static libc crash).
    // Others: static libs (linked directly).
    cfg.define("BUILD_SHARED_LIBS", if is_android { "ON" } else { "OFF" })
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_BUILD_SERVER", "OFF")
        .define("WHISPER_COREML", "OFF")
        .define("WHISPER_OPENVINO", "OFF")
        .define("WHISPER_CURL", "OFF")
        .define("WHISPER_SDL2", "OFF")
        .define("GGML_NATIVE", if is_cross { "OFF" } else { "ON" })
        .define("GGML_CUDA", "OFF")
        .define("GGML_METAL", "OFF")
        .define("GGML_VULKAN", if use_vulkan { "ON" } else { "OFF" })
        .define("GGML_OPENCL", "OFF")
        .define("GGML_SYCL", "OFF")
        .define("GGML_RPC", "OFF");

    if is_android {
        // OpenMP is now safe with dynamic linking (no static libc crash).
        // This enables multi-threaded ggml-cpu on Android.
        cfg.define("GGML_OPENMP", "ON");
        if let Ok(ndk) = env::var("ANDROID_NDK_HOME") {
            let toolchain = format!("{ndk}/build/cmake/android.toolchain.cmake");
            cfg.define("CMAKE_TOOLCHAIN_FILE", &toolchain)
                .define("ANDROID_ABI", "arm64-v8a")
                .define("ANDROID_PLATFORM", "android-35")
                .define("ANDROID_STL", "c++_shared");
        }
    }

    let dst = cfg.build();

    let lib_dir = dst.join("lib");
    let lib64_dir = dst.join("lib64");

    if is_android {
        // ============================================================
        // Android: dlopen-based loading.
        //
        // We do NOT link whisper/ggml/shim at all. The .so files are
        // built by cmake and deployed alongside the binary. At runtime,
        // WhisperEngine uses libloading to dlopen them.
        //
        // This avoids the NDK static libc getauxval crash on Android 15.
        // ============================================================

        // Build the shim as a shared library too.
        if let Ok(ndk) = env::var("ANDROID_NDK_HOME") {
            let ndk_bin =
                format!("{ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin");
            let cc = format!("{ndk_bin}/aarch64-linux-android35-clang");

            let include_dir = dst.join("include");
            let shim_src = manifest_dir.join("csrc/shim.c");
            let shim_out = lib_dir.join("libwhisper_shim.so");

            let status = std::process::Command::new(&cc)
                .args([
                    "-shared", "-fPIC", "-o",
                    shim_out.to_str().unwrap(),
                    shim_src.to_str().unwrap(),
                    &format!("-I{}", include_dir.display()),
                    &format!("-I{}", whisper_dir.join("include").display()),
                    &format!("-I{}", whisper_dir.join("ggml/include").display()),
                    &format!("-L{}", lib_dir.display()),
                    "-lwhisper",
                    &format!("-Wl,-rpath,$ORIGIN"),
                ])
                .status()
                .expect("failed to compile shim.so");
            assert!(status.success(), "shim.so compilation failed");
        }

        // Dynamic link the shared libraries.
        // At runtime, LD_LIBRARY_PATH must point to the directory containing
        // these .so files. This avoids the static libc getauxval crash while
        // still resolving symbols at link time.
        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
        println!("cargo:rustc-link-lib=dylib=whisper_shim");
        println!("cargo:rustc-link-lib=dylib=whisper");
        println!("cargo:rustc-link-lib=dylib=ggml");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
        println!("cargo:rustc-link-lib=dylib=ggml-cpu");

        if use_vulkan {
            println!("cargo:rustc-link-lib=dylib=ggml-vulkan");
        }

        // Output the lib dir path as a cargo env var so tests can find the .so files.
        println!(
            "cargo:rustc-env=WHISPER_LIB_DIR={}",
            lib_dir.display()
        );
    } else {
        // ============================================================
        // Non-Android: static linking (original approach).
        // ============================================================

        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
        if lib64_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib64_dir.display());
        }

        // Build shim as static lib.
        let include_dir = dst.join("include");
        cc::Build::new()
            .file(manifest_dir.join("csrc/shim.c"))
            .include(&include_dir)
            .include(whisper_dir.join("include"))
            .include(whisper_dir.join("ggml/include"))
            .warnings(false)
            .compile("whisper_shim");

        println!("cargo:rustc-link-lib=static=whisper");
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-base");
        println!("cargo:rustc-link-lib=static=ggml-cpu");

        if use_vulkan {
            println!("cargo:rustc-link-lib=static=ggml-vulkan");
            println!("cargo:rustc-link-lib=vulkan");
        }

        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=pthread");

        #[cfg(target_os = "macos")]
        {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=c++");
        }
    }

    println!("cargo:rerun-if-changed={}", whisper_dir.display());
    println!("cargo:rerun-if-changed=csrc/shim.c");
}
