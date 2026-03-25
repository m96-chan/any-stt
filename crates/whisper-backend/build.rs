use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Pinned whisper.cpp commit (m96-chan fork with encoder output injection API).
const WHISPER_CPP_REPO: &str = "https://github.com/m96-chan/whisper.cpp.git";
const WHISPER_CPP_REV: &str = "458ba53d";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Try submodule path first (git checkout), then fall back to fetching
    // source automatically (crates.io install).
    let whisper_dir = resolve_whisper_source(&manifest_dir);

    let target = env::var("TARGET").unwrap_or_default();
    let host = env::var("HOST").unwrap_or_default();
    let is_android = target.contains("android");
    let is_ios = target.contains("apple-ios");
    let is_macos = target.contains("apple-darwin");
    let is_windows = target.contains("windows");
    let is_apple = is_ios || is_macos;
    let is_cross = host != target;

    let use_vulkan = env::var("CARGO_FEATURE_VULKAN").is_ok();
    let use_metal = env::var("CARGO_FEATURE_METAL").is_ok() || is_ios;
    let use_coreml = env::var("CARGO_FEATURE_COREML").is_ok();
    let use_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();

    // --- Step 1: Configure cmake ---

    let mut cfg = cmake::Config::new(&whisper_dir);

    cfg.define("BUILD_SHARED_LIBS", if is_android { "ON" } else { "OFF" })
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_BUILD_SERVER", "OFF")
        .define("WHISPER_OPENVINO", "OFF")
        .define("WHISPER_CURL", "OFF")
        .define("WHISPER_SDL2", "OFF")
        .define("GGML_NATIVE", if is_cross { "OFF" } else { "ON" })
        .define("GGML_CUDA", if use_cuda { "ON" } else { "OFF" })
        .define("GGML_OPENCL", "OFF")
        .define("GGML_SYCL", "OFF")
        .define("GGML_RPC", "OFF");

    // --- Metal ---
    if use_metal {
        cfg.define("GGML_METAL", "ON");
        // Embed Metal shaders into the static library — avoids runtime .metallib file
        cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
    } else {
        cfg.define("GGML_METAL", "OFF");
    }

    // --- CoreML ---
    if use_coreml {
        cfg.define("WHISPER_COREML", "ON");
        // Allow fallback to CPU/Metal when .mlmodelc files are missing
        cfg.define("WHISPER_COREML_ALLOW_FALLBACK", "ON");
    } else {
        cfg.define("WHISPER_COREML", "OFF");
    }

    // --- Vulkan ---
    cfg.define("GGML_VULKAN", if use_vulkan { "ON" } else { "OFF" });

    // --- Platform-specific cmake config ---

    if is_ios {
        // iOS: static libs, no OpenMP, iOS SDK
        cfg.define("GGML_OPENMP", "OFF");
        cfg.define("CMAKE_SYSTEM_NAME", "iOS");
        cfg.define("CMAKE_OSX_SYSROOT", "iphoneos");
        cfg.define("CMAKE_OSX_ARCHITECTURES", "arm64");
        cfg.define("CMAKE_OSX_DEPLOYMENT_TARGET", "16.0");
    } else if is_android {
        cfg.define("GGML_OPENMP", "ON");
        if let Ok(ndk) = env::var("ANDROID_NDK_HOME") {
            let toolchain = format!("{ndk}/build/cmake/android.toolchain.cmake");
            cfg.define("CMAKE_TOOLCHAIN_FILE", &toolchain)
                .define("ANDROID_ABI", "arm64-v8a")
                .define("ANDROID_PLATFORM", "android-35")
                .define("ANDROID_STL", "c++_shared");
        }
    } else if is_macos {
        // macOS: enable OpenMP if available, Accelerate framework
        cfg.define("GGML_OPENMP", "OFF"); // macOS homebrew OpenMP is fragile
    } else if is_windows {
        // MSVC OpenMP uses vcomp (incompatible with gomp); disable for simplicity.
        cfg.define("GGML_OPENMP", "OFF");
    } else {
        // Linux: OpenMP enabled
        cfg.define("GGML_OPENMP", "ON");
    }

    // --- Step 2: Build ---

    let dst = cfg.build();
    let lib_dir = dst.join("lib");
    let lib64_dir = dst.join("lib64");

    // --- Step 3: Link ---

    if is_android {
        // Android: dynamic linking (avoids static libc crash)
        build_android_shim(&manifest_dir, &whisper_dir, &dst, &lib_dir);

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
        println!("cargo:rustc-env=WHISPER_LIB_DIR={}", lib_dir.display());
    } else {
        // Static linking (iOS, macOS, Linux, Windows)
        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
        if lib64_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib64_dir.display());
        }

        // Build shim
        let include_dir = dst.join("include");
        cc::Build::new()
            .file(manifest_dir.join("csrc/shim.c"))
            .include(&include_dir)
            .include(whisper_dir.join("include"))
            .include(whisper_dir.join("ggml/include"))
            .warnings(false)
            .compile("whisper_shim");

        // Core whisper.cpp libraries
        println!("cargo:rustc-link-lib=static=whisper");
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-base");
        println!("cargo:rustc-link-lib=static=ggml-cpu");

        // Metal
        if use_metal {
            println!("cargo:rustc-link-lib=static=ggml-metal");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }

        // CoreML
        if use_coreml {
            println!("cargo:rustc-link-lib=static=whisper.coreml");
            println!("cargo:rustc-link-lib=framework=CoreML");
        }

        // CUDA
        if use_cuda {
            println!("cargo:rustc-link-lib=static=ggml-cuda");
            // CUDA runtime + driver libraries
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cublas");
            println!("cargo:rustc-link-lib=cublasLt");
            println!("cargo:rustc-link-lib=cuda"); // driver API (cuMemAddressFree etc.)
            // CUDA lib paths
            if is_windows {
                // Windows: CUDA_PATH env var + lib\x64
                if let Ok(cuda_path) = env::var("CUDA_PATH") {
                    println!("cargo:rustc-link-search=native={cuda_path}\\lib\\x64");
                }
            } else {
                // Linux
                if let Ok(cuda_home) = env::var("CUDA_HOME") {
                    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
                } else {
                    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
                    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
                }
                // NVIDIA driver stub (for libcuda.so)
                println!("cargo:rustc-link-search=native=/usr/lib64");
                println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
            }
        }

        // Vulkan
        if use_vulkan {
            println!("cargo:rustc-link-lib=static=ggml-vulkan");
            if is_windows {
                // Windows: vulkan-1.lib (not libvulkan.so)
                println!("cargo:rustc-link-lib=vulkan-1");
                if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
                    println!("cargo:rustc-link-search=native={vulkan_sdk}\\Lib");
                }
            } else {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }

        // Platform-specific system libraries
        if is_apple {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=c++");
        } else if is_windows {
            // MSVC links C++ runtime automatically.
        } else {
            // Linux
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=gomp");
            println!("cargo:rustc-link-lib=pthread");
        }
    }

    println!("cargo:rerun-if-changed={}", whisper_dir.display());
    println!("cargo:rerun-if-changed=csrc/shim.c");
}

/// Build the C shim as a shared library for Android (NDK clang).
fn build_android_shim(
    manifest_dir: &PathBuf,
    whisper_dir: &PathBuf,
    dst: &PathBuf,
    lib_dir: &PathBuf,
) {
    if let Ok(ndk) = env::var("ANDROID_NDK_HOME") {
        let ndk_bin = format!("{ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin");
        let cc = format!("{ndk_bin}/aarch64-linux-android35-clang");

        let include_dir = dst.join("include");
        let shim_src = manifest_dir.join("csrc/shim.c");
        let shim_out = lib_dir.join("libwhisper_shim.so");

        let status = std::process::Command::new(&cc)
            .args([
                "-shared",
                "-fPIC",
                "-o",
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
}

/// Resolve the whisper.cpp source directory.
///
/// 1. If the git submodule exists (`../../third-party/whisper.cpp`), use it.
/// 2. Otherwise, clone the pinned commit into `OUT_DIR/whisper.cpp` (crates.io).
fn resolve_whisper_source(manifest_dir: &PathBuf) -> PathBuf {
    // Check submodule path (works in git checkout)
    let submodule_dir = manifest_dir.join("../../third-party/whisper.cpp");
    if submodule_dir.join("CMakeLists.txt").exists() {
        return submodule_dir.canonicalize().unwrap();
    }

    // crates.io: fetch source into OUT_DIR
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let clone_dir = out_dir.join("whisper.cpp");

    if clone_dir.join("CMakeLists.txt").exists() {
        return clone_dir;
    }

    eprintln!(
        "whisper.cpp submodule not found — cloning {} @ {}",
        WHISPER_CPP_REPO, WHISPER_CPP_REV
    );

    let status = Command::new("git")
        .args([
            "clone",
            "--depth", "1",
            "--single-branch",
            WHISPER_CPP_REPO,
            clone_dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run git — is git installed?");
    assert!(status.success(), "git clone failed");

    // Fetch the exact commit if shallow clone HEAD differs
    let status = Command::new("git")
        .current_dir(&clone_dir)
        .args(["fetch", "--depth", "1", "origin", WHISPER_CPP_REV])
        .status()
        .expect("git fetch failed");
    if status.success() {
        let status = Command::new("git")
            .current_dir(&clone_dir)
            .args(["checkout", WHISPER_CPP_REV])
            .status()
            .expect("git checkout failed");
        assert!(status.success(), "git checkout failed");
    }

    clone_dir
}
