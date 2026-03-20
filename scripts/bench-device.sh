#!/usr/bin/env bash
# Cross-compile bench-whisper for Android aarch64 and run on device via adb.
#
# Prerequisites:
#   - ANDROID_NDK_HOME set (e.g., ~/Android/Sdk/ndk/26.3.11579264)
#   - adb connected to device (adb devices)
#   - Rust target: rustup target add aarch64-linux-android
#   - tiny.en model: third-party/whisper.cpp/models/ggml-tiny.en.bin
#
# Usage:
#   ./scripts/bench-device.sh [--vulkan] [--runs N] [--backend cpu|gpu|npu|all]
#
# Example:
#   ./scripts/bench-device.sh --vulkan --backend all --runs 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET=aarch64-linux-android
DEVICE_DIR=/data/local/tmp/any-stt
MODEL=third-party/whisper.cpp/models/ggml-tiny.en.bin
AUDIO=third-party/whisper.cpp/samples/jfk.wav
RUNS=3
BACKEND=all
FEATURES=""
ADB_TRANSPORT=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vulkan)
            FEATURES="vulkan"
            shift ;;
        --runs)
            RUNS="$2"
            shift 2 ;;
        --backend)
            BACKEND="$2"
            shift 2 ;;
        --model)
            MODEL="$2"
            shift 2 ;;
        --transport|-t)
            ADB_TRANSPORT="$2"
            shift 2 ;;
        *)
            echo "Unknown arg: $1"
            exit 1 ;;
    esac
done

# Build adb command with optional transport_id
ADB="adb"
if [[ -n "$ADB_TRANSPORT" ]]; then
    ADB="adb -t $ADB_TRANSPORT"
fi

# Validate
if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    echo "ERROR: ANDROID_NDK_HOME not set"
    exit 1
fi

if [[ -z "$ADB_TRANSPORT" ]]; then
    if ! adb devices | grep -q "device$"; then
        echo "ERROR: no adb device connected"
        exit 1
    fi
fi

if [[ ! -f "$ROOT/$MODEL" ]]; then
    echo "ERROR: model not found: $ROOT/$MODEL"
    echo "Download it:  cd third-party/whisper.cpp && bash models/download-ggml-model.sh tiny.en"
    exit 1
fi

echo "=== Cross-compile for $TARGET ==="

# Set up Android linker
NDK_TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$NDK_TOOLCHAIN/bin/aarch64-linux-android35-clang"
export CC_aarch64_linux_android="$NDK_TOOLCHAIN/bin/aarch64-linux-android35-clang"
export CXX_aarch64_linux_android="$NDK_TOOLCHAIN/bin/aarch64-linux-android35-clang++"
export AR_aarch64_linux_android="$NDK_TOOLCHAIN/bin/llvm-ar"

CARGO_ARGS="--release --target $TARGET -p bench"
if [[ -n "$FEATURES" ]]; then
    CARGO_ARGS="$CARGO_ARGS --features $FEATURES"
fi

cd "$ROOT"
echo "cargo build $CARGO_ARGS"
cargo build $CARGO_ARGS

echo ""
echo "=== Deploy to device ==="

BENCH_BIN="target/$TARGET/release/bench-whisper"
if [[ ! -f "$BENCH_BIN" ]]; then
    echo "ERROR: binary not found: $BENCH_BIN"
    exit 1
fi

# Create device directory
$ADB shell "mkdir -p $DEVICE_DIR"

# Push binary
echo "Pushing bench-whisper..."
$ADB push "$BENCH_BIN" "$DEVICE_DIR/bench-whisper"
$ADB shell "chmod +x $DEVICE_DIR/bench-whisper"

# Push shared libraries (for Android whisper.cpp dlopen)
BUILD_LIB_DIR=$(find target/$TARGET/release/build/whisper-backend-*/out -name "lib" -type d 2>/dev/null | head -1)
if [[ -n "$BUILD_LIB_DIR" && -d "$BUILD_LIB_DIR" ]]; then
    echo "Pushing shared libraries from $BUILD_LIB_DIR..."
    for so in "$BUILD_LIB_DIR"/*.so; do
        if [[ -f "$so" ]]; then
            soname=$(basename "$so")
            # Skip Vulkan .so unless --vulkan flag is set (prevents ggml auto-detection
            # which causes 3.6x CPU slowdown on Adreno devices)
            if [[ "$soname" == "libggml-vulkan.so" && -z "$FEATURES" ]]; then
                echo "  $soname (SKIPPED — use --vulkan to include)"
                continue
            fi
            echo "  $soname"
            $ADB push "$so" "$DEVICE_DIR/"
        fi
    done
fi

# Push libc++_shared.so from NDK
LIBCXX="$NDK_TOOLCHAIN/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [[ -f "$LIBCXX" ]]; then
    echo "Pushing libc++_shared.so..."
    $ADB push "$LIBCXX" "$DEVICE_DIR/"
fi

# Push libomp.so from NDK (for OpenMP multi-threading)
LIBOMP="$NDK_TOOLCHAIN/lib/aarch64-linux-android/libomp.so"
if [[ -f "$LIBOMP" ]]; then
    echo "Pushing libomp.so..."
    $ADB push "$LIBOMP" "$DEVICE_DIR/"
else
    # Try alternative path
    LIBOMP_ALT=$(find "$NDK_TOOLCHAIN" -name "libomp.so" -path "*aarch64*" 2>/dev/null | head -1)
    if [[ -n "$LIBOMP_ALT" ]]; then
        echo "Pushing libomp.so from $LIBOMP_ALT..."
        $ADB push "$LIBOMP_ALT" "$DEVICE_DIR/"
    fi
fi

# Push model
echo "Pushing model..."
$ADB push "$ROOT/$MODEL" "$DEVICE_DIR/$(basename "$MODEL")"

# Push audio
echo "Pushing audio..."
$ADB push "$ROOT/$AUDIO" "$DEVICE_DIR/jfk.wav"

# Push QNN libs if available
if [[ -n "${QNN_SDK_ROOT:-}" ]]; then
    QNN_LIBS="$QNN_SDK_ROOT/lib/aarch64-android"
    if [[ -d "$QNN_LIBS" ]]; then
        echo "Pushing QNN libraries..."
        for lib in libQnnHtp.so libQnnHtpV75Stub.so libQnnSystem.so libQnnHtpPrepare.so; do
            if [[ -f "$QNN_LIBS/$lib" ]]; then
                echo "  $lib"
                $ADB push "$QNN_LIBS/$lib" "$DEVICE_DIR/"
            fi
        done
    fi
fi

echo ""
echo "=== Running benchmark on device ==="
echo ""

# Build the command
DEVICE_CMD="cd $DEVICE_DIR && "
DEVICE_CMD+="LD_LIBRARY_PATH=$DEVICE_DIR "
DEVICE_CMD+="ADSP_LIBRARY_PATH=$DEVICE_DIR "
DEVICE_CMD+="QNN_LIB_DIR=$DEVICE_DIR "
DEVICE_CMD+="WHISPER_LIB_DIR=$DEVICE_DIR "
DEVICE_CMD+="./bench-whisper "
DEVICE_CMD+="--model $DEVICE_DIR/$(basename "$MODEL") "
DEVICE_CMD+="--audio $DEVICE_DIR/jfk.wav "
DEVICE_CMD+="--runs $RUNS "
DEVICE_CMD+="--backend $BACKEND"

echo "$ $DEVICE_CMD"
echo ""

$ADB shell "$DEVICE_CMD"
