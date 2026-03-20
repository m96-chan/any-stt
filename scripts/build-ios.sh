#!/usr/bin/env bash
# Build any-stt for iOS (aarch64-apple-ios).
#
# Run on macOS with Xcode installed.
#
# Prerequisites:
#   - Xcode with iOS SDK
#   - rustup target add aarch64-apple-ios
#   - cmake (brew install cmake)
#
# Usage:
#   ./scripts/build-ios.sh [--coreml]
#
# Output:
#   target/aarch64-apple-ios/release/libwhisper_backend.a (static library)
#   + all whisper.cpp static libs

set -euo pipefail

TARGET=aarch64-apple-ios
FEATURES="metal"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --coreml)
            FEATURES="metal,coreml"
            shift ;;
        *)
            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Building for $TARGET (features: $FEATURES) ==="

# Verify toolchain
if ! rustup target list --installed | grep -q "$TARGET"; then
    echo "Installing Rust target $TARGET..."
    rustup target add "$TARGET"
fi

# Verify Xcode
if ! xcrun --sdk iphoneos --show-sdk-path >/dev/null 2>&1; then
    echo "ERROR: Xcode iOS SDK not found. Install Xcode and run 'xcode-select --install'"
    exit 1
fi

SDK_PATH=$(xcrun --sdk iphoneos --show-sdk-path)
echo "iOS SDK: $SDK_PATH"

# Build
cargo build \
    --release \
    --target "$TARGET" \
    -p whisper-backend \
    --features "$FEATURES"

echo ""
echo "=== Build complete ==="
echo "Static libraries in: target/$TARGET/release/"
echo ""
echo "To integrate with an iOS app:"
echo "  1. Link libwhisper_backend.a and all whisper.cpp static libs"
echo "  2. Add frameworks: Foundation, CoreML, Metal, MetalKit, Accelerate"
echo "  3. Bundle your .ggml/.gguf model file in the app"
echo "  4. For CoreML: include .mlmodelc companion files"
