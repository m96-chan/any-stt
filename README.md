# any-stt

Cross-platform Speech-to-Text engine for Rust. Platform differences abstracted behind a unified API.

## Architecture

```
any-stt/
  crates/
    any-stt/           # Core crate: trait + public API
      src/
        lib.rs         # pub trait SttEngine, SttResult, initialize(), transcribe()
        config.rs      # SttConfig (language, model_path, etc.)
    whisper-backend/   # kotoba-whisper via whisper.cpp
      src/
        lib.rs         # WhisperSttEngine: CPU backend, all platforms
    android-backend/   # Android SpeechRecognizer via JNI (future)
    ios-backend/       # Apple Speech.framework via ObjC bridge (future)
    web-backend/       # Web Speech API via wasm-bindgen (future)
```

## Usage

```toml
[dependencies]
any-stt = { path = "crates/any-stt" }

[features]
default = ["whisper"]
whisper = ["any-stt/whisper"]
android = ["any-stt/android"]  # future
ios = ["any-stt/ios"]          # future
```

```rust
use any_stt::{SttEngine, SttConfig};

let config = SttConfig {
    language: "ja".to_string(),
    model_path: Some("/path/to/ggml-kotoba-whisper-v2.0.bin".into()),
    ..Default::default()
};

let engine = any_stt::initialize(config)?;
let result = engine.transcribe(&audio_samples)?;
println!("{}", result.text);
```

## API

```rust
pub struct SttConfig {
    pub language: String,           // "ja", "en", etc.
    pub model_path: Option<PathBuf>, // whisper model path (whisper backend)
    pub sample_rate: u32,           // default: 16000
}

pub struct SttResult {
    pub text: String,
    pub language: String,
    pub duration_ms: f64,
}

pub trait SttEngine: Send + Sync {
    fn transcribe(&self, audio: &[f32]) -> Result<SttResult, SttError>;
    fn is_ready(&self) -> bool;
}

pub fn initialize(config: SttConfig) -> Result<Box<dyn SttEngine>, SttError>;
```

## Backends

### Phase 1: whisper (kotoba-whisper-v2.0)
- **All platforms** via whisper.cpp C bindings
- kotoba-whisper-v2.0: CER ~3% for Japanese
- CPU execution on background thread (no GPU/rendering interference)
- Model: `ggml-kotoba-whisper-v2.0-q5_0.bin` (~500MB quantized)

### Phase 2: Platform-native (future)
- **Android**: `SpeechRecognizer` via JNI — zero CPU/GPU load, Google on-device recognition
- **iOS**: `Speech.framework` via ObjC bridge — Apple on-device recognition
- **Web**: Web Speech API via wasm-bindgen

## Integration with Godot (charactor-assistant)

```toml
# godot-miotts/Cargo.toml
[dependencies]
any-stt = { git = "https://github.com/m96-chan/any-stt", features = ["whisper"] }
```

`MioStt` GodotClass calls `any_stt::initialize()` once, then `engine.transcribe()` per utterance.
VAD (Voice Activity Detection) remains in Godot GDScript side.

## Related
- [any-miotts](https://github.com/m96-chan/any-miotts) — TTS engine (same pattern)
- [charactor-assistant](https://github.com/ai-create-lab/charactor-assistant) — Consumer app
- [kotoba-whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) — Japanese ASR model
