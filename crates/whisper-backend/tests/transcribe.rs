//! Integration test: transcribe JFK audio sample and verify output.
//!
//! Requires:
//! - Model: third-party/whisper.cpp/models/ggml-tiny.en.bin
//!   (download via: bash third-party/whisper.cpp/models/download-ggml-model.sh tiny.en)
//! - Audio: third-party/whisper.cpp/samples/jfk.wav
//!
//! The test is ignored by default if the model file is not present,
//! so `cargo test` always passes even without downloading the model.

use std::path::{Path, PathBuf};

use any_stt::config::Backend;
use any_stt::SttEngine;
use whisper_backend::WhisperEngine;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn model_path() -> PathBuf {
    repo_root().join("third-party/whisper.cpp/models/ggml-tiny.en.bin")
}

fn jfk_wav_path() -> PathBuf {
    repo_root().join("third-party/whisper.cpp/samples/jfk.wav")
}

/// Read a WAV file and return f32 samples normalized to [-1, 1].
fn read_wav(path: &Path) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("failed to open WAV file");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected mono audio");
    assert_eq!(spec.sample_rate, 16000, "expected 16kHz sample rate");

    match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap())
            .collect(),
    }
}

/// Reference text from JFK's inaugural address excerpt.
/// The exact expected output from whisper tiny.en on jfk.wav.
const _JFK_REFERENCE: &str =
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

#[test]
fn transcribe_jfk_english() {
    let model = model_path();
    if !model.exists() {
        eprintln!(
            "SKIPPED: model not found at {}. Run: bash third-party/whisper.cpp/models/download-ggml-model.sh tiny.en",
            model.display()
        );
        return;
    }

    let wav = jfk_wav_path();
    assert!(wav.exists(), "jfk.wav not found");

    let hw = any_stt::detect_hardware();
    let engine = WhisperEngine::new(&model, "en", Backend::Cpu, hw)
        .expect("failed to load model");

    assert!(engine.is_ready());
    assert_eq!(engine.active_backend(), Backend::Cpu);

    let audio = read_wav(&wav);
    let result = engine.transcribe(&audio).expect("transcription failed");

    eprintln!("Transcribed text: {:?}", result.text);
    eprintln!("Language: {}", result.language);
    eprintln!("Duration: {:.1}ms", result.duration_ms);

    // Normalize whitespace for comparison.
    let text = result.text.trim();

    // Check that the core content matches.
    // Allow minor variations (whisper may produce slightly different punctuation).
    assert!(
        text.contains("ask not what your country can do for you"),
        "expected JFK quote in output, got: {text:?}"
    );
    assert!(
        text.contains("ask what you can do for your country"),
        "expected JFK quote continuation in output, got: {text:?}"
    );

    // Verify language detection.
    assert_eq!(result.language, "en");

    // Verify backend.
    assert_eq!(result.backend_used, Backend::Cpu);
}
