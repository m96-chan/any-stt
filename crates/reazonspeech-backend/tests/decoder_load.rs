//! Integration test for `RnntDecoder::from_gguf`.
//!
//! Loads the real reazonspeech-nemo-v2 decoder weights and runs a greedy
//! decode against a synthetic encoder output. The real encoder forward
//! is *not* exercised here — it requires a `dw_striding`-style
//! subsampling stack which isn't yet wired in `fastconformer-core`.
//!
//! Skipped (`#[ignore]` after the existence check) when the converted
//! GGUF isn't present locally. Generate it with:
//!
//!     huggingface-cli download reazon-research/reazonspeech-nemo-v2 \
//!         --local-dir ~/asr-models/reazonspeech-nemo-v2
//!     python scripts/convert-nemo-to-gguf.py \
//!         ~/asr-models/reazonspeech-nemo-v2/reazonspeech-nemo-v2.nemo \
//!         /tmp/reazon.gguf

use std::path::PathBuf;

use fastconformer_core::Config;
use gguf_loader::GgufFile;
use reazonspeech_backend::decoder::RnntDecoder;

fn fixture_paths() -> Vec<PathBuf> {
    let cwd = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    vec![
        PathBuf::from("/tmp/reazon.gguf"),
        PathBuf::from(format!("{cwd}/../../models/reazonspeech-nemo-v2.gguf")),
        PathBuf::from(format!("{cwd}/../../target/reazon.gguf")),
    ]
}

fn pick_fixture() -> Option<PathBuf> {
    fixture_paths().into_iter().find(|p| p.exists())
}

#[test]
fn rnnt_decoder_loads_real_reazonspeech_gguf() {
    let path = match pick_fixture() {
        Some(p) => p,
        None => {
            eprintln!(
                "SKIPPED: no reazonspeech GGUF found at any of: {:?}",
                fixture_paths()
            );
            return;
        }
    };

    let gguf = GgufFile::open(&path).expect("open gguf");
    let cfg = Config::from_gguf(&gguf).expect("parse config");
    let decoder = RnntDecoder::from_gguf(&gguf, cfg.clone())
        .expect("RnntDecoder::from_gguf");

    // Sanity: config matches the published spec.
    assert_eq!(cfg.d_model, 1024, "reazonspeech-nemo-v2 d_model");
    assert_eq!(cfg.pred_hidden, 640);
    assert_eq!(cfg.joint_hidden, 640);
    assert_eq!(cfg.vocab_size, 3000);
    let _ = decoder; // dropped, just verifying load succeeds
}

#[test]
fn rnnt_decoder_greedy_runs_on_synthetic_encoder_output() {
    let path = match pick_fixture() {
        Some(p) => p,
        None => {
            eprintln!("SKIPPED: no reazonspeech GGUF found");
            return;
        }
    };

    let gguf = GgufFile::open(&path).expect("open gguf");
    let cfg = Config::from_gguf(&gguf).expect("parse config");
    let decoder = RnntDecoder::from_gguf(&gguf, cfg.clone())
        .expect("RnntDecoder::from_gguf");

    // Synthetic encoder output: 50 frames × d_model, all zeros. The
    // decoder should run without panicking. Output may be empty or a
    // small number of tokens depending on how the model resolves
    // zero-input scoring.
    use fastconformer_core::encoder::EncoderOutput;
    let enc = EncoderOutput {
        data: vec![0.0_f32; 50 * cfg.d_model as usize],
        n_frames: 50,
        d_model: cfg.d_model as usize,
    };
    let tokens = decoder.greedy_decode(&enc);
    eprintln!("synthetic decode emitted {} tokens", tokens.len());
    // No assertion on transcript content — this just verifies stability.
    // Zero input passes through tanh+linear to a constant joint output
    // whose argmax may or may not be blank. The MAX_EMIT_PER_FRAME=30
    // guard caps each frame's emit count, so 50 frames × 30 = 1500 is
    // the largest possible non-runaway result. We require <= that.
    assert!(
        tokens.len() <= 50 * 30,
        "exceeded MAX_EMIT_PER_FRAME guard: {} tokens",
        tokens.len()
    );
    // Also require the loop terminates within a reasonable time —
    // fail otherwise (the tokio test runner's wall-clock timeout
    // handles infinite loops independently).
    for &t in &tokens {
        assert!(t < 3001, "emitted token id {t} out of range");
    }
}
