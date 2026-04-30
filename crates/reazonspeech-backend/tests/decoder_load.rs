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
    let _ = tokens.len();
}

#[test]
fn full_encoder_load_and_forward_runs_against_reazonspeech() {
    // End-to-end smoke test: load encoder + decoder from real GGUF,
    // run a short audio through mel → encoder → decoder pipeline,
    // verify it terminates without error or panic. Numerical
    // correctness against the Python reference is in #N9.
    let path = match pick_fixture() {
        Some(p) => p,
        None => {
            eprintln!("SKIPPED: no reazonspeech GGUF found");
            return;
        }
    };

    let gguf = GgufFile::open(&path).expect("open gguf");
    let cfg = Config::from_gguf(&gguf).expect("parse config");

    let encoder = reazonspeech_backend::encoder::load(&gguf, cfg.clone())
        .expect("encoder load");
    let decoder = RnntDecoder::from_gguf(&gguf, cfg.clone())
        .expect("decoder load");

    // Synthetic audio: 0.5s of zeros at 16 kHz. Real transcription
    // requires real audio + correct mel preprocessor. This just
    // exercises the call graph.
    let audio = vec![0.0_f32; 8000];
    let mel = fastconformer_core::log_mel_spectrogram(&audio, &cfg);
    eprintln!(
        "mel: {} frames × {} mels",
        mel.n_frames, mel.n_mels
    );

    let enc_out = encoder
        .forward(&mel.data, mel.n_frames)
        .expect("encoder forward");
    eprintln!(
        "encoder out: {} frames × {} d_model",
        enc_out.n_frames, enc_out.d_model
    );
    assert_eq!(enc_out.d_model, cfg.d_model as usize);
    // Encoder reduces time by 8× (subsampling factor).
    assert!(enc_out.n_frames > 0, "encoder produced 0 frames");
    assert!(enc_out.n_frames <= mel.n_frames / 4, "subsampling didn't reduce");

    let tokens = decoder.greedy_decode(&enc_out);
    eprintln!(
        "transcribe of zeros emitted {} tokens (no validation)",
        tokens.len()
    );
}

#[test]
fn end_to_end_transcribe_real_japanese_audio() {
    let path = match pick_fixture() {
        Some(p) => p,
        None => {
            eprintln!("SKIPPED: no reazonspeech GGUF found");
            return;
        }
    };
    let wav_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../third-party/whisper.cpp/samples/japanese_test.wav");
    if !wav_path.exists() {
        eprintln!("SKIPPED: japanese_test.wav not found at {}", wav_path.display());
        return;
    }

    let audio = read_wav_16k_mono(&wav_path).expect("wav read");
    eprintln!(
        "loaded {} samples = {:.2}s of audio",
        audio.len(),
        audio.len() as f32 / 16000.0
    );

    use any_stt::{Backend, SttEngine};
    let hw = any_stt::detect_hardware();
    let engine = reazonspeech_backend::ReazonSpeechEngine::new(&path, "ja", Backend::Cpu, hw)
        .expect("engine new");

    let start = std::time::Instant::now();
    let result = engine.transcribe(&audio).expect("transcribe");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    eprintln!("=== ReazonSpeech transcription result ===");
    eprintln!("  text:        {:?}", result.text);
    eprintln!("  language:    {}", result.language);
    eprintln!("  internal ms: {:.1}", result.duration_ms);
    eprintln!("  wallclock:   {:.0}ms", elapsed_ms);
    eprintln!("  audio:       {:.2}s", audio.len() as f32 / 16000.0);
    eprintln!(
        "  RTF:         {:.3}",
        elapsed_ms / 1000.0 / (audio.len() as f32 / 16000.0) as f64
    );

    // No accuracy assertion — numerical correctness is in #N9. Just
    // verify the pipeline runs to completion and yields a string.
    assert!(result.duration_ms >= 0.0);
}

fn read_wav_16k_mono(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{e}"))?;
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a WAV".into());
    }
    let mut pos = 12;
    let mut fmt = None;
    let mut data = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let body = &bytes[pos + 8..pos + 8 + sz];
        if id == b"fmt " {
            fmt = Some(body);
        } else if id == b"data" {
            data = Some(body);
        }
        pos = pos + 8 + sz + (sz & 1);
    }
    let fmt = fmt.ok_or("no fmt")?;
    let data = data.ok_or("no data")?;
    let bps = u16::from_le_bytes([fmt[14], fmt[15]]);
    let mut out = Vec::with_capacity(data.len() / (bps as usize / 8));
    for i in (0..data.len()).step_by(2) {
        out.push(i16::from_le_bytes([data[i], data[i + 1]]) as f32 / 32768.0);
    }
    Ok(out)
}
