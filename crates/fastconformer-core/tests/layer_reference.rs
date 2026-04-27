//! Compare fastconformer-core outputs against Python NeMo reference
//! tensors produced by `scripts/dump-nemo-fixtures.py`.
//!
//! Each test is `#[ignore]` by default — they only run when the fixture
//! files are present at `tests/fixtures/<model_id>/`. Generate fixtures
//! with the Python script (heavy: ~3 GB for nemo-toolkit), then run:
//!
//!     cargo test -p fastconformer-core --test layer_reference -- --ignored
//!
//! When a real model is wired up via `dump-nemo-fixtures.py`, drop the
//! `#[ignore]` attribute on the relevant tests.

use std::fs;
use std::path::{Path, PathBuf};

use fastconformer_core::{log_mel_spectrogram, Config};

const MODEL_ID: &str = "reazonspeech-nemo-v2";
const SAMPLE_ID: &str = "ja";

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(MODEL_ID)
}

fn fixture_path(name: &str) -> PathBuf {
    fixtures_root().join(format!("{name}.npy"))
}

fn fixtures_present() -> bool {
    fixture_path(&format!("mel_{SAMPLE_ID}")).exists()
}

// ---------------------------------------------------------------------------
// Minimal .npy reader. Only handles 1-D / 2-D f32 arrays in C-order, which
// is all `dump-nemo-fixtures.py` writes.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct NpyArray {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl NpyArray {
    /// Load a `.npy` v1.0 file containing a `<f4` (little-endian f32) array
    /// in C-order.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes =
            fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::parse(&bytes)
    }

    fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 10 {
            return Err("npy: too short".into());
        }
        if &bytes[..6] != b"\x93NUMPY" {
            return Err("npy: bad magic".into());
        }
        let major = bytes[6];
        let minor = bytes[7];
        let header_len_offset = 8;
        let (header_len, header_start) = match (major, minor) {
            (1, 0) => (
                u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
                header_len_offset + 2,
            ),
            (2, 0) | (3, 0) => (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                header_len_offset + 4,
            ),
            _ => return Err(format!("npy: unsupported version {major}.{minor}")),
        };
        let header_end = header_start + header_len;
        if bytes.len() < header_end {
            return Err("npy: header truncated".into());
        }
        let header = std::str::from_utf8(&bytes[header_start..header_end])
            .map_err(|e| format!("npy: header not utf-8: {e}"))?;

        // Header looks like:
        //   {'descr': '<f4', 'fortran_order': False, 'shape': (12, 3), }
        if !header.contains("'<f4'") && !header.contains("\"<f4\"") {
            return Err(format!("npy: only <f4 supported, header={header}"));
        }
        if header.contains("True") {
            return Err("npy: fortran_order=True not supported".into());
        }
        let shape = parse_shape(header)?;
        let n_elems: usize = shape.iter().product();
        let body = &bytes[header_end..];
        if body.len() < n_elems * 4 {
            return Err(format!(
                "npy: body has {} bytes, expected {}",
                body.len(),
                n_elems * 4
            ));
        }
        let mut data = Vec::with_capacity(n_elems);
        for i in 0..n_elems {
            let b = &body[i * 4..i * 4 + 4];
            data.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
        Ok(Self { data, shape })
    }

    pub fn n_elements(&self) -> usize {
        self.data.len()
    }
}

fn parse_shape(header: &str) -> Result<Vec<usize>, String> {
    let key = "'shape':";
    let i = header
        .find(key)
        .ok_or_else(|| format!("npy: 'shape' missing in header: {header}"))?;
    let rest = &header[i + key.len()..];
    let open = rest
        .find('(')
        .ok_or_else(|| "npy: shape '(' missing".to_string())?;
    let close = rest
        .find(')')
        .ok_or_else(|| "npy: shape ')' missing".to_string())?;
    let inside = &rest[open + 1..close];
    let mut dims = Vec::new();
    for tok in inside.split(',') {
        let s = tok.trim();
        if s.is_empty() {
            continue;
        }
        dims.push(
            s.parse::<usize>()
                .map_err(|e| format!("npy: bad shape token {s:?}: {e}"))?,
        );
    }
    Ok(dims)
}

/// Compare two flat f32 buffers element-wise. Returns the maximum absolute
/// difference and the max relative difference (skipping zero references).
fn max_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        let denom = x.abs().max(y.abs());
        if denom > 1e-6 {
            let r = d / denom;
            if r > max_rel {
                max_rel = r;
            }
        }
    }
    (max_abs, max_rel)
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn npy_parser_roundtrip_synthetic() {
    // Build a tiny v1.0 npy in memory (matches numpy.save output for a
    // 1-D f32 array of length 3 = [1.0, 2.0, 3.0]).
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.push(1);
    bytes.push(0);
    let header_str = "{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }";
    let mut padded = header_str.to_string();
    // numpy pads to a 16-byte boundary, but our parser doesn't care as
    // long as the declared length matches.
    while (10 + padded.len()) % 16 != 0 {
        padded.push(' ');
    }
    let header_len = padded.len() as u16;
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(padded.as_bytes());
    for v in [1.0_f32, 2.0, 3.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let a = NpyArray::parse(&bytes).unwrap();
    assert_eq!(a.shape, vec![3]);
    assert_eq!(a.data, vec![1.0, 2.0, 3.0]);
}

#[test]
#[ignore = "requires reazonspeech .nemo + nemo-toolkit; run after dump-nemo-fixtures.py"]
fn mel_matches_nemo_reference() {
    if !fixtures_present() {
        eprintln!(
            "SKIPPED: no fixtures at {}. Generate with:\n  \
             python scripts/dump-nemo-fixtures.py --model-path <model.nemo> \\\n  \
             --model-id {MODEL_ID} --audio <jp.wav> --sample-id {SAMPLE_ID} \\\n  \
             --out-dir crates/fastconformer-core/tests/fixtures",
            fixtures_root().display(),
        );
        return;
    }

    // Reference mel from NeMo.
    let ref_mel = NpyArray::load(&fixture_path(&format!("mel_{SAMPLE_ID}"))).unwrap();
    assert_eq!(ref_mel.shape.len(), 2, "expected 2-D mel, got {:?}", ref_mel.shape);
    let n_frames = ref_mel.shape[0];
    let n_mels = ref_mel.shape[1];

    // Re-run our preprocessor on the same audio.
    let audio_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../third-party/whisper.cpp/samples/japanese_test.wav");
    if !audio_path.exists() {
        eprintln!("SKIPPED: audio sample not found at {}", audio_path.display());
        return;
    }
    let audio = read_wav_16k_mono(&audio_path).unwrap();
    let cfg = Config::dummy_reazonspeech_nemo_v2();
    let mel = log_mel_spectrogram(&audio, &cfg);

    assert_eq!(mel.n_frames, n_frames, "frame count differs");
    assert_eq!(mel.n_mels, n_mels, "mel bins differ");

    let (max_abs, max_rel) = max_diff(&mel.data, &ref_mel.data);
    eprintln!("mel max_abs={max_abs:.6}, max_rel={max_rel:.6}");
    // Loose tolerance for the first wire-up — tighten once the encoder
    // path is also verified.
    assert!(max_abs < 1e-2, "mel max_abs={max_abs} exceeds 1e-2");
}

fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>, String> {
    // Lightweight WAV reader for tests — avoids pulling `hound` into
    // fastconformer-core.
    let bytes = fs::read(path).map_err(|e| format!("{e}"))?;
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    // Find "fmt " and "data" chunks.
    let mut pos = 12;
    let mut fmt = None;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let body_start = pos + 8;
        let body_end = body_start + sz;
        if id == b"fmt " {
            fmt = Some(&bytes[body_start..body_end]);
        } else if id == b"data" {
            data = Some(&bytes[body_start..body_end]);
        }
        pos = body_end + (sz & 1);
    }
    let fmt = fmt.ok_or_else(|| "missing fmt chunk".to_string())?;
    let data = data.ok_or_else(|| "missing data chunk".to_string())?;
    let audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
    let channels = u16::from_le_bytes([fmt[2], fmt[3]]);
    let sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
    let bps = u16::from_le_bytes([fmt[14], fmt[15]]);
    if channels != 1 {
        return Err("expected mono".into());
    }
    if sample_rate != 16000 {
        return Err(format!("expected 16 kHz, got {sample_rate}"));
    }
    if audio_format == 1 && bps == 16 {
        let mut out = Vec::with_capacity(data.len() / 2);
        for i in (0..data.len()).step_by(2) {
            let s = i16::from_le_bytes([data[i], data[i + 1]]);
            out.push(s as f32 / 32768.0);
        }
        Ok(out)
    } else {
        Err(format!("unsupported wav fmt={audio_format} bps={bps}"))
    }
}
