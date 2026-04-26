//! Manifest format for accuracy benchmarks.
//!
//! A manifest is a JSON document describing a dataset of (audio, reference
//! transcript) pairs. It is engine-agnostic — the same manifest feeds every
//! backend (whisper-backend, reazonspeech-backend, parakeet-backend, ...).
//!
//! Example:
//! ```json
//! {
//!   "name": "any-stt default accuracy bench",
//!   "sample_rate": 16000,
//!   "items": [
//!     {
//!       "id": "jfk_en",
//!       "audio": "third-party/whisper.cpp/samples/jfk.wav",
//!       "language": "en",
//!       "reference": "And so, my fellow Americans..."
//!     }
//!   ]
//! }
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Top-level manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Human-readable name (appears in report output).
    pub name: String,
    /// Audio sample rate expected by all items. Default: 16000 Hz.
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    /// Dataset entries.
    pub items: Vec<Item>,
}

/// A single (audio, reference) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    /// Short identifier used in reports (e.g. "jfk_en", "wagahai_ja").
    pub id: String,
    /// Path to the audio file. May be absolute or relative to the manifest.
    pub audio: PathBuf,
    /// BCP-47 language code ("en", "ja", ...) passed to the backend.
    pub language: String,
    /// Ground-truth transcript.
    pub reference: String,
}

fn default_sample_rate() -> u32 {
    16000
}

impl Manifest {
    /// Parse a manifest from a JSON file. Relative `audio` paths in the
    /// manifest are resolved against the manifest's parent directory.
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let content = fs::read_to_string(path).map_err(|e| ManifestError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut manifest: Manifest = serde_json::from_str(&content)?;

        let base = path.parent().unwrap_or(Path::new("."));
        for item in &mut manifest.items {
            if item.audio.is_relative() {
                item.audio = base.join(&item.audio);
            }
        }

        Ok(manifest)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("failed to read manifest {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse manifest JSON: {0}")]
    Parse(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_minimal_manifest() {
        let json = r#"{
            "name": "t",
            "items": [
                {"id": "a", "audio": "a.wav", "language": "en", "reference": "hello"}
            ]
        }"#;
        let m: Manifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.name, "t");
        assert_eq!(m.sample_rate, 16000); // default
        assert_eq!(m.items.len(), 1);
    }

    #[test]
    fn load_resolves_relative_audio_paths() {
        // Write a tmp manifest + dummy audio file in the same dir.
        let dir = std::env::temp_dir().join("bench_manifest_test");
        fs::create_dir_all(&dir).unwrap();
        let manifest_path = dir.join("m.json");
        let mut f = fs::File::create(&manifest_path).unwrap();
        writeln!(
            f,
            r#"{{
                "name": "t",
                "items": [
                    {{"id": "a", "audio": "sub/foo.wav", "language": "en", "reference": "x"}}
                ]
            }}"#
        )
        .unwrap();
        drop(f);

        let m = Manifest::load(&manifest_path).unwrap();
        assert_eq!(m.items[0].audio, dir.join("sub/foo.wav"));
    }

    #[test]
    fn missing_file_returns_io_error() {
        let err = Manifest::load(Path::new("/does/not/exist.json")).unwrap_err();
        assert!(matches!(err, ManifestError::Io { .. }));
    }
}
