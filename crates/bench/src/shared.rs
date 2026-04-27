//! Utilities shared by both the accuracy driver and the legacy
//! whisper-internals bench modes.

use std::path::Path;

/// Load a 16 kHz mono WAV file as f32 PCM in [-1, 1].
///
/// Panics if the file is missing, not mono, or not 16 kHz — bench is a
/// developer tool, not a library, so fail-fast is appropriate.
pub fn load_audio(path: &Path) -> Vec<f32> {
    let reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected mono audio: {}", path.display());
    assert_eq!(
        spec.sample_rate, 16000,
        "expected 16kHz sample rate: {}",
        path.display()
    );
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
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

/// Number of seconds of audio at 16 kHz.
pub fn audio_duration_secs(samples: &[f32]) -> f64 {
    samples.len() as f64 / 16000.0
}

/// Median of a slice. `values` must be non-empty; the slice will be sorted
/// in place. Used by legacy whisper-internals bench code.
pub fn median(values: &mut [f64]) -> f64 {
    assert!(!values.is_empty(), "median of empty slice");
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}
