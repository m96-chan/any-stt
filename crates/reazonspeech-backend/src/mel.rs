//! Log-mel filterbank feature extraction for FastConformer input.
//!
//! NeMo FastConformer preprocessing is subtly different from Whisper:
//!   - NeMo uses `AudioToMelSpectrogramPreprocessor` with:
//!     - n_window_size (samples) = 0.025 * sr = 400 @ 16kHz
//!     - n_window_stride (samples) = 0.010 * sr = 160 @ 16kHz
//!     - n_fft = next_pow2(n_window_size) = 512
//!     - features = 80 mel filters
//!     - log scale with +1e-5 offset
//!     - per-feature normalization (zero mean, unit stddev across time)
//!   - Whisper uses 80 filters but different log compression (log10 clamp 1e-10)
//!     and global normalization across time×mel.
//!
//! To match Python reference outputs within ~1e-4, this file must implement
//! the NeMo variant exactly. See `tests/mel_reference.rs` (TODO — depends
//! on `scripts/dump-nemo-fixtures.py` producing reference mel spectrograms).
//!
//! ## Status
//! Skeleton with the public API defined but the compute marked TODO.

use crate::config::ReazonSpeechConfig;

/// Mel spectrogram: `[n_frames, n_mels]` row-major f32.
pub struct MelSpectrogram {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

/// Compute NeMo-style log-mel filterbank features.
///
/// # Returns
/// A `MelSpectrogram` with `n_frames = (len(audio) - win_length) / hop_length + 1`
/// and `n_mels` as configured. Data layout is row-major.
pub fn log_mel_spectrogram(audio: &[f32], cfg: &ReazonSpeechConfig) -> MelSpectrogram {
    assert_eq!(
        cfg.sample_rate, 16000,
        "only 16kHz sample_rate is supported"
    );
    let n_fft = cfg.win_length.next_power_of_two() as usize;
    let win_length = cfg.win_length as usize;
    let hop_length = cfg.hop_length as usize;
    let n_mels = cfg.n_mels as usize;

    // TODO(#N4): implement
    //   1. Pre-emphasis filter (α=0.97 per NeMo default)
    //   2. Frame the signal (hann window, win_length, hop_length)
    //   3. FFT → |X|²
    //   4. Mel filterbank projection (precomputed matrix, `build_mel_matrix`)
    //   5. log(mag + 1e-5)
    //   6. Per-feature normalization
    let n_frames = if audio.len() < win_length {
        1
    } else {
        (audio.len() - win_length) / hop_length + 1
    };
    let _ = n_fft;

    MelSpectrogram {
        data: vec![0.0; n_frames * n_mels],
        n_frames,
        n_mels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_matches_formula() {
        let cfg = ReazonSpeechConfig::dummy();
        // 1 second @ 16kHz = 16000 samples
        // win=400, hop=160 → (16000 - 400) / 160 + 1 = 98 frames
        let audio = vec![0.0f32; 16000];
        let mel = log_mel_spectrogram(&audio, &cfg);
        assert_eq!(mel.n_frames, 98);
        assert_eq!(mel.n_mels, 80);
        assert_eq!(mel.data.len(), 98 * 80);
    }

    #[test]
    fn short_audio_returns_single_frame() {
        let cfg = ReazonSpeechConfig::dummy();
        // Less than win_length → 1 frame (edge case)
        let audio = vec![0.0f32; 100];
        let mel = log_mel_spectrogram(&audio, &cfg);
        assert_eq!(mel.n_frames, 1);
    }
}
