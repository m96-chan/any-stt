//! Log-mel filterbank feature extraction matching NeMo's
//! `AudioToMelSpectrogramPreprocessor` (the preprocessor shipped inside
//! reazonspeech-nemo-v2 and parakeet-tdt-0.6b-v3).
//!
//! Pipeline per call:
//! ```text
//!   audio (f32 PCM) ──► pre-emphasis (α=0.97)
//!                  ──► hann window
//!                  ──► rfft  (n_fft = next_pow2(win_length))
//!                  ──► |X|²  (power spectrum)
//!                  ──► mel filterbank  (slaney scale, 80 filters)
//!                  ──► log(mag + 1e-5)
//!                  ──► per-feature normalize  (zero mean / unit stddev across time)
//!                  ──► [n_frames, n_mels]
//! ```
//!
//! Reference: NeMo `AudioToMelSpectrogramPreprocessor` defaults
//! (sample_rate=16000, window_size=0.025, window_stride=0.010,
//! features=80, n_fft=512, preemph=0.97, log_zero_guard=1e-5,
//! normalize="per_feature", mag_power=2.0, dither=0.0).

use std::f32::consts::PI;

use realfft::RealFftPlanner;

use crate::config::Config;

/// Mel spectrogram in row-major `[n_frames, n_mels]` layout.
pub struct MelSpectrogram {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

impl MelSpectrogram {
    /// Index helper: `mel[(t, m)]` returns the value at frame `t`, mel `m`.
    pub fn at(&self, frame: usize, mel: usize) -> f32 {
        self.data[frame * self.n_mels + mel]
    }
}

/// Pre-emphasis coefficient used by NeMo (`preemph` default).
const PREEMPH: f32 = 0.97;
/// Log compression floor used by NeMo (`log_zero_guard_value` default).
const LOG_ZERO_GUARD: f32 = 1e-5;

/// Compute log-mel filterbank features for `audio` using `cfg`.
///
/// Audio is assumed to be 16 kHz mono f32 in [-1, 1]. Out-of-range values
/// are not clamped — that is the caller's responsibility.
pub fn log_mel_spectrogram(audio: &[f32], cfg: &Config) -> MelSpectrogram {
    assert_eq!(
        cfg.sample_rate, 16000,
        "only 16 kHz sample rate is wired; got {}",
        cfg.sample_rate,
    );
    let win_length = cfg.win_length as usize;
    let hop_length = cfg.hop_length as usize;
    let n_mels = cfg.n_mels as usize;
    let n_fft = win_length.next_power_of_two().max(2);

    if audio.len() < win_length {
        // Degenerate input — emit one frame of -log(LOG_ZERO_GUARD), which
        // is what an all-zero magnitude would normalize to. Keeps callers
        // from having to special-case empty / very-short audio.
        return MelSpectrogram {
            data: vec![0.0; n_mels],
            n_frames: 1,
            n_mels,
        };
    }

    // 1. Pre-emphasis: y[n] = x[n] - α * x[n-1], with y[0] = x[0].
    let mut emph = Vec::with_capacity(audio.len());
    emph.push(audio[0]);
    for i in 1..audio.len() {
        emph.push(audio[i] - PREEMPH * audio[i - 1]);
    }

    // 2. Center-pad with reflection by n_fft / 2 on each side. This
    //    matches torch.stft(center=True, pad_mode="reflect"), which is
    //    NeMo's default. Without it we lose the first and last 2-3
    //    frames vs the Python reference.
    let pad = n_fft / 2;
    let mut padded: Vec<f32> = Vec::with_capacity(emph.len() + 2 * pad);
    for i in (1..=pad).rev() {
        padded.push(emph[i.min(emph.len() - 1)]);
    }
    padded.extend_from_slice(&emph);
    let n = emph.len();
    for i in 1..=pad {
        let idx = n.saturating_sub(1 + i);
        padded.push(emph[idx]);
    }
    let emph = padded;

    // 3. Hann window of length win_length, padded into n_fft-sized buffer.
    let hann = hann_window(win_length);

    // 4. Frame the signal and run rFFT. With center=True semantics the
    //    frame count formula matches torch.stft: floor(N / hop) + 1.
    let n_frames = if emph.len() < n_fft {
        1
    } else {
        (emph.len() - n_fft) / hop_length + 1
    };
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut fft_in = vec![0.0_f32; n_fft];
    let mut fft_out = vec![realfft::num_complex::Complex::<f32>::new(0.0, 0.0); n_fft / 2 + 1];

    let mel_fb = mel_filterbank(n_mels, n_fft, cfg.sample_rate as f32);

    let mut log_mel = vec![0.0_f32; n_frames * n_mels];

    for t in 0..n_frames {
        let start = t * hop_length;
        // torch.stft with win_length < n_fft pads the window with
        // trailing zeros (window at fft_in[0..win_length], zeros after).
        for v in fft_in.iter_mut() {
            *v = 0.0;
        }
        for i in 0..win_length {
            let src = start + i;
            if src < emph.len() {
                fft_in[i] = emph[src] * hann[i];
            }
        }

        fft.process(&mut fft_in, &mut fft_out)
            .expect("rfft length mismatch");

        // Power spectrum |X|².
        // realfft returns n_fft/2 + 1 bins.
        let n_bins = n_fft / 2 + 1;
        let mut power = vec![0.0_f32; n_bins];
        for i in 0..n_bins {
            let c = fft_out[i];
            power[i] = c.re * c.re + c.im * c.im;
        }

        // Mel projection: mel[m] = Σ_b filter[m][b] * power[b].
        let row = &mut log_mel[t * n_mels..(t + 1) * n_mels];
        for m in 0..n_mels {
            let filter = &mel_fb[m];
            let mut acc = 0.0_f32;
            for (b, &w) in filter.iter().enumerate() {
                acc += w * power[b];
            }
            // Log compression with NeMo's zero guard.
            row[m] = (acc + LOG_ZERO_GUARD).ln();
        }
    }

    // 4. Per-feature normalization across time.
    normalize_per_feature(&mut log_mel, n_frames, n_mels);

    MelSpectrogram {
        data: log_mel,
        n_frames,
        n_mels,
    }
}

/// Hann window of given length, NeMo-compatible (`periodic=True` —
/// torch.stft / torch.hann_window default for spectrogram use). Formula:
///     w[i] = 0.5 * (1 - cos(2π · i / n))   for i in 0..n
fn hann_window(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = n as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / denom).cos()))
        .collect()
}

/// Mel-filterbank matrix, slaney scale (matches librosa default + NeMo).
///
/// Returns `n_mels` filters, each of length `n_fft / 2 + 1`. Each filter
/// is a triangle in mel space, area-normalized via Slaney convention
/// (`enorm = 2 / (high_hz - low_hz)`). The resulting matrix matches
/// `torchaudio.functional.melscale_fbanks(norm="slaney", mel_scale="slaney")`
/// up to float epsilon.
pub fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: f32) -> Vec<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    let f_min = 0.0_f32;
    let f_max = sample_rate / 2.0;

    let mel_min = hz_to_mel_slaney(f_min);
    let mel_max = hz_to_mel_slaney(f_max);

    // n_mels + 2 mel points (lower + n_mels triangles + upper).
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();
    // FFT bin frequencies.
    let bin_freqs: Vec<f32> = (0..n_bins)
        .map(|b| b as f32 * sample_rate / n_fft as f32)
        .collect();

    let mut filters = vec![vec![0.0_f32; n_bins]; n_mels];
    for m in 0..n_mels {
        let lo = hz_points[m];
        let mid = hz_points[m + 1];
        let hi = hz_points[m + 2];
        let inv_l = 1.0 / (mid - lo).max(1e-9);
        let inv_r = 1.0 / (hi - mid).max(1e-9);
        // Slaney area normalization: weight = 2 / (hi - lo).
        let enorm = 2.0 / (hi - lo).max(1e-9);
        for (b, &f) in bin_freqs.iter().enumerate() {
            let w = if f < lo || f > hi {
                0.0
            } else if f <= mid {
                (f - lo) * inv_l
            } else {
                (hi - f) * inv_r
            };
            filters[m][b] = w * enorm;
        }
    }
    filters
}

/// Slaney-style hz → mel.
fn hz_to_mel_slaney(hz: f32) -> f32 {
    // Below 1000 Hz: linear. Above: log.
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f32).ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        (hz - F_MIN) / F_SP
    }
}

/// Slaney-style mel → hz (inverse of `hz_to_mel_slaney`).
fn mel_to_hz_slaney(mel: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f32).ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * logstep).exp()
    } else {
        F_MIN + F_SP * mel
    }
}

/// Per-feature normalization across time: zero mean, unit stddev for each
/// mel bin independently. Matches NeMo `normalize="per_feature"` with
/// epsilon=1e-5.
fn normalize_per_feature(data: &mut [f32], n_frames: usize, n_mels: usize) {
    if n_frames == 0 {
        return;
    }
    // Accumulate in f64 to keep mean rounding error below the 1e-5 epsilon
    // floor — otherwise constant-input edge cases (silence) blow up
    // because (data - mean) ≠ 0 in f32 but std² ≈ 0, amplifying the
    // residual by ~1/eps.
    let inv_n = 1.0_f64 / n_frames as f64;
    let mut mean = vec![0.0_f64; n_mels];
    for t in 0..n_frames {
        for m in 0..n_mels {
            mean[m] += data[t * n_mels + m] as f64;
        }
    }
    for v in &mut mean {
        *v *= inv_n;
    }
    let mut var = vec![0.0_f64; n_mels];
    for t in 0..n_frames {
        for m in 0..n_mels {
            let d = data[t * n_mels + m] as f64 - mean[m];
            var[m] += d * d;
        }
    }
    // NeMo formula: std = sqrt(var) + 1e-5 (additive epsilon, not clamp).
    let std: Vec<f64> = var.iter().map(|&v| (v * inv_n).sqrt() + 1e-5).collect();
    for t in 0..n_frames {
        for m in 0..n_mels {
            let i = t * n_mels + m;
            data[i] = ((data[i] as f64 - mean[m]) / std[m]) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_matches_formula() {
        let cfg = Config::dummy_reazonspeech_nemo_v2();
        // 1s @ 16kHz, n_fft=512, hop=160. With NeMo's center=True
        // (reflect-pad by n_fft/2=256 each side):
        //   padded len = 16000 + 512 = 16512
        //   frames = (16512 - 512) / 160 + 1 = 101
        let audio = vec![0.0_f32; 16000];
        let mel = log_mel_spectrogram(&audio, &cfg);
        assert_eq!(mel.n_frames, 101);
        assert_eq!(mel.n_mels, 80);
        assert_eq!(mel.data.len(), 101 * 80);
    }

    #[test]
    fn short_audio_returns_single_frame() {
        let cfg = Config::dummy_reazonspeech_nemo_v2();
        let mel = log_mel_spectrogram(&[0.0_f32; 100], &cfg);
        assert_eq!(mel.n_frames, 1);
        assert_eq!(mel.n_mels, 80);
    }

    #[test]
    fn silent_audio_yields_finite_normalized_output() {
        let cfg = Config::dummy_reazonspeech_nemo_v2();
        let audio = vec![0.0_f32; 16000];
        let mel = log_mel_spectrogram(&audio, &cfg);
        // After per-feature normalization of constant input, each row is
        // exactly 0 (mean subtracted; variance is 0 → epsilon-clamped div).
        for &v in &mel.data {
            assert!(v.is_finite(), "got non-finite value");
            assert!(v.abs() < 1e-3, "expected near-zero, got {v}");
        }
    }

    #[test]
    fn sine_wave_peaks_at_expected_mel_bin() {
        // 1 kHz sine wave for 1s at 16 kHz. The mel filter that covers
        // 1 kHz should have the highest pre-normalization magnitude. We
        // verify by skipping the normalize step (constant input across
        // time means normalize zeros everything) — instead, run the
        // transform and inspect the *first* frame's pre-normalize energy
        // by re-running without normalize.
        let cfg = Config::dummy_reazonspeech_nemo_v2();
        let n = 16000;
        let freq = 1000.0_f32;
        let audio: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / 16000.0).sin())
            .collect();

        // Take a single frame and extract mel without time-normalization.
        let win = cfg.win_length as usize;
        let n_fft = win.next_power_of_two();
        let mut emph = vec![0.0_f32; win];
        emph[0] = audio[0];
        for i in 1..win {
            emph[i] = audio[i] - PREEMPH * audio[i - 1];
        }
        let hann = hann_window(win);
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let mut fft_in = vec![0.0_f32; n_fft];
        let mut fft_out =
            vec![realfft::num_complex::Complex::<f32>::new(0.0, 0.0); n_fft / 2 + 1];
        for i in 0..win {
            fft_in[i] = emph[i] * hann[i];
        }
        fft.process(&mut fft_in, &mut fft_out).unwrap();
        let mut power = vec![0.0_f32; n_fft / 2 + 1];
        for (i, c) in fft_out.iter().enumerate() {
            power[i] = c.re * c.re + c.im * c.im;
        }
        let mel_fb = mel_filterbank(80, n_fft, 16000.0);
        let mel_energies: Vec<f32> = mel_fb
            .iter()
            .map(|f| f.iter().zip(power.iter()).map(|(a, b)| a * b).sum())
            .collect();
        let argmax = mel_energies
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // Slaney mel scale: hz_to_mel(1000) = 15, hz_to_mel(8000) ≈ 45.21.
        // With 80 filters the 1 kHz center lands at filter ~27. Allow a
        // window for FFT leakage and filter triangle overlap.
        assert!(
            (20..=35).contains(&argmax),
            "1 kHz sine should peak at mel bin ~27, got {argmax}",
        );
    }

    #[test]
    fn hann_window_sums_to_n_over_2() {
        // ∑ hann(n) = (N-1)/2 * 1.0 ≈ N/2 for large N (periodic=False form).
        let w = hann_window(400);
        let s: f32 = w.iter().sum();
        // Allow 1% relative tolerance (formula gives (N-1)/2 = 199.5).
        assert!((s - 199.5).abs() < 2.0, "hann sum was {s}");
    }

    #[test]
    fn mel_to_hz_inverse_is_consistent() {
        for &hz in &[0.0_f32, 100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0] {
            let m = hz_to_mel_slaney(hz);
            let back = mel_to_hz_slaney(m);
            assert!(
                (hz - back).abs() < 1e-2,
                "round-trip failed at {hz}: -> {m} -> {back}"
            );
        }
    }

    #[test]
    fn normalize_per_feature_centers_each_mel() {
        let n_frames = 4;
        let n_mels = 3;
        // mel 0: 1, 2, 3, 4 (mean 2.5)
        // mel 1: 10, 10, 10, 10 (constant)
        // mel 2: -5, 5, -5, 5 (mean 0, std 5)
        let mut data: Vec<f32> = vec![
            1.0, 10.0, -5.0,
            2.0, 10.0, 5.0,
            3.0, 10.0, -5.0,
            4.0, 10.0, 5.0,
        ];
        normalize_per_feature(&mut data, n_frames, n_mels);
        // Each mel column should now have ~zero mean.
        for m in 0..n_mels {
            let mean: f32 = (0..n_frames).map(|t| data[t * n_mels + m]).sum::<f32>()
                / n_frames as f32;
            assert!(mean.abs() < 1e-5, "mel {m} mean={mean}");
        }
    }
}
