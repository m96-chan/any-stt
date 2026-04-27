//! FastConformer model configuration parsed from GGUF metadata.
//!
//! All keys are written by `scripts/convert-nemo-to-gguf.py` under the
//! `fastconformer.*` namespace. This module is family-agnostic — it
//! accepts both `rel_pos` and `rel_pos_local_attn` (Longformer) attention
//! types, and both `rnnt` and `tdt` decoder types. Per-family crates check
//! the type they expect.

use gguf_loader::GgufFile;
use thiserror::Error;

/// Parsed FastConformer model config (encoder + decoder + audio).
#[derive(Debug, Clone)]
pub struct Config {
    // --- Encoder ---
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub feat_in: u32,
    pub conv_kernel_size: u32,
    pub subsampling_factor: u32,
    pub attention_type: AttentionType,
    /// Longformer local window size (one side). Zero for plain rel-pos.
    pub local_window: u32,
    /// Number of global tokens. Typically 0 (rel-pos) or 1 (Longformer).
    pub global_tokens: u32,

    // --- Decoder ---
    pub decoder_type: DecoderType,
    pub vocab_size: u32,
    pub pred_hidden: u32,
    pub joint_hidden: u32,
    pub blank_id: u32,

    // --- Audio preprocessing ---
    pub sample_rate: u32,
    pub win_length: u32,
    pub hop_length: u32,
    pub n_mels: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Standard relative-position multi-head attention.
    /// Used by parakeet-tdt-0.6b-v3.
    RelPos,
    /// Longformer-style: local sliding window + global attention tokens.
    /// Used by reazonspeech-nemo-v2 with window=256, global=1.
    RelPosLocalAttn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderType {
    /// Standard RNN-T (reazonspeech).
    Rnnt,
    /// Token-Duration Transducer (parakeet).
    Tdt,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("missing GGUF metadata key: {0}")]
    MissingKey(String),
    #[error("invalid value for {key}: {value}")]
    InvalidValue { key: String, value: String },
    #[error("expected general.architecture='fastconformer', got {0:?}")]
    WrongArchitecture(String),
}

impl Config {
    /// Parse the config from a GGUF file's metadata block.
    /// Does not load any tensor data.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, ConfigError> {
        let arch = gguf
            .meta_str("general.architecture")
            .ok_or_else(|| ConfigError::MissingKey("general.architecture".into()))?;
        if arch != "fastconformer" {
            return Err(ConfigError::WrongArchitecture(arch.to_string()));
        }

        let attention_type = match gguf.meta_str("fastconformer.encoder.attention_type") {
            Some("rel_pos_local_attn") => AttentionType::RelPosLocalAttn,
            Some("rel_pos") | None => AttentionType::RelPos,
            Some(other) => {
                return Err(ConfigError::InvalidValue {
                    key: "fastconformer.encoder.attention_type".into(),
                    value: other.into(),
                });
            }
        };

        let decoder_type = match gguf.meta_str("fastconformer.decoder.type") {
            Some("rnnt") => DecoderType::Rnnt,
            Some("tdt") => DecoderType::Tdt,
            Some(other) => {
                return Err(ConfigError::InvalidValue {
                    key: "fastconformer.decoder.type".into(),
                    value: other.into(),
                });
            }
            None => return Err(ConfigError::MissingKey("fastconformer.decoder.type".into())),
        };

        Ok(Self {
            n_layers: u32_meta(gguf, "fastconformer.encoder.n_layers")?,
            d_model: u32_meta(gguf, "fastconformer.encoder.d_model")?,
            n_heads: u32_meta(gguf, "fastconformer.encoder.n_heads")?,
            feat_in: u32_meta(gguf, "fastconformer.encoder.feat_in")?,
            conv_kernel_size: u32_meta(gguf, "fastconformer.encoder.conv_kernel_size")?,
            subsampling_factor: u32_meta(gguf, "fastconformer.encoder.subsampling_factor")?,
            attention_type,
            local_window: u32_meta(gguf, "fastconformer.encoder.local_window").unwrap_or(0),
            global_tokens: u32_meta(gguf, "fastconformer.encoder.global_tokens").unwrap_or(0),
            decoder_type,
            vocab_size: u32_meta(gguf, "fastconformer.decoder.vocab_size")?,
            pred_hidden: u32_meta(gguf, "fastconformer.decoder.pred_hidden")?,
            joint_hidden: u32_meta(gguf, "fastconformer.decoder.joint_hidden")?,
            blank_id: u32_meta(gguf, "fastconformer.decoder.blank_id")?,
            sample_rate: u32_meta(gguf, "fastconformer.audio.sample_rate")?,
            win_length: u32_meta(gguf, "fastconformer.audio.win_length")?,
            hop_length: u32_meta(gguf, "fastconformer.audio.hop_length")?,
            n_mels: u32_meta(gguf, "fastconformer.audio.n_mels")?,
        })
    }

    /// Reasonable in-memory default for unit tests that don't have a real
    /// GGUF on hand. Matches reazonspeech-nemo-v2 dimensions.
    #[doc(hidden)]
    pub fn dummy_reazonspeech_nemo_v2() -> Self {
        Self {
            n_layers: 17,
            d_model: 512,
            n_heads: 8,
            feat_in: 80,
            conv_kernel_size: 9,
            subsampling_factor: 8,
            attention_type: AttentionType::RelPosLocalAttn,
            local_window: 256,
            global_tokens: 1,
            decoder_type: DecoderType::Rnnt,
            vocab_size: 3000,
            pred_hidden: 640,
            joint_hidden: 640,
            blank_id: 2999,
            sample_rate: 16000,
            win_length: 400,
            hop_length: 160,
            n_mels: 80,
        }
    }

    /// Default matching parakeet-tdt-0.6b-v3.
    #[doc(hidden)]
    pub fn dummy_parakeet_tdt_v3() -> Self {
        Self {
            n_layers: 24,
            d_model: 1024,
            n_heads: 8,
            feat_in: 80,
            conv_kernel_size: 9,
            subsampling_factor: 8,
            attention_type: AttentionType::RelPos,
            local_window: 0,
            global_tokens: 0,
            decoder_type: DecoderType::Tdt,
            vocab_size: 8192,
            pred_hidden: 640,
            joint_hidden: 640,
            blank_id: 8191,
            sample_rate: 16000,
            win_length: 400,
            hop_length: 160,
            n_mels: 80,
        }
    }
}

fn u32_meta(gguf: &GgufFile, key: &str) -> Result<u32, ConfigError> {
    gguf.meta_u32(key)
        .ok_or_else(|| ConfigError::MissingKey(key.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy_reazonspeech_profile() {
        let c = Config::dummy_reazonspeech_nemo_v2();
        assert_eq!(c.decoder_type, DecoderType::Rnnt);
        assert_eq!(c.attention_type, AttentionType::RelPosLocalAttn);
        assert_eq!(c.local_window, 256);
        assert_eq!(c.global_tokens, 1);
        assert_eq!(c.subsampling_factor, 8);
        assert_eq!(c.vocab_size, 3000);
    }

    #[test]
    fn dummy_parakeet_profile() {
        let c = Config::dummy_parakeet_tdt_v3();
        assert_eq!(c.decoder_type, DecoderType::Tdt);
        assert_eq!(c.attention_type, AttentionType::RelPos);
        assert_eq!(c.local_window, 0);
        assert_eq!(c.global_tokens, 0);
        assert_eq!(c.vocab_size, 8192);
    }
}
