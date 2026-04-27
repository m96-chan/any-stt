//! Model configuration parsed from the GGUF header.
//!
//! All keys come from `scripts/convert-nemo-to-gguf.py` under the
//! `fastconformer.*` namespace.

use gguf_loader::GgufFile;

#[derive(Debug, Clone)]
pub struct ReazonSpeechConfig {
    // Encoder
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub feat_in: u32,           // n_mels, typically 80
    pub conv_kernel_size: u32,
    pub subsampling_factor: u32, // 8 for FastConformer
    pub attention_type: AttentionType,
    pub local_window: u32,       // Longformer local window (0 if regular attn)
    pub global_tokens: u32,      // Longformer global tokens (typically 0 or 1)

    // Decoder
    pub decoder_type: DecoderType, // rnnt for reazonspeech-nemo-v2
    pub vocab_size: u32,
    pub pred_hidden: u32,
    pub joint_hidden: u32,
    pub blank_id: u32,

    // Audio preprocessing
    pub sample_rate: u32, // 16000
    pub win_length: u32,  // 400 samples = 25ms @ 16kHz
    pub hop_length: u32,  // 160 samples = 10ms @ 16kHz
    pub n_mels: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Standard relative-position multi-head attention.
    RelPos,
    /// Longformer-style local attention + global tokens (ReazonSpeech v2).
    RelPosLocalAttn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderType {
    /// RNN-T (reazonspeech).
    Rnnt,
    /// TDT (parakeet). Not used by ReazonSpeech but declared for shared
    /// metadata extraction.
    Tdt,
}

impl ReazonSpeechConfig {
    /// Parse a config from the GGUF header. Does not load tensor data.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf
            .meta_str("general.architecture")
            .ok_or_else(|| "missing general.architecture".to_string())?;
        if arch != "fastconformer" {
            return Err(format!(
                "expected general.architecture='fastconformer', got {arch:?}"
            ));
        }

        let attention_type = match gguf.meta_str("fastconformer.encoder.attention_type") {
            Some("rel_pos_local_attn") => AttentionType::RelPosLocalAttn,
            Some("rel_pos") | None => AttentionType::RelPos,
            Some(other) => {
                return Err(format!("unknown attention_type: {other}"));
            }
        };

        let decoder_type = match gguf.meta_str("fastconformer.decoder.type") {
            Some("rnnt") => DecoderType::Rnnt,
            Some("tdt") => DecoderType::Tdt,
            Some(other) => return Err(format!("unknown decoder.type: {other}")),
            None => return Err("missing fastconformer.decoder.type".into()),
        };

        Ok(Self {
            n_layers: metadata_u32(gguf, "fastconformer.encoder.n_layers")?,
            d_model: metadata_u32(gguf, "fastconformer.encoder.d_model")?,
            n_heads: metadata_u32(gguf, "fastconformer.encoder.n_heads")?,
            feat_in: metadata_u32(gguf, "fastconformer.encoder.feat_in")?,
            conv_kernel_size: metadata_u32(
                gguf,
                "fastconformer.encoder.conv_kernel_size",
            )?,
            subsampling_factor: metadata_u32(
                gguf,
                "fastconformer.encoder.subsampling_factor",
            )?,
            attention_type,
            local_window: metadata_u32(gguf, "fastconformer.encoder.local_window")
                .unwrap_or(0),
            global_tokens: metadata_u32(gguf, "fastconformer.encoder.global_tokens")
                .unwrap_or(0),
            decoder_type,
            vocab_size: metadata_u32(gguf, "fastconformer.decoder.vocab_size")?,
            pred_hidden: metadata_u32(gguf, "fastconformer.decoder.pred_hidden")?,
            joint_hidden: metadata_u32(gguf, "fastconformer.decoder.joint_hidden")?,
            blank_id: metadata_u32(gguf, "fastconformer.decoder.blank_id")?,
            sample_rate: metadata_u32(gguf, "fastconformer.audio.sample_rate")?,
            win_length: metadata_u32(gguf, "fastconformer.audio.win_length")?,
            hop_length: metadata_u32(gguf, "fastconformer.audio.hop_length")?,
            n_mels: metadata_u32(gguf, "fastconformer.audio.n_mels")?,
        })
    }

    /// Placeholder config for unit tests that don't need a real GGUF file.
    #[doc(hidden)]
    pub fn dummy() -> Self {
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
}

fn metadata_u32(gguf: &GgufFile, key: &str) -> Result<u32, String> {
    gguf.meta_u32(key)
        .ok_or_else(|| format!("missing metadata key: {key}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy_matches_reazonspeech_nemo_v2_profile() {
        let c = ReazonSpeechConfig::dummy();
        assert_eq!(c.decoder_type, DecoderType::Rnnt);
        assert_eq!(c.attention_type, AttentionType::RelPosLocalAttn);
        assert_eq!(c.subsampling_factor, 8);
        assert_eq!(c.n_mels, 80);
        assert_eq!(c.sample_rate, 16000);
    }
}
