//! Whisper family variants (OpenAI Whisper + distilled + kotoba).
//!
//! All variants are encoder-decoder transformers loaded via the
//! `whisper-backend` crate (whisper.cpp FFI).

/// Whisper model variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WhisperVariant {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    LargeV1,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
    DistilLargeV2,
    DistilLargeV3,
    DistilMediumEn,
    DistilSmallEn,
    KotobaV1,
    KotobaV2,
}

impl WhisperVariant {
    /// Rough f16 model size estimate in MB.
    pub const fn size_estimate_f16_mb(&self) -> u64 {
        match self {
            Self::Tiny | Self::TinyEn => 75,
            Self::Base | Self::BaseEn => 150,
            Self::Small | Self::SmallEn => 500,
            Self::DistilSmallEn => 400,
            Self::Medium | Self::MediumEn => 1500,
            Self::DistilMediumEn => 1200,
            Self::LargeV1 | Self::LargeV2 | Self::LargeV3 => 3100,
            Self::LargeV3Turbo => 1600,
            Self::DistilLargeV2 | Self::DistilLargeV3 => 1600,
            Self::KotobaV1 | Self::KotobaV2 => 1600,
        }
    }
}
