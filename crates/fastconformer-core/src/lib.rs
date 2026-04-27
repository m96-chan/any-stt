//! Shared FastConformer ASR primitives.
//!
//! Both `reazonspeech-backend` (NeMo + Longformer + RNN-T) and
//! `parakeet-backend` (NeMo + rel-pos + TDT) load FastConformer encoders
//! from GGUF files produced by `scripts/convert-nemo-to-gguf.py`. Their
//! preprocessor, tokenizer, and encoder stack are largely identical.
//! This crate hosts the shared parts so the family-specific crates only
//! contain the pieces that actually differ (decoder + a thin engine
//! wrapper).
//!
//! ## Layout
//!
//! ```text
//!   audio (f32, 16 kHz mono)
//!     │
//!     ▼
//!   mel.rs            — log-mel filterbank, NeMo-compatible
//!     │
//!     ▼
//!   encoder/
//!     subsample.rs    — striding conv2d ×8
//!     attention.rs    — multi-head rel-pos + Longformer variant
//!     conv_module.rs  — pointwise/depthwise/GLU/batchnorm
//!     ff.rs           — feed-forward macaron half
//!     block.rs        — full Conformer block
//!     mod.rs          — N-block stack + ln_post
//!     │
//!     ▼
//!   tokenizer.rs      — SentencePiece detokenize
//! ```
//!
//! ## What is NOT here
//!
//! - Decoders. RNN-T (reazonspeech) and TDT (parakeet) live in their own
//!   crates because their joint-network shapes and emit loops are
//!   different enough that sharing would obscure intent.
//! - Family-specific config validation. `Config` accepts both Longformer
//!   and rel-pos variants; the per-family crate decides which to use.

pub mod config;
pub mod encoder;
pub mod mel;
pub mod tokenizer;

pub use config::{AttentionType, Config, ConfigError, DecoderType};
pub use mel::{log_mel_spectrogram, MelSpectrogram};
pub use tokenizer::{PieceType, SentencePieceTokenizer};
