//! SentencePiece tokenizer wrapper.
//!
//! ReazonSpeech-NeMo-v2 ships a SentencePiece unigram model with a 3,000
//! token vocabulary (Japanese). The `.nemo` archive includes the raw
//! `.model` file, which `scripts/convert-nemo-to-gguf.py` copies alongside
//! the `.gguf` output as `<name>.tokenizer.model`.
//!
//! ## Status
//! Stub. Intentionally does not pull in a sentencepiece dep yet to avoid
//! committing to a specific binding crate before reviewing maintenance
//! status (see `feedback_dep_policy` in memory).
//!
//! Candidates under evaluation:
//!   - `sentencepiece` (rust wrapper around C++)  — 0.11, maintained
//!   - `tokenizers` (HF)                         — heavy but very active
//!   - custom minimal unigram decoder            — avoids C++ dep on mobile
//!
//! For the greedy decode → text path, only the detokenize step is strictly
//! required (the encoder works on audio features, not tokens).

use std::path::Path;

/// SentencePiece tokenizer handle.
pub struct SentencePieceTokenizer {
    // TODO(#N4): hold the actual sp model handle
}

impl SentencePieceTokenizer {
    /// Load a SentencePiece model from the companion file produced by
    /// `convert-nemo-to-gguf.py`.
    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("tokenizer model not found: {}", path.display()));
        }
        // TODO(#N4): parse SentencePiece model and build piece table.
        Ok(Self {})
    }

    /// Decode a sequence of token IDs back into text.
    /// For Japanese, this collapses SentencePiece pieces and strips the
    /// leading ▁ whitespace markers.
    pub fn detokenize(&self, _ids: &[u32]) -> Result<String, String> {
        // TODO(#N4): implement piece lookup + whitespace handling.
        Err("SentencePieceTokenizer::detokenize not yet implemented".into())
    }
}
