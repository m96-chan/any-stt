//! Minimal SentencePiece detokenizer for ASR decoder output.
//!
//! ReazonSpeech-NeMo-v2 ships a SentencePiece unigram model with 3,000
//! pieces (Japanese). Parakeet-TDT-0.6B-v3 ships an 8,192-piece unigram
//! model. Both use the same `.model` proto3 schema, so the parser here
//! handles both — only the embedded `pieces` repeated field is read.
//!
//! ## Why hand-rolled
//!
//! The full `sentencepiece` C++ binding adds ~1 MB of binary and complex
//! cross-compilation steps for Android/iOS. For ASR backend output we only
//! need the **detokenize** direction (token IDs → text), which boils down
//! to: piece-table lookup + concat + replace `▁` (U+2581) with space. This
//! file ships that minimum without a foreign dependency.
//!
//! ## What's NOT here
//!
//! - Encoding (text → token IDs). RNN-T / TDT decoding emits IDs from the
//!   joint network argmax; there is no text-to-id path on the inference
//!   side of this crate.
//! - Trie / Viterbi scoring. Pieces with floating-point scores are read
//!   but the score is discarded — we never need to re-segment text here.
//! - Normalization. NeMo runs SP without an extra NFKC layer (already
//!   applied during training preproc), so we don't either.

use std::path::Path;

/// SentencePiece piece type, mirroring the values in
/// sentencepiece_model.proto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    /// Regular subword piece. Default when not explicitly set.
    Normal,
    /// `<unk>` — emitted by the model when the input character is not in
    /// the vocab. Kept in the output so callers can spot OOV cases.
    Unknown,
    /// `<s>`, `</s>` and similar — never appears in transcript output.
    Control,
    /// User-supplied special token.
    UserDefined,
    /// Reserved slot, ignored.
    Unused,
    /// Byte-fallback piece (rare in NeMo unigram models).
    Byte,
}

impl PieceType {
    fn from_proto(n: i32) -> Self {
        match n {
            // 1 = NORMAL (default)
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

#[derive(Debug, Clone)]
struct Piece {
    text: String,
    ptype: PieceType,
}

/// SentencePiece tokenizer (decode-only).
pub struct SentencePieceTokenizer {
    pieces: Vec<Piece>,
}

impl SentencePieceTokenizer {
    /// Load a SentencePiece `.model` file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("read tokenizer model {}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    /// Parse a SentencePiece model from raw proto3 bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut pieces = Vec::new();
        let mut p = ProtoReader::new(data);
        while p.has_more() {
            let (tag, wire) = p.read_tag()?;
            // ModelProto.pieces = tag 1, wire = length-delimited
            if tag == 1 && wire == WIRE_LENGTH_DELIMITED {
                let inner = p.read_length_delimited()?;
                let piece = parse_piece(inner)?;
                pieces.push(piece);
            } else {
                p.skip(wire)?;
            }
        }
        if pieces.is_empty() {
            return Err("SentencePiece model contains no pieces".into());
        }
        Ok(Self { pieces })
    }

    /// Total number of pieces (== vocab size).
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Look up a piece's surface text by id.
    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(|p| p.text.as_str())
    }

    /// Detokenize a sequence of piece IDs into text.
    ///
    /// - Pieces marked `Control` or `Unused` (e.g. `<s>`, `</s>`) are
    ///   stripped silently.
    /// - The SP whitespace marker `▁` (U+2581) is replaced with a single
    ///   ASCII space.
    /// - Leading whitespace is trimmed (matches SP-Python's behavior on
    ///   the first piece, which carries a leading `▁`).
    pub fn detokenize(&self, ids: &[u32]) -> Result<String, String> {
        let mut out = String::new();
        for &id in ids {
            let idx = id as usize;
            if idx >= self.pieces.len() {
                return Err(format!(
                    "token id {id} out of vocab range {}",
                    self.pieces.len()
                ));
            }
            let p = &self.pieces[idx];
            if matches!(p.ptype, PieceType::Control | PieceType::Unused) {
                continue;
            }
            out.push_str(&p.text);
        }
        // U+2581 LOWER ONE EIGHTH BLOCK is the SentencePiece whitespace
        // marker. Detokenize replaces with a regular space.
        let with_spaces = out.replace('\u{2581}', " ");
        Ok(with_spaces.trim_start().to_string())
    }
}

fn parse_piece(data: &[u8]) -> Result<Piece, String> {
    let mut text = String::new();
    let mut ptype = PieceType::Normal;
    let mut p = ProtoReader::new(data);
    while p.has_more() {
        let (tag, wire) = p.read_tag()?;
        match (tag, wire) {
            // SentencePiece.piece = tag 1, string
            (1, WIRE_LENGTH_DELIMITED) => {
                let s = p.read_length_delimited()?;
                text = std::str::from_utf8(s)
                    .map_err(|e| format!("piece text not UTF-8: {e}"))?
                    .to_string();
            }
            // SentencePiece.type = tag 3, varint
            (3, WIRE_VARINT) => {
                let v = p.read_varint()? as i32;
                ptype = PieceType::from_proto(v);
            }
            // tag 2 = score (float), and any future fields — skip safely.
            _ => p.skip(wire)?,
        }
    }
    Ok(Piece { text, ptype })
}

// ---------------------------------------------------------------------------
// Minimal proto3 wire-format reader. Handles the subset we need: varint,
// length-delimited, and skip-over for unknown / unused fields.
// ---------------------------------------------------------------------------

const WIRE_VARINT: u32 = 0;
const WIRE_64BIT: u32 = 1;
const WIRE_LENGTH_DELIMITED: u32 = 2;
const WIRE_32BIT: u32 = 5;

struct ProtoReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn has_more(&self) -> bool {
        self.pos < self.data.len()
    }

    /// Read a (tag, wire_type) pair.
    fn read_tag(&mut self) -> Result<(u32, u32), String> {
        let v = self.read_varint()? as u32;
        Ok((v >> 3, v & 0b111))
    }

    /// Read a base-128 variable-length integer.
    fn read_varint(&mut self) -> Result<u64, String> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            if self.pos >= self.data.len() {
                return Err("unexpected EOF in varint".into());
            }
            let b = self.data[self.pos];
            self.pos += 1;
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err("varint exceeds 64 bits".into());
            }
        }
    }

    /// Read a length-prefixed byte slice.
    fn read_length_delimited(&mut self) -> Result<&'a [u8], String> {
        let len = self.read_varint()? as usize;
        if self.pos + len > self.data.len() {
            return Err(format!(
                "length-delimited overruns buffer: pos={} len={} data_len={}",
                self.pos,
                len,
                self.data.len()
            ));
        }
        let start = self.pos;
        self.pos += len;
        Ok(&self.data[start..self.pos])
    }

    /// Skip over an unknown / unused field of a given wire type.
    fn skip(&mut self, wire: u32) -> Result<(), String> {
        match wire {
            WIRE_VARINT => {
                self.read_varint()?;
            }
            WIRE_64BIT => {
                if self.pos + 8 > self.data.len() {
                    return Err("EOF skipping 64-bit field".into());
                }
                self.pos += 8;
            }
            WIRE_LENGTH_DELIMITED => {
                self.read_length_delimited()?;
            }
            WIRE_32BIT => {
                if self.pos + 4 > self.data.len() {
                    return Err("EOF skipping 32-bit field".into());
                }
                self.pos += 4;
            }
            other => return Err(format!("unknown wire type: {other}")),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Test fixtures ---
    //
    // We synthesize a minimal SP model proto in-memory so the tests can
    // run without committing a binary fixture. The encoding side mirrors
    // the reading side just enough to round-trip.

    fn encode_varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn encode_tag(tag: u32, wire: u32, out: &mut Vec<u8>) {
        encode_varint(((tag << 3) | wire) as u64, out);
    }

    /// Build one ModelProto.pieces submessage entry.
    fn encode_piece(text: &str, ptype: i32, out: &mut Vec<u8>) {
        let mut inner = Vec::new();
        // SentencePiece.piece (tag 1, length-delimited)
        encode_tag(1, WIRE_LENGTH_DELIMITED, &mut inner);
        encode_varint(text.len() as u64, &mut inner);
        inner.extend_from_slice(text.as_bytes());
        // SentencePiece.type (tag 3, varint) — emit only if non-default.
        if ptype != 1 {
            encode_tag(3, WIRE_VARINT, &mut inner);
            encode_varint(ptype as u64, &mut inner);
        }
        // ModelProto.pieces (tag 1, length-delimited)
        encode_tag(1, WIRE_LENGTH_DELIMITED, out);
        encode_varint(inner.len() as u64, out);
        out.extend_from_slice(&inner);
    }

    fn build_model(entries: &[(&str, i32)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (text, ptype) in entries {
            encode_piece(text, *ptype, &mut out);
        }
        out
    }

    // --- Tests ---

    #[test]
    fn varint_roundtrip_covers_edges() {
        for v in [0_u64, 1, 127, 128, 16383, 16384, 12345678, u64::MAX] {
            let mut buf = Vec::new();
            encode_varint(v, &mut buf);
            let mut p = ProtoReader::new(&buf);
            let got = p.read_varint().unwrap();
            assert_eq!(got, v, "roundtrip failed for {v}");
            assert!(!p.has_more(), "leftover bytes at v={v}");
        }
    }

    #[test]
    fn parse_minimal_model_matches_pieces() {
        let bytes = build_model(&[
            ("<unk>", 2), // UNKNOWN
            ("<s>", 3),   // CONTROL
            ("</s>", 3),  // CONTROL
            ("\u{2581}hello", 1),
            ("world", 1),
        ]);
        let tok = SentencePieceTokenizer::from_bytes(&bytes).unwrap();
        assert_eq!(tok.vocab_size(), 5);
        assert_eq!(tok.piece(0), Some("<unk>"));
        assert_eq!(tok.piece(3), Some("\u{2581}hello"));
        assert_eq!(tok.piece(99), None);
    }

    #[test]
    fn detokenize_english_strips_specials_and_replaces_marker() {
        let bytes = build_model(&[
            ("<unk>", 2),
            ("<s>", 3),
            ("</s>", 3),
            ("\u{2581}hello", 1),
            ("\u{2581}world", 1),
            ("!", 1),
        ]);
        let tok = SentencePieceTokenizer::from_bytes(&bytes).unwrap();
        // <s>, ▁hello, ▁world, !, </s>
        let text = tok.detokenize(&[1, 3, 4, 5, 2]).unwrap();
        assert_eq!(text, "hello world!");
    }

    #[test]
    fn detokenize_japanese_no_whitespace() {
        // Japanese pieces never carry the ▁ marker (NeMo SP unigram for
        // ja). Sequence "我輩 は 猫 で ある 。" round-trips losslessly.
        let bytes = build_model(&[
            ("<unk>", 2),
            ("<s>", 3),
            ("</s>", 3),
            ("我輩", 1),
            ("は", 1),
            ("猫", 1),
            ("で", 1),
            ("ある", 1),
            ("。", 1),
        ]);
        let tok = SentencePieceTokenizer::from_bytes(&bytes).unwrap();
        let text = tok.detokenize(&[3, 4, 5, 6, 7, 8]).unwrap();
        assert_eq!(text, "我輩は猫である。");
    }

    #[test]
    fn detokenize_unknown_token_kept_as_unk_text() {
        // PieceType::Unknown is NOT a control token, so the literal
        // "<unk>" should appear in the output (matches sp.decode behavior).
        let bytes = build_model(&[("<unk>", 2), ("a", 1)]);
        let tok = SentencePieceTokenizer::from_bytes(&bytes).unwrap();
        assert_eq!(tok.detokenize(&[1, 0, 1]).unwrap(), "a<unk>a");
    }

    #[test]
    fn detokenize_out_of_range_id_errors() {
        let bytes = build_model(&[("a", 1)]);
        let tok = SentencePieceTokenizer::from_bytes(&bytes).unwrap();
        assert!(tok.detokenize(&[5]).is_err());
    }

    #[test]
    fn empty_proto_is_rejected() {
        assert!(SentencePieceTokenizer::from_bytes(&[]).is_err());
    }

    #[test]
    fn missing_file_returns_io_error() {
        assert!(SentencePieceTokenizer::load(Path::new("/does/not/exist.model")).is_err());
    }

    #[test]
    fn proto_with_score_field_is_skipped_safely() {
        // Manually build a piece with a score (tag 2, fixed32 = wire 5).
        // This exercises the skip path in parse_piece for unknown wire
        // types we don't care about.
        let mut inner = Vec::new();
        encode_tag(1, WIRE_LENGTH_DELIMITED, &mut inner);
        encode_varint(3, &mut inner); // length 3
        inner.extend_from_slice(b"abc");
        encode_tag(2, WIRE_32BIT, &mut inner); // score (float)
        inner.extend_from_slice(&[0x00, 0x00, 0x80, 0x3f]); // 1.0 as little-endian f32
        encode_tag(3, WIRE_VARINT, &mut inner);
        encode_varint(1, &mut inner);

        let mut model = Vec::new();
        encode_tag(1, WIRE_LENGTH_DELIMITED, &mut model);
        encode_varint(inner.len() as u64, &mut model);
        model.extend_from_slice(&inner);

        let tok = SentencePieceTokenizer::from_bytes(&model).unwrap();
        assert_eq!(tok.piece(0), Some("abc"));
    }
}
