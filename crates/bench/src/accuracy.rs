//! Accuracy metrics for ASR output: CER (Character Error Rate) and WER
//! (Word Error Rate).
//!
//! Implements Levenshtein edit distance at character and word granularity
//! with text normalization (NFKC, case folding, optional punctuation strip).
//! Values match Python `jiwer` on the covered cases — see `tests/`.

use unicode_normalization::UnicodeNormalization;

/// Text normalization options applied before computing CER/WER.
#[derive(Debug, Clone)]
pub struct NormalizeOpts {
    /// Apply Unicode NFKC normalization (full-width → half-width, etc).
    /// Important for Japanese text where the same character may appear in
    /// multiple codepoints.
    pub nfkc: bool,
    /// Lowercase ASCII. No-op for CJK.
    pub lowercase: bool,
    /// Strip punctuation characters and common ASR markers.
    pub strip_punctuation: bool,
    /// Collapse consecutive whitespace into a single space; trim ends.
    pub collapse_whitespace: bool,
}

impl Default for NormalizeOpts {
    fn default() -> Self {
        Self {
            nfkc: true,
            lowercase: true,
            strip_punctuation: true,
            collapse_whitespace: true,
        }
    }
}

/// Characters stripped by `strip_punctuation`. Covers Latin + CJK common.
const PUNCT: &[char] = &[
    '.', ',', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}',
    '-', '—', '–', '…',
    '、', '。', '！', '？', '「', '」', '『', '』', '（', '）', '・', '〜',
];

/// Normalize text for fair comparison.
pub fn normalize(text: &str, opts: &NormalizeOpts) -> String {
    let mut s: String = if opts.nfkc {
        text.nfkc().collect()
    } else {
        text.to_string()
    };

    if opts.lowercase {
        s = s.to_lowercase();
    }

    if opts.strip_punctuation {
        s = s.chars().filter(|c| !PUNCT.contains(c)).collect();
    }

    if opts.collapse_whitespace {
        s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    }

    s
}

/// Tokenize for WER: whitespace split after normalization.
/// Suitable for space-separated languages (English, German, ...).
pub fn tokenize_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(|w| w.to_string()).collect()
}

/// Tokenize for CER: each Unicode scalar value is a token.
/// Suitable for all languages; primary metric for Japanese/Chinese/Korean.
pub fn tokenize_chars(text: &str) -> Vec<char> {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Edit-distance components and their error rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorRate {
    /// Number of substitutions in the optimal alignment.
    pub substitutions: usize,
    /// Number of deletions (tokens missing from hypothesis).
    pub deletions: usize,
    /// Number of insertions (extra tokens in hypothesis).
    pub insertions: usize,
    /// Length of the reference in tokens.
    pub ref_len: usize,
    /// (sub + del + ins) / ref_len. `NaN` if ref_len == 0.
    pub rate: f64,
}

impl ErrorRate {
    pub fn total_edits(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }
}

/// Levenshtein edit distance with substitution/deletion/insertion breakdown
/// via backtracking over the DP matrix.
///
/// `ref_tokens`: the ground-truth token sequence.
/// `hyp_tokens`: the model's output token sequence.
pub fn edit_distance<T: Eq>(ref_tokens: &[T], hyp_tokens: &[T]) -> ErrorRate {
    let n = ref_tokens.len();
    let m = hyp_tokens.len();

    // dp[i][j] = edit distance between ref_tokens[..i] and hyp_tokens[..j]
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            if ref_tokens[i - 1] == hyp_tokens[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + dp[i - 1][j - 1] // substitute
                    .min(dp[i - 1][j])           // delete from ref
                    .min(dp[i][j - 1]);          // insert into ref
            }
        }
    }

    // Backtrack to classify each edit.
    let (mut i, mut j) = (n, m);
    let mut subs = 0;
    let mut dels = 0;
    let mut inss = 0;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && ref_tokens[i - 1] == hyp_tokens[j - 1] {
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
            subs += 1;
            i -= 1;
            j -= 1;
        } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            dels += 1;
            i -= 1;
        } else {
            // j > 0 && dp[i][j] == dp[i][j-1] + 1
            inss += 1;
            j -= 1;
        }
    }

    let rate = if n == 0 {
        f64::NAN
    } else {
        (subs + dels + inss) as f64 / n as f64
    };

    ErrorRate {
        substitutions: subs,
        deletions: dels,
        insertions: inss,
        ref_len: n,
        rate,
    }
}

/// Compute CER between `reference` and `hypothesis`.
pub fn cer(reference: &str, hypothesis: &str, opts: &NormalizeOpts) -> ErrorRate {
    let r = normalize(reference, opts);
    let h = normalize(hypothesis, opts);
    let r_tokens = tokenize_chars(&r);
    let h_tokens = tokenize_chars(&h);
    edit_distance(&r_tokens, &h_tokens)
}

/// Compute WER between `reference` and `hypothesis`.
/// Prefer `cer` for CJK languages.
pub fn wer(reference: &str, hypothesis: &str, opts: &NormalizeOpts) -> ErrorRate {
    let r = normalize(reference, opts);
    let h = normalize(hypothesis, opts);
    let r_tokens = tokenize_words(&r);
    let h_tokens = tokenize_words(&h);
    edit_distance(&r_tokens, &h_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings_zero_error() {
        let opts = NormalizeOpts::default();
        assert_eq!(cer("hello world", "hello world", &opts).rate, 0.0);
        assert_eq!(wer("hello world", "hello world", &opts).rate, 0.0);
    }

    #[test]
    fn wer_single_substitution() {
        // "the cat sat" vs "the bat sat" → 1 sub over 3 words = 1/3
        let opts = NormalizeOpts::default();
        let r = wer("the cat sat", "the bat sat", &opts);
        assert_eq!(r.substitutions, 1);
        assert_eq!(r.deletions, 0);
        assert_eq!(r.insertions, 0);
        assert_eq!(r.ref_len, 3);
        assert!((r.rate - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn wer_insertion_and_deletion() {
        // ref:  "a b c"        (3 words)
        // hyp:  "a x b"        (3 words)
        // Aligns as: a=a, insert x, b=b, delete c → 1 ins + 1 del = 2 edits / 3
        let opts = NormalizeOpts::default();
        let r = wer("a b c", "a x b", &opts);
        assert_eq!(r.ref_len, 3);
        assert_eq!(r.total_edits(), 2);
        assert!((r.rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn cer_japanese_single_substitution() {
        // ref: "我輩は猫である" (7 chars)
        // hyp: "我輩は犬である" (7 chars, 1 char differs: 猫→犬)
        // → 1 sub / 7 = 0.1428...
        let opts = NormalizeOpts::default();
        let r = cer("我輩は猫である", "我輩は犬である", &opts);
        assert_eq!(r.ref_len, 7);
        assert_eq!(r.substitutions, 1);
        assert_eq!(r.deletions, 0);
        assert_eq!(r.insertions, 0);
        assert!((r.rate - 1.0 / 7.0).abs() < 1e-9);
    }

    #[test]
    fn cer_ignores_punctuation_and_case() {
        let opts = NormalizeOpts::default();
        // Different punctuation + case, same spoken content.
        let r = cer("Hello, World!", "hello world", &opts);
        assert_eq!(r.rate, 0.0);
    }

    #[test]
    fn nfkc_normalizes_halfwidth_fullwidth() {
        let opts = NormalizeOpts::default();
        // Full-width "ＡＢＣ" vs half-width "ABC" → 0 after NFKC.
        let r = cer("ＡＢＣ", "ABC", &opts);
        assert_eq!(r.rate, 0.0);
    }

    #[test]
    fn empty_reference_produces_nan_rate() {
        let opts = NormalizeOpts::default();
        let r = cer("", "hello", &opts);
        assert!(r.rate.is_nan());
        // All hypothesis tokens count as insertions but rate undefined.
        assert_eq!(r.insertions, 5); // h,e,l,l,o
    }

    #[test]
    fn empty_hypothesis_all_deletions() {
        let opts = NormalizeOpts::default();
        let r = cer("hello", "", &opts);
        assert_eq!(r.deletions, 5);
        assert_eq!(r.insertions, 0);
        assert_eq!(r.substitutions, 0);
        assert_eq!(r.rate, 1.0);
    }

    #[test]
    fn edit_distance_matches_classic_example() {
        // Wikipedia example: kitten → sitting, edit distance 3.
        let r: Vec<char> = "kitten".chars().collect();
        let h: Vec<char> = "sitting".chars().collect();
        let ed = edit_distance(&r, &h);
        assert_eq!(ed.total_edits(), 3);
        assert_eq!(ed.ref_len, 6);
    }

    #[test]
    fn normalize_collapses_whitespace() {
        let opts = NormalizeOpts::default();
        assert_eq!(normalize("  hello   world  ", &opts), "hello world");
    }

    #[test]
    fn normalize_preserves_non_punct_cjk() {
        let opts = NormalizeOpts::default();
        // Japanese period stripped, hiragana/katakana preserved.
        assert_eq!(normalize("我輩は猫である。", &opts), "我輩は猫である");
    }

    #[test]
    fn disabling_normalize_opts_makes_cer_stricter() {
        let strict = NormalizeOpts {
            nfkc: false,
            lowercase: false,
            strip_punctuation: false,
            collapse_whitespace: false,
        };
        // With default normalization this is zero error; strict sees diffs.
        let default_opts = NormalizeOpts::default();
        assert_eq!(cer("Hello.", "hello", &default_opts).rate, 0.0);
        let r = cer("Hello.", "hello", &strict);
        assert!(r.rate > 0.0);
    }
}
