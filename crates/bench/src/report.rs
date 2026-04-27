//! Report formatting: markdown + JSON.

use std::fmt::Write;

use crate::driver::{ItemResult, ReportRow};

/// Render a manifest run as a markdown report suitable for stdout or
/// a `.md` file.
pub fn markdown(row: &ReportRow) -> String {
    let mut out = String::new();

    let _ = writeln!(out, "# {} — accuracy report", row.manifest_name);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| id | lang | audio (s) | runs | median ms | min ms | max ms | RTF | CER | WER |"
    );
    let _ = writeln!(
        out,
        "|----|------|-----------|------|-----------|--------|--------|-----|-----|-----|"
    );
    for it in &row.items {
        let _ = writeln!(
            out,
            "| {} | {} | {:.2} | {} | {:.1} | {:.1} | {:.1} | {:.3} | {} | {} |",
            it.id,
            it.language,
            it.audio_secs,
            it.stats.runs,
            it.stats.median_ms,
            it.stats.min_ms,
            it.stats.max_ms,
            it.stats.rtf,
            fmt_rate(it.cer.rate),
            fmt_rate(it.wer.rate),
        );
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "## Transcriptions");
    let _ = writeln!(out);
    for it in &row.items {
        let _ = writeln!(out, "### {}", it.id);
        let _ = writeln!(out, "- ref: `{}`", it.reference);
        let _ = writeln!(out, "- hyp: `{}`", it.hypothesis);
        let _ = writeln!(
            out,
            "- CER: {} ({} sub / {} del / {} ins / {} ref)",
            fmt_rate(it.cer.rate),
            it.cer.substitutions,
            it.cer.deletions,
            it.cer.insertions,
            it.cer.ref_len,
        );
        let _ = writeln!(
            out,
            "- WER: {} ({} sub / {} del / {} ins / {} ref)",
            fmt_rate(it.wer.rate),
            it.wer.substitutions,
            it.wer.deletions,
            it.wer.insertions,
            it.wer.ref_len,
        );
    }

    out
}

fn fmt_rate(r: f64) -> String {
    if r.is_nan() {
        "n/a".to_string()
    } else {
        format!("{:.3}", r)
    }
}

/// Render a report as JSON for machine consumption (CI, dashboards).
pub fn json(row: &ReportRow) -> Result<String, serde_json::Error> {
    #[derive(serde::Serialize)]
    struct JsonItem<'a> {
        id: &'a str,
        language: &'a str,
        audio_secs: f64,
        runs: usize,
        median_ms: f64,
        min_ms: f64,
        max_ms: f64,
        rtf: f64,
        cer: f64,
        wer: f64,
        reference: &'a str,
        hypothesis: &'a str,
    }
    #[derive(serde::Serialize)]
    struct JsonReport<'a> {
        manifest: &'a str,
        items: Vec<JsonItem<'a>>,
    }

    let items: Vec<_> = row
        .items
        .iter()
        .map(|it| JsonItem {
            id: &it.id,
            language: &it.language,
            audio_secs: it.audio_secs,
            runs: it.stats.runs,
            median_ms: it.stats.median_ms,
            min_ms: it.stats.min_ms,
            max_ms: it.stats.max_ms,
            rtf: it.stats.rtf,
            cer: it.cer.rate,
            wer: it.wer.rate,
            reference: &it.reference,
            hypothesis: &it.hypothesis,
        })
        .collect();

    serde_json::to_string_pretty(&JsonReport {
        manifest: &row.manifest_name,
        items,
    })
}

/// One-line summary for legacy (non-accuracy) bench output.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub label: String,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub text: String,
}

impl BenchResult {
    pub fn print(&self) {
        eprintln!(
            "  → median: {:.1}ms, min: {:.1}ms, max: {:.1}ms",
            self.median_ms, self.min_ms, self.max_ms
        );
        if !self.text.is_empty() && !self.text.starts_with('(') {
            eprintln!("  → text: \"{}\"", self.text.trim());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accuracy::ErrorRate;
    use crate::driver::{ItemResult, RunStats};

    fn mock_item(id: &str, cer_rate: f64, wer_rate: f64) -> ItemResult {
        ItemResult {
            id: id.into(),
            language: "en".into(),
            reference: "hello world".into(),
            hypothesis: "hello world".into(),
            audio_secs: 11.0,
            stats: RunStats {
                runs: 3,
                median_ms: 100.0,
                min_ms: 90.0,
                max_ms: 110.0,
                rtf: 0.009,
            },
            cer: ErrorRate {
                substitutions: 0,
                deletions: 0,
                insertions: 0,
                ref_len: 11,
                rate: cer_rate,
            },
            wer: ErrorRate {
                substitutions: 0,
                deletions: 0,
                insertions: 0,
                ref_len: 2,
                rate: wer_rate,
            },
        }
    }

    #[test]
    fn markdown_contains_header_and_row() {
        let row = ReportRow {
            manifest_name: "test".into(),
            items: vec![mock_item("a", 0.0, 0.0)],
        };
        let md = markdown(&row);
        assert!(md.contains("| id | lang"));
        assert!(md.contains("| a | en"));
    }

    #[test]
    fn json_roundtrips() {
        let row = ReportRow {
            manifest_name: "test".into(),
            items: vec![mock_item("a", 0.05, 0.1)],
        };
        let j = json(&row).unwrap();
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        assert_eq!(v["manifest"], "test");
        assert_eq!(v["items"][0]["id"], "a");
        assert!((v["items"][0]["cer"].as_f64().unwrap() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn nan_rate_renders_as_na() {
        assert_eq!(fmt_rate(f64::NAN), "n/a");
        assert_eq!(fmt_rate(0.5), "0.500");
    }
}
