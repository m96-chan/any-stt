//! Generic accuracy bench driver.
//!
//! Takes a [`Manifest`] and a `Box<dyn SttEngine>` (constructed by the
//! caller for whatever family is under test) and produces per-item
//! [`ItemResult`]s with CER / WER / RTF.

use std::time::Instant;

use any_stt::{SttEngine, SttError};

use crate::accuracy::{cer, wer, ErrorRate, NormalizeOpts};
use crate::manifest::{Item, Manifest};
use crate::shared::{audio_duration_secs, load_audio};

/// Per-sample timing statistics over N runs.
#[derive(Debug, Clone, Copy)]
pub struct RunStats {
    pub runs: usize,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    /// Real-time factor: median_ms / audio_duration_ms. Lower is faster.
    pub rtf: f64,
}

/// Result for a single manifest item.
#[derive(Debug, Clone)]
pub struct ItemResult {
    pub id: String,
    pub language: String,
    pub reference: String,
    pub hypothesis: String,
    pub audio_secs: f64,
    pub stats: RunStats,
    pub cer: ErrorRate,
    pub wer: ErrorRate,
}

/// Aggregate result for a manifest run.
#[derive(Debug, Clone)]
pub struct ReportRow {
    pub manifest_name: String,
    pub items: Vec<ItemResult>,
}

/// Run the engine over every manifest item and collect metrics.
pub fn run_manifest(
    manifest: &Manifest,
    engine: &dyn SttEngine,
    runs: usize,
    warmup: bool,
    normalize_opts: &NormalizeOpts,
) -> Result<ReportRow, BenchError> {
    assert!(runs >= 1, "runs must be >= 1");
    let mut rows = Vec::with_capacity(manifest.items.len());

    for item in &manifest.items {
        let result = run_item(item, engine, runs, warmup, normalize_opts)?;
        rows.push(result);
    }

    Ok(ReportRow {
        manifest_name: manifest.name.clone(),
        items: rows,
    })
}

fn run_item(
    item: &Item,
    engine: &dyn SttEngine,
    runs: usize,
    warmup: bool,
    normalize_opts: &NormalizeOpts,
) -> Result<ItemResult, BenchError> {
    let audio = load_audio(&item.audio);
    let audio_secs = audio_duration_secs(&audio);

    if warmup {
        let _ = engine.transcribe(&audio);
    }

    let mut timings = Vec::with_capacity(runs);
    let mut last_text = String::new();

    for _ in 0..runs {
        let start = Instant::now();
        let result = engine
            .transcribe(&audio)
            .map_err(|e| BenchError::Transcribe {
                item_id: item.id.clone(),
                source: e,
            })?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        timings.push(ms);
        last_text = result.text;
    }

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = timings[timings.len() / 2];
    let min_ms = timings[0];
    let max_ms = *timings.last().unwrap();
    let rtf = if audio_secs > 0.0 {
        median_ms / 1000.0 / audio_secs
    } else {
        f64::NAN
    };

    let cer_rate = cer(&item.reference, &last_text, normalize_opts);
    let wer_rate = wer(&item.reference, &last_text, normalize_opts);

    Ok(ItemResult {
        id: item.id.clone(),
        language: item.language.clone(),
        reference: item.reference.clone(),
        hypothesis: last_text,
        audio_secs,
        stats: RunStats {
            runs,
            median_ms,
            min_ms,
            max_ms,
            rtf,
        },
        cer: cer_rate,
        wer: wer_rate,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    #[error("transcribe failed for item {item_id}: {source}")]
    Transcribe {
        item_id: String,
        #[source]
        source: SttError,
    },
}
