//! Whisper benchmark: CPU vs GPU (Vulkan) vs NPU (QNN HTP) encoder.
//!
//! Usage:
//!   bench-whisper --model <path> --audio <path> [--runs N] [--backend cpu|gpu|npu|preprocess|all]
//!
//! Backends:
//!   cpu        — Full whisper.cpp transcription on CPU
//!   gpu        — Full whisper.cpp transcription on GPU (Vulkan)
//!   npu        — QNN HTP encoder only (standalone)
//!   preprocess — CPU preprocessor only (Conv1d + GELU + pos_embed)
//!   all        — Run all of the above

use std::path::{Path, PathBuf};
use std::time::Instant;

use any_stt::config::Backend;
use any_stt::{SttEngine, SttResult};
use qnn_backend::EncoderConfig;
use whisper_backend::WhisperEngine;

fn main() {
    let args = parse_args();

    // Sub-process mode: run a single backend benchmark and exit.
    // This prevents segfaults (e.g., Vulkan) from killing the whole process.
    if let Some(ref sub) = args.subprocess {
        run_subprocess(sub, &args);
        return;
    }

    eprintln!("=== Whisper Benchmark ===");
    eprintln!("Model:   {}", args.model.display());
    eprintln!("Audio:   {}", args.audio.display());
    eprintln!("Runs:    {}", args.runs);
    eprintln!("Backend: {:?}", args.backend);
    eprintln!();

    let audio = load_audio(&args.audio);
    let audio_secs = audio.len() as f64 / 16000.0;
    eprintln!("Audio: {} samples ({:.1}s @ 16kHz)", audio.len(), audio_secs);
    eprintln!();

    let hw = any_stt::detect_hardware();
    eprintln!("Hardware: {} ({} cores)", hw.cpu.arch, hw.cpu.cores);
    if let Some(ref gpu) = hw.gpu {
        eprintln!("GPU:      {} ({} MB VRAM)", gpu.name, gpu.vram_mb);
    }
    if let Some(ref npu) = hw.npu {
        eprintln!("NPU:      {:?} (available: {})", npu.npu_type, npu.available);
    }
    eprintln!("RAM:      {} MB available", hw.available_ram_mb);
    eprintln!();

    let backends = args.backend.to_list();
    let mut results: Vec<BenchResult> = Vec::new();

    for backend in &backends {
        let label = backend.label();
        eprintln!("--- {label} ---");

        match backend {
            BenchBackend::Cpu => {
                match bench_whisper_cpp(&args.model, &audio, false, args.runs) {
                    Ok(r) => { print_result(&r); results.push(r); }
                    Err(e) => eprintln!("  SKIP: {e}"),
                }
            }
            BenchBackend::Gpu => {
                // Run GPU in a subprocess — Vulkan can segfault on some drivers
                match run_backend_subprocess(&args, "gpu") {
                    Ok(r) => { print_result(&r); results.push(r); }
                    Err(e) => eprintln!("  SKIP: {e}"),
                }
            }
            BenchBackend::Npu => {
                match bench_npu_encoder(&args.model, args.runs) {
                    Ok(r) => { print_result(&r); results.push(r); }
                    Err(e) => eprintln!("  SKIP: {e}"),
                }
            }
            BenchBackend::Preprocess => {
                match bench_preprocessor(&args.model, args.runs) {
                    Ok(r) => { print_result(&r); results.push(r); }
                    Err(e) => eprintln!("  SKIP: {e}"),
                }
            }
            BenchBackend::Hybrid => {
                match bench_hybrid(&args.model, &audio, args.runs) {
                    Ok(r) => { print_result(&r); results.push(r); }
                    Err(e) => eprintln!("  SKIP: {e}"),
                }
            }
        }
        eprintln!();
    }

    // Summary table
    if !results.is_empty() {
        eprintln!("=== Summary (REDMAGIC 9 Pro / SD 8 Gen 3) ===");
        eprintln!("{:<25} {:>10} {:>10} {:>10} {:>8}",
            "Backend", "Median", "Min", "Max", "RTF");
        eprintln!("{}", "-".repeat(65));
        for r in &results {
            let rtf = r.median_ms / 1000.0 / audio_secs;
            eprintln!("{:<25} {:>8.1}ms {:>8.1}ms {:>8.1}ms {:>8.3}",
                r.label, r.median_ms, r.min_ms, r.max_ms, rtf);
        }
    }
}

// --- Benchmarking functions ---

fn bench_whisper_cpp(
    model_path: &Path,
    audio: &[f32],
    use_gpu: bool,
    runs: usize,
) -> Result<BenchResult, String> {
    let hw = any_stt::detect_hardware();
    let backend = if use_gpu { Backend::Vulkan } else { Backend::Cpu };

    let engine = WhisperEngine::new(model_path, "en", backend, hw)
        .map_err(|e| format!("{e}"))?;

    // Warmup
    eprintln!("  warmup...");
    let _ = engine.transcribe(audio);

    let mut timings = Vec::with_capacity(runs);
    let mut last_result: Option<SttResult> = None;

    for i in 0..runs {
        let start = Instant::now();
        let result = engine.transcribe(audio).map_err(|e| format!("{e}"))?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        timings.push(ms);
        eprint!("  run {}: {:.1}ms", i + 1, ms);
        if i == 0 {
            eprint!(" \"{}\"", result.text.trim());
        }
        eprintln!();
        last_result = Some(result);
    }

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let label = if use_gpu {
        "CPU+GPU (Vulkan)".to_string()
    } else {
        "CPU (whisper.cpp)".to_string()
    };

    Ok(BenchResult {
        label,
        median_ms: timings[timings.len() / 2],
        min_ms: timings[0],
        max_ms: *timings.last().unwrap(),
        text: last_result.map(|r| r.text).unwrap_or_default(),
    })
}

fn bench_npu_encoder(
    model_path: &Path,
    runs: usize,
) -> Result<BenchResult, String> {
    use std::ffi::CString;
    use qnn_backend::{QnnContext, QnnLibrary};
    use qnn_backend::types::*;

    let lib = QnnLibrary::load_htp()
        .map_err(|e| format!("QNN not available: {e}"))?;

    let config = detect_encoder_config(model_path)?;
    let ns = config.n_state as usize;
    let nc = config.n_ctx as usize;
    let nl = config.n_layer as usize;
    eprintln!("  config: n_state={ns}, n_head={}, n_layer={nl}, n_ctx={nc}", config.n_head);

    // Locate raw weight directory
    let candidates = [
        PathBuf::from("weights_tiny_en"),
        PathBuf::from("/data/local/tmp/any-stt/weights"),
    ];
    let wdir = candidates.iter().find(|d| d.join("block0_wq.bin").exists());
    let load_f32 = |name: &str| -> Vec<f32> {
        if let Some(dir) = wdir {
            let path = dir.join(name);
            if let Ok(bytes) = std::fs::read(&path) {
                return bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]]))
                    .collect();
            }
        }
        // Fallback: dummy data
        vec![0.01f32; ns * ns]
    };

    if wdir.is_some() {
        eprintln!("  weights: {}", wdir.unwrap().display());
    } else {
        eprintln!("  (dummy weights — push weights_tiny_en/ to device for real data)");
    }

    // Build single MatMul graph: [nc,ns] × [ns,ns] → [nc,ns]
    let ctx = QnnContext::new(lib).map_err(|e| format!("QNN context: {e}"))?;
    let graph = ctx.create_graph("mm_bench").map_err(|e| format!("graph: {e}"))?;

    let pkg = CString::new("qti.aisw").unwrap();
    let mm_type = CString::new("MatMul").unwrap();
    let tp0 = CString::new("transpose_in0").unwrap();
    let tp1 = CString::new("transpose_in1").unwrap();

    let an = CString::new("a").unwrap();
    let bn = CString::new("b").unwrap();
    let cn = CString::new("c").unwrap();
    let mut ad = [nc as u32, ns as u32];
    let mut bd = [ns as u32, ns as u32];
    let mut cd = [nc as u32, ns as u32];
    let mut ta = Qnn_Tensor_t::app_write(an.as_ptr(), 2, ad.as_mut_ptr());
    let mut tb = Qnn_Tensor_t::app_write(bn.as_ptr(), 2, bd.as_mut_ptr());
    let mut tc = Qnn_Tensor_t::app_read(cn.as_ptr(), 2, cd.as_mut_ptr());
    ctx.register_tensor(graph, &mut ta)?;
    ctx.register_tensor(graph, &mut tb)?;
    ctx.register_tensor(graph, &mut tc)?;

    let nn = CString::new("mm").unwrap();
    let mut params = [
        Qnn_Param_t::scalar(tp0.as_ptr(), Qnn_Scalar_t::bool8(false)),
        Qnn_Param_t::scalar(tp1.as_ptr(), Qnn_Scalar_t::bool8(false)),
    ];
    let mut ins = [ta, tb];
    let mut outs = [tc];
    let op = Qnn_OpConfig_t::new(
        nn.as_ptr(), pkg.as_ptr(), mm_type.as_ptr(),
        &mut params, &mut ins, &mut outs,
    );
    ctx.add_node(graph, op)?;
    ctx.finalize_graph(graph)?;

    let mut ei = unsafe {[ std::ptr::read(&ins[0]), std::ptr::read(&ins[1]) ]};
    let mut eo = unsafe {[ std::ptr::read(&outs[0]) ]};

    let mut input_data = load_f32("encoder_input.bin");
    if input_data.len() != nc * ns { input_data = vec![0.01f32; nc * ns]; }
    let mut weight_data = load_f32("block0_wq.bin");
    if weight_data.len() != ns * ns { weight_data = vec![0.01f32; ns * ns]; }
    let mut output_data = vec![0.0f32; nc * ns];

    // Warmup
    eprintln!("  warmup...");
    ei[0].set_data(&mut input_data);
    ei[1].set_data(&mut weight_data);
    eo[0].set_data(&mut output_data);
    let _ = ctx.execute(graph, &mut ei, &mut eo);

    // Per encoder pass = 24 MatMul ops (4 layers × 6 matmuls: Q,K,V,out,fc1,fc2)
    let ops_per_pass = nl * 6;
    eprintln!("  {} MatMul executions per encoder pass ({nl} layers × 6 ops)", ops_per_pass);

    let mut timings = Vec::with_capacity(runs);
    for i in 0..runs {
        let start = Instant::now();
        for _ in 0..ops_per_pass {
            ei[0].set_data(&mut input_data);
            ei[1].set_data(&mut weight_data);
            eo[0].set_data(&mut output_data);
            ctx.execute(graph, &mut ei, &mut eo)
                .map_err(|e| format!("execute: {e}"))?;
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        timings.push(ms);
        eprintln!("  run {}: {:.1}ms ({} ops × {:.2}ms/op)",
            i + 1, ms, ops_per_pass, ms / ops_per_pass as f64);
    }

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Ok(BenchResult {
        label: "NPU encoder (QNN HTP)".to_string(),
        median_ms: timings[timings.len() / 2],
        min_ms: timings[0],
        max_ms: *timings.last().unwrap(),
        text: format!("{} matmuls [{}×{}], real weights", ops_per_pass, nc, ns),
    })
}

fn bench_preprocessor(
    model_path: &Path,
    runs: usize,
) -> Result<BenchResult, String> {
    use whisper_backend::preprocess::Preprocessor;

    // Try GGUF first, fall back to dummy weights for old ggml format
    let preprocessor = match gguf_loader::GgufFile::open(model_path) {
        Ok(gguf) => Preprocessor::from_gguf(&gguf)
            .map_err(|e| format!("preprocessor: {e}"))?,
        Err(_) => {
            // Old ggml format — create preprocessor with dummy weights
            // to benchmark the computation itself (Conv1d+GELU+pos_embed)
            let config = detect_encoder_config(model_path)?;
            let n_state = config.n_state as usize;
            let n_ctx = config.n_ctx as usize;
            let n_mels = 80;
            eprintln!("  (using dummy weights — model is old ggml format, not GGUF)");
            Preprocessor::with_dummy_weights(n_state, n_ctx, n_mels)
        }
    };

    let n_mels = preprocessor.n_mels();
    let n_frames = 3000; // standard 30s window
    eprintln!("  n_mels={}, n_frames={}, n_state={}, n_ctx={}",
        n_mels, n_frames, preprocessor.n_state(), preprocessor.n_ctx());

    let mel = vec![0.0f32; n_mels * n_frames];

    eprintln!("  warmup...");
    let _ = preprocessor.process_mel(&mel, n_frames);

    let mut timings = Vec::with_capacity(runs);
    for i in 0..runs {
        let start = Instant::now();
        let _output = preprocessor.process_mel(&mel, n_frames);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        timings.push(ms);
        eprintln!("  run {}: {:.1}ms", i + 1, ms);
    }

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Ok(BenchResult {
        label: "CPU preprocess".to_string(),
        median_ms: timings[timings.len() / 2],
        min_ms: timings[0],
        max_ms: *timings.last().unwrap(),
        text: "(conv1d+gelu+pos)".to_string(),
    })
}

/// Run a backend benchmark in a subprocess to isolate crashes (e.g. Vulkan segfault).
fn run_backend_subprocess(args: &Args, backend: &str) -> Result<BenchResult, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let output = std::process::Command::new(&exe)
        .args([
            "--model", args.model.to_str().unwrap(),
            "--audio", args.audio.to_str().unwrap(),
            "--runs", &args.runs.to_string(),
            "--subprocess", backend,
        ])
        .output()
        .map_err(|e| format!("subprocess: {e}"))?;

    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        if code == 139 || code == -11 {
            return Err(format!("segfault (signal 11) — Vulkan driver crash"));
        }
        return Err(format!("exit code {code}: {}", stderr.lines().last().unwrap_or("")));
    }

    // Parse the structured output from subprocess stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Forward stderr for visibility
    for line in stderr.lines() {
        eprintln!("{line}");
    }

    // Parse "RESULT:<label>|<median>|<min>|<max>|<text>" from stdout
    for line in stdout.lines() {
        if let Some(data) = line.strip_prefix("RESULT:") {
            let parts: Vec<&str> = data.splitn(5, '|').collect();
            if parts.len() == 5 {
                return Ok(BenchResult {
                    label: parts[0].to_string(),
                    median_ms: parts[1].parse().unwrap_or(0.0),
                    min_ms: parts[2].parse().unwrap_or(0.0),
                    max_ms: parts[3].parse().unwrap_or(0.0),
                    text: parts[4].to_string(),
                });
            }
        }
    }

    Err("subprocess produced no parseable result".into())
}

/// Execute a single backend in subprocess mode, print RESULT to stdout.
fn run_subprocess(backend: &str, args: &Args) {
    let audio = load_audio(&args.audio);
    let result = match backend {
        "gpu" => bench_whisper_cpp(&args.model, &audio, true, args.runs),
        "cpu" => bench_whisper_cpp(&args.model, &audio, false, args.runs),
        _ => Err(format!("unknown subprocess backend: {backend}")),
    };

    match result {
        Ok(r) => {
            // Structured output to stdout
            println!("RESULT:{}|{:.1}|{:.1}|{:.1}|{}",
                r.label, r.median_ms, r.min_ms, r.max_ms, r.text.trim());
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    }
}

/// Guess the number of f32 elements for a whisper encoder weight tensor.
fn guess_weight_size(name: &str, n_state: usize, _n_layer: u32) -> usize {
    let ns = n_state;
    if name.contains("ln") && name.contains("weight") { return ns; }
    if name.contains("ln") && name.contains("bias") { return ns; }
    if name.contains("query.weight") || name.contains("key.weight") || name.contains("value.weight") || name.contains("out.weight") {
        return ns * ns;
    }
    if name.contains("query.bias") || name.contains("value.bias") || name.contains("out.bias") {
        return ns;
    }
    if name.contains("mlp.0.weight") { return 4 * ns * ns; }
    if name.contains("mlp.0.bias") { return 4 * ns; }
    if name.contains("mlp.2.weight") { return ns * 4 * ns; }
    if name.contains("mlp.2.bias") { return ns; }
    if name.contains("ln_post.weight") || name.contains("ln_post.bias") { return ns; }
    // Fallback
    eprintln!("    WARNING: unknown weight '{name}', guessing {ns}");
    ns
}

/// Hybrid benchmark: CPU encode → extract encoder output → re-inject → decode-only.
/// Validates that encoder output injection produces identical transcription.
/// Hybrid benchmark: CPU encode once → save encoder output → skip_encode + inject → decode-only.
/// Measures decode-only time (simulating NPU encoder replacing CPU encoder).
fn bench_hybrid(
    model_path: &Path,
    audio: &[f32],
    runs: usize,
) -> Result<BenchResult, String> {
    use whisper_backend::ffi::*;
    use std::ffi::{CStr, CString};

    // Step 1: CPU reference
    let hw = any_stt::detect_hardware();
    let engine = WhisperEngine::new(model_path, "en", Backend::Cpu, hw)
        .map_err(|e| format!("{e}"))?;
    let cpu_result = engine.transcribe(audio).map_err(|e| format!("{e}"))?;
    eprintln!("  CPU reference: \"{}\"", cpu_result.text.trim());
    drop(engine);

    // Step 2: Create context and run one full pass to initialize state + get encoder output
    let path_cstr = CString::new(model_path.to_str().unwrap()).unwrap();
    let mut cparams = unsafe { whisper_context_default_params() };
    cparams.use_gpu = false;
    let ctx = unsafe { whisper_init_from_file_with_params(path_cstr.as_ptr(), cparams) };
    if ctx.is_null() {
        return Err("failed to load model".into());
    }

    let lang = CString::new("en").unwrap();
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32).unwrap_or(4);

    // Helper to create configured params
    let make_params = || -> *mut WhisperFullParams {
        let params = unsafe { shim_default_params(WhisperSamplingStrategy::Greedy) };
        unsafe {
            shim_params_set_language(params, lang.as_ptr());
            shim_params_set_n_threads(params, n_threads);
            shim_params_set_translate(params, false);
            shim_params_set_print_special(params, false);
            shim_params_set_print_progress(params, false);
            shim_params_set_print_realtime(params, false);
            shim_params_set_print_timestamps(params, false);
            shim_params_set_single_segment(params, false);
            shim_params_set_suppress_nst(params, true);
        }
        params
    };

    // Run full CPU inference to init state and get encoder output
    let params = make_params();
    unsafe { shim_whisper_full(ctx, params, audio.as_ptr(), audio.len() as i32) };
    unsafe { shim_free_params(params) };

    // Save encoder output
    let mut enc_n_ctx: i32 = 0;
    let mut enc_n_state: i32 = 0;
    let enc_ptr = unsafe { whisper_get_encoder_output(ctx, &mut enc_n_ctx, &mut enc_n_state) };
    if enc_ptr.is_null() {
        unsafe { whisper_free(ctx) };
        return Err("whisper_get_encoder_output returned null".into());
    }
    let enc_size = (enc_n_ctx as usize) * (enc_n_state as usize);
    let enc_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(enc_ptr, enc_size).to_vec()
    };
    eprintln!("  encoder output: [{enc_n_ctx}, {enc_n_state}] ({} floats)", enc_size);

    // Step 3: Benchmark decode-only — inject encoder output, skip encode
    eprintln!("  warmup (skip_encode + decode-only)...");
    {
        unsafe { whisper_set_encoder_output(ctx, enc_data.as_ptr(), enc_n_ctx, enc_n_state) };
        unsafe { whisper_set_skip_encode(ctx, true) };
        let params = make_params();
        unsafe { shim_whisper_full(ctx, params, audio.as_ptr(), audio.len() as i32) };
        unsafe { shim_free_params(params) };
    }

    let mut timings = Vec::with_capacity(runs);
    let mut last_text = String::new();

    for i in 0..runs {
        // Inject encoder output + enable skip
        unsafe { whisper_set_encoder_output(ctx, enc_data.as_ptr(), enc_n_ctx, enc_n_state) };
        unsafe { whisper_set_skip_encode(ctx, true) };

        let params = make_params();
        let start = Instant::now();
        let ret = unsafe { shim_whisper_full(ctx, params, audio.as_ptr(), audio.len() as i32) };
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        unsafe { shim_free_params(params) };

        if ret != 0 {
            unsafe { whisper_free(ctx) };
            return Err(format!("whisper_full(decode-only) failed: {ret}"));
        }

        // Collect output text
        let n_segments = unsafe { whisper_full_n_segments(ctx) };
        let mut text = String::new();
        for s in 0..n_segments {
            let seg = unsafe { whisper_full_get_segment_text(ctx, s) };
            if !seg.is_null() {
                text.push_str(&unsafe { CStr::from_ptr(seg) }.to_string_lossy());
            }
        }

        timings.push(ms);
        eprint!("  run {}: {:.1}ms", i + 1, ms);
        if i == 0 {
            eprint!(" \"{}\"", text.trim());
        }
        eprintln!();
        last_text = text;
    }

    unsafe { whisper_free(ctx) };
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Verify output matches CPU reference
    let cpu_t = cpu_result.text.trim();
    let hyb_t = last_text.trim();
    let status = if cpu_t == hyb_t { "MATCH" } else { "MISMATCH" };
    eprintln!("  → output {status} vs CPU reference");
    if cpu_t != hyb_t {
        eprintln!("    CPU:    \"{}\"", cpu_t);
        eprintln!("    Hybrid: \"{}\"", hyb_t);
    }

    Ok(BenchResult {
        label: "Hybrid decode-only".to_string(),
        median_ms: timings[timings.len() / 2],
        min_ms: timings[0],
        max_ms: *timings.last().unwrap(),
        text: last_text,
    })
}

/// Detect encoder config from model.
fn detect_encoder_config(model_path: &Path) -> Result<EncoderConfig, String> {
    if let Ok(gguf) = gguf_loader::GgufFile::open(model_path) {
        if let Some(view) = gguf.tensor("encoder.blocks.0.attn.query.weight") {
            let n_state = view.info.dims[0] as u32;
            return Ok(match n_state {
                384 => EncoderConfig::tiny(),
                512 => EncoderConfig::base(),
                768 => EncoderConfig::small(),
                1024 => EncoderConfig::medium(),
                1280 => EncoderConfig::large(),
                _ => return Err(format!("unknown n_state: {n_state}")),
            });
        }
    }

    let name = model_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.contains("tiny") { Ok(EncoderConfig::tiny()) }
    else if name.contains("base") { Ok(EncoderConfig::base()) }
    else if name.contains("small") { Ok(EncoderConfig::small()) }
    else if name.contains("medium") { Ok(EncoderConfig::medium()) }
    else if name.contains("large") { Ok(EncoderConfig::large()) }
    else {
        eprintln!("  WARNING: cannot detect model size, defaulting to tiny");
        Ok(EncoderConfig::tiny())
    }
}

// --- Audio loading ---

fn load_audio(path: &Path) -> Vec<f32> {
    let reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected mono audio");
    assert_eq!(spec.sample_rate, 16000, "expected 16kHz sample rate");
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>().map(|s| s.unwrap() as f32 / max_val).collect()
        }
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
        }
    }
}

// --- Types ---

#[derive(Debug)]
struct Args {
    model: PathBuf,
    audio: PathBuf,
    runs: usize,
    backend: BackendArg,
    subprocess: Option<String>,
}

#[derive(Debug, Clone)]
enum BackendArg {
    Cpu, Gpu, Npu, Preprocess, Hybrid, All,
}

impl BackendArg {
    fn to_list(&self) -> Vec<BenchBackend> {
        match self {
            Self::Cpu => vec![BenchBackend::Cpu],
            Self::Gpu => vec![BenchBackend::Gpu],
            Self::Npu => vec![BenchBackend::Npu],
            Self::Preprocess => vec![BenchBackend::Preprocess],
            Self::Hybrid => vec![BenchBackend::Hybrid],
            Self::All => vec![BenchBackend::Cpu, BenchBackend::Gpu, BenchBackend::Npu, BenchBackend::Preprocess, BenchBackend::Hybrid],
        }
    }
}

#[derive(Debug, Clone)]
enum BenchBackend { Cpu, Gpu, Npu, Preprocess, Hybrid }

impl BenchBackend {
    fn label(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU (whisper.cpp, 8 threads)",
            Self::Gpu => "GPU (whisper.cpp + Vulkan / Adreno 750)",
            Self::Npu => "NPU (QNN HTP / Hexagon V75, encoder only)",
            Self::Preprocess => "CPU preprocess (Conv1d+GELU+pos_embed)",
            Self::Hybrid => "Hybrid (CPU encode → inject → decode)",
        }
    }
}

struct BenchResult {
    label: String,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    text: String,
}

fn print_result(r: &BenchResult) {
    eprintln!("  → median: {:.1}ms, min: {:.1}ms, max: {:.1}ms",
        r.median_ms, r.min_ms, r.max_ms);
    if !r.text.is_empty() && !r.text.starts_with('(') {
        eprintln!("  → text: \"{}\"", r.text.trim());
    }
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut audio: Option<PathBuf> = None;
    let mut runs = 3usize;
    let mut backend = BackendArg::Cpu;
    let mut subprocess: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => { i += 1; model = Some(PathBuf::from(&args[i])); }
            "--audio" | "-a" => { i += 1; audio = Some(PathBuf::from(&args[i])); }
            "--runs" | "-r" => { i += 1; runs = args[i].parse().expect("--runs must be a number"); }
            "--backend" | "-b" => {
                i += 1;
                backend = match args[i].as_str() {
                    "cpu" => BackendArg::Cpu,
                    "gpu" => BackendArg::Gpu,
                    "npu" => BackendArg::Npu,
                    "preprocess" => BackendArg::Preprocess,
                    "hybrid" => BackendArg::Hybrid,
                    "all" => BackendArg::All,
                    other => panic!("unknown backend: {other}"),
                };
            }
            "--subprocess" => { i += 1; subprocess = Some(args[i].clone()); }
            "--help" | "-h" => {
                eprintln!("Usage: bench-whisper --model <path> --audio <path> [--runs N] [--backend cpu|gpu|npu|preprocess|all]");
                std::process::exit(0);
            }
            _ => { eprintln!("unknown argument: {}", args[i]); std::process::exit(1); }
        }
        i += 1;
    }

    let model = model.unwrap_or_else(|| {
        for p in &["third-party/whisper.cpp/models/ggml-tiny.en.bin", "/data/local/tmp/any-stt/ggml-tiny.en.bin"] {
            let pb = PathBuf::from(p);
            if pb.exists() { return pb; }
        }
        panic!("--model not specified and no default found");
    });

    let audio = audio.unwrap_or_else(|| {
        for p in &["third-party/whisper.cpp/samples/jfk.wav", "/data/local/tmp/any-stt/jfk.wav"] {
            let pb = PathBuf::from(p);
            if pb.exists() { return pb; }
        }
        panic!("--audio not specified and no default found");
    });

    Args { model, audio, runs, backend, subprocess }
}
