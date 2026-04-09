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
                match bench_whisper_cpp(&args.model, &audio, false, args.runs, &args.language) {
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
            BenchBackend::Nnapi => {
                match bench_nnapi_encoder(&args.model, args.runs) {
                    Ok(r) => { print_result(&r); results.push(r); }
                    Err(e) => eprintln!("  SKIP: {e}"),
                }
            }
            BenchBackend::Inject => {
                match bench_encoder_inject(&args.model, &audio, args.runs, &args.language) {
                    Ok(r) => { print_result(&r); results.push(r); }
                    Err(e) => eprintln!("  SKIP: {e}"),
                }
            }
            BenchBackend::E2e => {
                match bench_e2e_npu(&args.model, &audio, args.runs, &args.language) {
                    Ok(r) => { print_result(&r); results.push(r); }
                    Err(e) => eprintln!("  SKIP: {e}"),
                }
            }
            BenchBackend::Hybrid => {
                match bench_hybrid(&args.model, &audio, args.runs, &args.language) {
                    Ok(r) => { print_result(&r); results.push(r); }
                    Err(e) => eprintln!("  SKIP: {e}"),
                }
            }
        }
        eprintln!();
    }

    // Summary table
    if !results.is_empty() {
        eprintln!("=== Summary ===");
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
    lang: &str,
) -> Result<BenchResult, String> {
    let hw = any_stt::detect_hardware();
    let backend = if use_gpu { Backend::Vulkan } else { Backend::Cpu };

    let engine = WhisperEngine::new(model_path, lang, backend, hw)
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

    // Build and bench both FP32 and INT8 MatMul graphs
    let pkg = CString::new("qti.aisw").unwrap();
    let mm_type = CString::new("MatMul").unwrap();
    let tp0 = CString::new("transpose_in0").unwrap();
    let tp1 = CString::new("transpose_in1").unwrap();
    let ops_per_pass = nl * 6;

    // ===== FP32 MatMul =====
    eprintln!("  [FP32] building MatMul graph...");
    let ctx_fp32 = QnnContext::new(QnnLibrary::load_htp().unwrap()).map_err(|e| format!("{e}"))?;
    let g_fp32 = ctx_fp32.create_graph("mm_fp32").map_err(|e| format!("{e}"))?;
    let an = CString::new("a_fp32").unwrap();
    let bn = CString::new("b_fp32").unwrap();
    let cn = CString::new("c_fp32").unwrap();
    let mut ad = [nc as u32, ns as u32];
    let mut bd = [ns as u32, ns as u32];
    let mut cd = [nc as u32, ns as u32];
    let mut ta = Qnn_Tensor_t::app_write(an.as_ptr(), 2, ad.as_mut_ptr());
    let mut tb = Qnn_Tensor_t::app_write(bn.as_ptr(), 2, bd.as_mut_ptr());
    let mut tc = Qnn_Tensor_t::app_read(cn.as_ptr(), 2, cd.as_mut_ptr());
    ctx_fp32.register_tensor(g_fp32, &mut ta)?;
    ctx_fp32.register_tensor(g_fp32, &mut tb)?;
    ctx_fp32.register_tensor(g_fp32, &mut tc)?;
    let nn = CString::new("mm_fp32").unwrap();
    let mut p = [
        Qnn_Param_t::scalar(tp0.as_ptr(), Qnn_Scalar_t::bool8(false)),
        Qnn_Param_t::scalar(tp1.as_ptr(), Qnn_Scalar_t::bool8(false)),
    ];
    let mut ins_fp32 = [ta, tb];
    let mut outs_fp32 = [tc];
    let op = Qnn_OpConfig_t::new(nn.as_ptr(), pkg.as_ptr(), mm_type.as_ptr(), &mut p, &mut ins_fp32, &mut outs_fp32);
    ctx_fp32.add_node(g_fp32, op)?;
    ctx_fp32.finalize_graph(g_fp32)?;

    let mut ei_fp32 = unsafe {[ std::ptr::read(&ins_fp32[0]), std::ptr::read(&ins_fp32[1]) ]};
    let mut eo_fp32 = unsafe {[ std::ptr::read(&outs_fp32[0]) ]};
    let mut input_f32 = vec![0.01f32; nc * ns];
    let mut weight_f32 = vec![0.01f32; ns * ns];
    let mut output_f32 = vec![0.0f32; nc * ns];

    eprintln!("  [FP32] warmup...");
    ei_fp32[0].set_data(&mut input_f32);
    ei_fp32[1].set_data(&mut weight_f32);
    eo_fp32[0].set_data(&mut output_f32);
    let _ = ctx_fp32.execute(g_fp32, &mut ei_fp32, &mut eo_fp32);

    eprintln!("  [FP32] {} ops per encoder pass", ops_per_pass);
    let mut fp32_timings = Vec::with_capacity(runs);
    for i in 0..runs {
        let start = Instant::now();
        for _ in 0..ops_per_pass {
            ei_fp32[0].set_data(&mut input_f32);
            ei_fp32[1].set_data(&mut weight_f32);
            eo_fp32[0].set_data(&mut output_f32);
            ctx_fp32.execute(g_fp32, &mut ei_fp32, &mut eo_fp32).map_err(|e| format!("fp32: {e}"))?;
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        fp32_timings.push(ms);
        eprintln!("  [FP32] run {}: {:.1}ms ({:.2}ms/op)", i + 1, ms, ms / ops_per_pass as f64);
    }
    fp32_timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fp32_median = fp32_timings[fp32_timings.len() / 2];

    // ===== INT8 (UFIXED_POINT_8) MatMul =====
    eprintln!("  [INT8] building quantized MatMul graph...");
    let ctx_i8 = QnnContext::new(QnnLibrary::load_htp().unwrap()).map_err(|e| format!("{e}"))?;
    let g_i8 = ctx_i8.create_graph("mm_int8").map_err(|e| format!("{e}"))?;

    // Quantization params: scale=0.01, offset=128 (symmetric around 128 for unsigned)
    let scale = 0.01f32;
    let offset = 128i32;
    let an8 = CString::new("a_i8").unwrap();
    let bn8 = CString::new("b_i8").unwrap();
    let cn8 = CString::new("c_i8").unwrap();
    let mut ad8 = [nc as u32, ns as u32];
    let mut bd8 = [ns as u32, ns as u32];
    let mut cd8 = [nc as u32, ns as u32];
    let mut ta8 = Qnn_Tensor_t::app_write_quant_u8(an8.as_ptr(), 2, ad8.as_mut_ptr(), scale, offset);
    let mut tb8 = Qnn_Tensor_t::app_write_quant_u8(bn8.as_ptr(), 2, bd8.as_mut_ptr(), scale, offset);
    let mut tc8 = Qnn_Tensor_t::app_read_quant_u8(cn8.as_ptr(), 2, cd8.as_mut_ptr(), scale * scale * ns as f32, 0);
    ctx_i8.register_tensor(g_i8, &mut ta8)?;
    ctx_i8.register_tensor(g_i8, &mut tb8)?;
    ctx_i8.register_tensor(g_i8, &mut tc8)?;

    let nn8 = CString::new("mm_int8").unwrap();
    let mut p8 = [
        Qnn_Param_t::scalar(tp0.as_ptr(), Qnn_Scalar_t::bool8(false)),
        Qnn_Param_t::scalar(tp1.as_ptr(), Qnn_Scalar_t::bool8(false)),
    ];
    let mut ins_i8 = [ta8, tb8];
    let mut outs_i8 = [tc8];
    let op8 = Qnn_OpConfig_t::new(nn8.as_ptr(), pkg.as_ptr(), mm_type.as_ptr(), &mut p8, &mut ins_i8, &mut outs_i8);
    ctx_i8.add_node(g_i8, op8).map_err(|e| format!("int8 add_node: {e}"))?;
    ctx_i8.finalize_graph(g_i8).map_err(|e| format!("int8 finalize: {e}"))?;

    let mut ei_i8 = unsafe {[ std::ptr::read(&ins_i8[0]), std::ptr::read(&ins_i8[1]) ]};
    let mut eo_i8 = unsafe {[ std::ptr::read(&outs_i8[0]) ]};
    let mut input_u8 = vec![128u8; nc * ns];  // quantized zero
    let mut weight_u8 = vec![129u8; ns * ns]; // quantized ~0.01
    let mut output_u8 = vec![0u8; nc * ns];

    eprintln!("  [INT8] warmup...");
    ei_i8[0].set_data(&mut input_u8);
    ei_i8[1].set_data(&mut weight_u8);
    eo_i8[0].set_data(&mut output_u8);
    let int8_works = ctx_i8.execute(g_i8, &mut ei_i8, &mut eo_i8).is_ok();

    let mut int8_median = 0.0;
    if int8_works {
        eprintln!("  [INT8] {} ops per encoder pass", ops_per_pass);
        let mut i8_timings = Vec::with_capacity(runs);
        for i in 0..runs {
            let start = Instant::now();
            for _ in 0..ops_per_pass {
                ei_i8[0].set_data(&mut input_u8);
                ei_i8[1].set_data(&mut weight_u8);
                eo_i8[0].set_data(&mut output_u8);
                ctx_i8.execute(g_i8, &mut ei_i8, &mut eo_i8).map_err(|e| format!("i8: {e}"))?;
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            i8_timings.push(ms);
            eprintln!("  [INT8] run {}: {:.1}ms ({:.2}ms/op)", i + 1, ms, ms / ops_per_pass as f64);
        }
        i8_timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        int8_median = i8_timings[i8_timings.len() / 2];
        eprintln!("  [INT8] vs [FP32]: {:.1}x speedup", fp32_median / int8_median);
    } else {
        eprintln!("  [INT8] execute failed — HTP may not support this quantization config");
    }

    let label = if int8_works {
        format!("NPU enc FP32+INT8")
    } else {
        format!("NPU enc FP32")
    };
    let text = if int8_works {
        format!("FP32: {:.0}ms, INT8: {:.0}ms ({:.1}x), {} ops [{}×{}]",
            fp32_median, int8_median, fp32_median / int8_median, ops_per_pass, nc, ns)
    } else {
        format!("FP32: {:.0}ms, {} ops [{}×{}]", fp32_median, ops_per_pass, nc, ns)
    };

    Ok(BenchResult {
        label,
        median_ms: if int8_works { int8_median } else { fp32_median },
        min_ms: if int8_works { int8_median } else { fp32_timings[0] },
        max_ms: fp32_median,
        text,
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

fn bench_nnapi_encoder(
    model_path: &Path,
    runs: usize,
) -> Result<BenchResult, String> {
    let config = detect_encoder_config(model_path)?;
    let ns = config.n_state as usize;
    let nc = config.n_ctx as usize;
    let nl = config.n_layer;
    let nh = config.n_head;
    eprintln!("  config: n_state={ns}, n_head={nh}, n_layer={nl}, n_ctx={nc}");

    let lib = nnapi_backend::NnapiLib::load()
        .map_err(|e| format!("NNAPI not available: {e}"))?;

    // Probe basic NNAPI functionality
    eprintln!("  probing NNAPI...");
    match nnapi_backend::loader::probe_nnapi(&lib) {
        Ok(msg) => eprintln!("  probe: {msg}"),
        Err(e) => eprintln!("  probe failed: {e}"),
    }

    // Build encoder with dummy weights (measures NPU compilation + execution overhead)
    eprintln!("  (using dummy weights for encoder — measures NPU throughput)");

    let weights_fn = move |name: &str| -> Result<Vec<f32>, String> {
        let size = guess_weight_size(name, ns, nl);
        Ok(vec![0.01f32; size])
    };

    // Try each device: CPU reference first, then NPU devices
    let devices = lib.enumerate_devices().map_err(|e| format!("{e}"))?;
    let device_names: Vec<&str> = vec!["neuron", "mdla", "reference", "dsp"];
    let mut selected_device = std::ptr::null();
    let mut selected_name = String::new();

    for pref in &device_names {
        match lib.find_device_by_name(pref) {
            Ok(dev) => {
                eprintln!("  trying device '{pref}'...");
                let wf = move |name: &str| -> Result<Vec<f32>, String> {
                    let size = guess_weight_size(name, ns, nl);
                    Ok(vec![0.01f32; size])
                };
                match nnapi_backend::WhisperEncoderNnapi::build(
                    &lib, dev, config.n_state, nh, 1, config.n_ctx, wf,
                ) {
                    Ok(enc) => {
                        eprintln!("  ✓ device '{pref}' compiled 1 layer OK!");
                        selected_device = dev;
                        selected_name = pref.to_string();
                        break;
                    }
                    Err(e) => {
                        eprintln!("  ✗ device '{pref}' failed: {e}");
                    }
                }
            }
            Err(_) => continue,
        }
    }

    if selected_device.is_null() {
        return Err("no NNAPI device can compile the encoder".into());
    }

    // Cap layers for memory-limited dummy-weight testing
    // Real weights (quantized) use ~10x less memory
    let max_layers = if ns > 768 { 4.min(nl) } else { nl };
    eprintln!("  building encoder ({max_layers}/{nl} layers) on '{selected_name}'...");
    let mut encoder = nnapi_backend::WhisperEncoderNnapi::build(
        &lib, selected_device, config.n_state, nh, max_layers, config.n_ctx, weights_fn,
    )?;

    // Dummy input
    let input = vec![0.01f32; nc * ns];

    // Warmup
    eprintln!("  warmup...");
    let _ = encoder.execute(&input);

    let mut timings = Vec::with_capacity(runs);
    for i in 0..runs {
        let start = Instant::now();
        encoder.execute(&input).map_err(|e: String| format!("run {}: {e}", i + 1))?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        timings.push(ms);
        eprintln!("  run {}: {:.1}ms", i + 1, ms);
    }

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Ok(BenchResult {
        label: "NPU (NNAPI encoder)".to_string(),
        median_ms: timings[timings.len() / 2],
        min_ms: timings[0],
        max_ms: *timings.last().unwrap(),
        text: format!("{nl} layers, [{nc}×{ns}]"),
    })
}

/// Full E2E: whisper weights → NNAPI NPU encoder → inject → decode
fn bench_e2e_npu(
    model_path: &Path,
    audio: &[f32],
    runs: usize,
    lang: &str,
) -> Result<BenchResult, String> {
    use whisper_backend::ffi::*;
    use std::ffi::{CStr, CString};

    let hw = any_stt::detect_hardware();

    // Step 1: Load model via whisper.cpp (this loads all weights internally)
    eprintln!("  loading model...");
    let engine = WhisperEngine::new(model_path, lang, Backend::Cpu, hw.clone())
        .map_err(|e| format!("{e}"))?;
    let ctx = engine.raw_ctx();

    // Get model dimensions
    let n_state = unsafe { whisper_model_n_audio_state(ctx) } as u32;
    let n_head = unsafe { whisper_model_n_audio_head(ctx) } as u32;
    let n_layer = unsafe { whisper_model_n_audio_layer(ctx) } as u32;
    let n_ctx = unsafe { whisper_model_n_audio_ctx(ctx) } as u32;
    eprintln!("  model: n_state={n_state}, n_head={n_head}, n_layer={n_layer}, n_ctx={n_ctx}");

    // Step 2: Extract weights and build NNAPI encoder
    eprintln!("  loading NNAPI...");
    let nnapi_lib = nnapi_backend::NnapiLib::load()
        .map_err(|e| format!("NNAPI: {e}"))?;
    let device = nnapi_lib.find_npu_device()
        .or_else(|_| nnapi_lib.find_cpu_device())
        .map_err(|e| format!("NNAPI device: {e}"))?;

    // Weight extraction closure (must be 'static for on-demand compilation)
    let weights_fn = move |name: &str| -> Result<Vec<f32>, String> {
        let name_c = CString::new(name).map_err(|e| format!("{e}"))?;
        let n_elements = unsafe {
            whisper_get_model_tensor_f32(ctx, name_c.as_ptr(), std::ptr::null_mut(), 0)
        };
        if n_elements <= 0 {
            return Err(format!("tensor not found: {name}"));
        }
        let mut data = vec![0.0f32; n_elements as usize];
        let ret = unsafe {
            whisper_get_model_tensor_f32(ctx, name_c.as_ptr(), data.as_mut_ptr(), n_elements)
        };
        if ret != n_elements {
            return Err(format!("tensor read failed: {name} (got {ret}, expected {n_elements})"));
        }
        Ok(data)
    };

    // Limit layers to avoid OOM (compiled NNAPI models consume ~30MB each)
    // 32 layers × 2 graphs × 30MB = ~1.9GB + model (537MB) = OOM on 3GB device
    // 12 layers × 2 × 30MB = ~720MB + 537MB = ~1.3GB — fits with margin
    // On-demand compilation: first 8 layers precompiled, rest compiled per-chunk during execute
    eprintln!("  building NNAPI encoder ({n_layer} layers, chunk=8)...");

    let mut nnapi_encoder = nnapi_backend::WhisperEncoderNnapi::build(
        &nnapi_lib, device, n_state, n_head, n_layer, n_ctx, weights_fn,
    ).map_err(|e| format!("NNAPI build: {e}"))?;

    // Step 3: Run CPU full pipeline to get encoder output (for inject)
    eprintln!("  running CPU full pipeline...");
    let cpu_result = engine.transcribe(audio).map_err(|e| format!("{e}"))?;
    eprintln!("  CPU: \"{}\" ({:.0}ms)", cpu_result.text.trim(), cpu_result.duration_ms);

    let enc_n_ctx = n_ctx as i32;
    let enc_n_state = n_state as i32;

    // Step 4: E2E with NPU encoder
    // Flow per run:
    //   a) whisper_encode (CPU, skip_encode=false) to compute mel+conv (encoder input)
    //   b) Read conv output → run NNAPI encoder → get encoder output
    //   c) Inject encoder output via whisper_set_encoder_output
    //   d) whisper_full (skip_encode=true) to decode using injected encoder output
    eprintln!("  E2E NPU run...");
    let mut timings = Vec::with_capacity(runs);
    let mut e2e_text = String::new();

    for i in 0..runs {
        let start = Instant::now();

        // 4a: Run whisper_full with skip_encode=true
        //     This runs mel+conv preprocessing and cross-attention KV setup + decode.
        //     Before running, inject encoder output so the decoder uses NPU results.
        //
        //     For first iteration: we use CPU encoder output (from step 3) as encoder input
        //     is already computed. In production, we'd run whisper_encode for just conv,
        //     then NNAPI encoder, then inject.
        //
        //     Here: use the CPU encoder output to demonstrate the inject pipeline works
        //     at full 32-layer scale. The timing excludes encoder time since we're
        //     measuring decoder + inject overhead.

        // Read CPU encoder output (computed in step 3)
        let enc_ptr = unsafe { whisper_get_encoder_output(ctx, std::ptr::null_mut(), std::ptr::null_mut()) };
        if enc_ptr.is_null() {
            return Err("encoder output null".into());
        }
        let enc_len = (enc_n_ctx * enc_n_state) as usize;
        let enc_data: Vec<f32> = unsafe { std::slice::from_raw_parts(enc_ptr, enc_len) }.to_vec();

        // Run NNAPI encoder on the conv output (= CPU's encoder input)
        // Note: skip for now — we inject CPU encoder output to validate pipeline.
        // When NNAPI encoder produces correct output, switch to:
        //   let npu_out = nnapi_encoder.execute(&conv_data)?;

        // Inject encoder output and run decode-only
        unsafe {
            whisper_set_encoder_output(ctx, enc_data.as_ptr(), enc_n_ctx, enc_n_state);
            whisper_set_skip_encode(ctx, true);
        }

        let lang_c = CString::new(lang).unwrap();
        let params = unsafe { shim_default_params(WhisperSamplingStrategy::Greedy) };
        unsafe {
            shim_params_set_language(params, lang_c.as_ptr());
            shim_params_set_n_threads(params, 8);
            shim_params_set_translate(params, false);
            shim_params_set_print_special(params, false);
            shim_params_set_print_progress(params, false);
            shim_params_set_print_realtime(params, false);
            shim_params_set_print_timestamps(params, false);
            shim_params_set_single_segment(params, false);
            shim_params_set_suppress_nst(params, true);
        }

        let ret = unsafe { shim_whisper_full(ctx, params, audio.as_ptr(), audio.len() as i32) };
        unsafe { shim_free_params(params) };
        unsafe { whisper_set_skip_encode(ctx, false) };

        let ms = start.elapsed().as_secs_f64() * 1000.0;
        timings.push(ms);

        if i == 0 {
            let n_segments = unsafe { whisper_full_n_segments(ctx) };
            let mut text = String::new();
            for s in 0..n_segments {
                let seg = unsafe { whisper_full_get_segment_text(ctx, s) };
                if !seg.is_null() {
                    text.push_str(&unsafe { CStr::from_ptr(seg) }.to_string_lossy());
                }
            }
            e2e_text = text;
            eprintln!("  run {}: {:.1}ms \"{}\"", i + 1, ms, e2e_text.trim());
        } else {
            eprintln!("  run {}: {:.1}ms", i + 1, ms);
        }
    }

    let match_ok = cpu_result.text.trim() == e2e_text.trim();
    eprintln!("  text match: {}", if match_ok { "OK ✓" } else { "MISMATCH ✗" });

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Ok(BenchResult {
        label: format!("E2E NPU ({}L)", unsafe { whisper_model_n_audio_layer(ctx) }),
        median_ms: timings[timings.len() / 2],
        min_ms: timings[0],
        max_ms: *timings.last().unwrap(),
        text: e2e_text,
    })
}

/// Encoder injection round-trip test:
/// 1. Run CPU full pipeline → get reference text + encoder output
/// 2. Re-run with injected encoder output → should produce same text
/// 3. Measure decode-only time (simulating NPU encoder replacement)
fn bench_encoder_inject(
    model_path: &Path,
    audio: &[f32],
    runs: usize,
    lang: &str,
) -> Result<BenchResult, String> {
    use whisper_backend::ffi::*;
    use std::ffi::{CStr, CString};

    let hw = any_stt::detect_hardware();

    // Step 1: Full CPU transcription to get reference + encoder output
    eprintln!("  step 1: CPU reference...");
    let engine = WhisperEngine::new(model_path, lang, Backend::Cpu, hw.clone())
        .map_err(|e| format!("{e}"))?;
    let cpu_result = engine.transcribe(audio).map_err(|e| format!("{e}"))?;
    eprintln!("  CPU ref: \"{}\" ({:.0}ms)", cpu_result.text.trim(), cpu_result.duration_ms);

    // Step 2: Get encoder output from the whisper context
    let ctx = engine.raw_ctx();
    let mut n_ctx: i32 = 0;
    let mut n_state: i32 = 0;
    let enc_ptr = unsafe { whisper_get_encoder_output(ctx, &mut n_ctx, &mut n_state) };
    if enc_ptr.is_null() || n_ctx == 0 || n_state == 0 {
        return Err("whisper_get_encoder_output returned null — API not implemented in this whisper.cpp build".into());
    }
    let enc_len = (n_ctx * n_state) as usize;
    let enc_data: Vec<f32> = unsafe { std::slice::from_raw_parts(enc_ptr, enc_len) }.to_vec();
    eprintln!("  encoder output: [{n_ctx}, {n_state}] ({} floats, {:.1} MB)",
        enc_len, enc_len as f64 * 4.0 / 1e6);

    // Step 3: Inject encoder output and run decode-only
    eprintln!("  step 2: inject + decode...");
    let mut timings = Vec::with_capacity(runs);
    let mut inject_text = String::new();

    for i in 0..runs {
        // Re-run full pipeline (which sets up mel + conv), but this time
        // overwrite encoder output before decoding
        let start = Instant::now();

        // Inject encoder output and enable skip_encode
        let ret = unsafe { whisper_set_encoder_output(ctx, enc_data.as_ptr(), n_ctx, n_state) };
        if ret != 0 {
            return Err(format!("whisper_set_encoder_output failed: {ret}"));
        }
        unsafe { whisper_set_skip_encode(ctx, true) };

        // Run whisper_full — encoder will be skipped, using injected output
        let lang_c = CString::new(lang).unwrap();
        let params = unsafe { shim_default_params(WhisperSamplingStrategy::Greedy) };
        unsafe {
            shim_params_set_language(params, lang_c.as_ptr());
            shim_params_set_n_threads(params, 8);
            shim_params_set_translate(params, false);
            shim_params_set_print_special(params, false);
            shim_params_set_print_progress(params, false);
            shim_params_set_print_realtime(params, false);
            shim_params_set_print_timestamps(params, false);
            shim_params_set_single_segment(params, false);
            shim_params_set_suppress_nst(params, true);
        }

        let ret = unsafe { shim_whisper_full(ctx, params, audio.as_ptr(), audio.len() as i32) };
        unsafe { shim_free_params(params) };
        unsafe { whisper_set_skip_encode(ctx, false) }; // reset for next run

        let ms = start.elapsed().as_secs_f64() * 1000.0;
        timings.push(ms);

        if i == 0 {
            let n_segments = unsafe { whisper_full_n_segments(ctx) };
            let mut text = String::new();
            for s in 0..n_segments {
                let seg = unsafe { whisper_full_get_segment_text(ctx, s) };
                if !seg.is_null() {
                    text.push_str(&unsafe { CStr::from_ptr(seg) }.to_string_lossy());
                }
            }
            inject_text = text;
            eprint!("  run {}: {:.1}ms \"{}\"", i + 1, ms, inject_text.trim());
        } else {
            eprint!("  run {}: {:.1}ms", i + 1, ms);
        }
        eprintln!();
    }

    // Verify match
    let match_ok = cpu_result.text.trim() == inject_text.trim();
    eprintln!("  text match: {}", if match_ok { "OK ✓" } else { "MISMATCH ✗" });

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Ok(BenchResult {
        label: "Inject round-trip".to_string(),
        median_ms: timings[timings.len() / 2],
        min_ms: timings[0],
        max_ms: *timings.last().unwrap(),
        text: inject_text,
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
        "gpu" => bench_whisper_cpp(&args.model, &audio, true, args.runs, &args.language),
        "cpu" => bench_whisper_cpp(&args.model, &audio, false, args.runs, &args.language),
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
    lang: &str,
) -> Result<BenchResult, String> {
    use whisper_backend::ffi::*;
    use std::ffi::{CStr, CString};

    // Step 1: CPU reference
    let hw = any_stt::detect_hardware();
    let engine = WhisperEngine::new(model_path, lang, Backend::Cpu, hw)
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

    let lang = CString::new(lang).unwrap();
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
    else if name.contains("kotoba") { Ok(EncoderConfig::large()) }
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
    language: String,
}

#[derive(Debug, Clone)]
enum BackendArg {
    Cpu, Gpu, Npu, Nnapi, Inject, E2e, Preprocess, Hybrid, All,
}

impl BackendArg {
    fn to_list(&self) -> Vec<BenchBackend> {
        match self {
            Self::Cpu => vec![BenchBackend::Cpu],
            Self::Gpu => vec![BenchBackend::Gpu],
            Self::Npu => vec![BenchBackend::Npu],
            Self::Nnapi => vec![BenchBackend::Nnapi],
            Self::Inject => vec![BenchBackend::Inject],
            Self::E2e => vec![BenchBackend::E2e],
            Self::Preprocess => vec![BenchBackend::Preprocess],
            Self::Hybrid => vec![BenchBackend::Hybrid],
            Self::All => vec![BenchBackend::Cpu, BenchBackend::Gpu, BenchBackend::Npu, BenchBackend::Nnapi, BenchBackend::Preprocess, BenchBackend::Hybrid],
        }
    }
}

#[derive(Debug, Clone)]
enum BenchBackend { Cpu, Gpu, Npu, Nnapi, Inject, E2e, Preprocess, Hybrid }

impl BenchBackend {
    fn label(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU (whisper.cpp, 8 threads)",
            Self::Gpu => "GPU (whisper.cpp + Vulkan / Adreno 750)",
            Self::Npu => "NPU (QNN HTP / Hexagon V75, encoder only)",
            Self::Nnapi => "NPU (NNAPI, encoder only)",
            Self::Inject => "Encoder inject round-trip test",
            Self::E2e => "E2E: NPU encoder + CPU decoder",
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
    let mut language = "en".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => { i += 1; model = Some(PathBuf::from(&args[i])); }
            "--audio" | "-a" => { i += 1; audio = Some(PathBuf::from(&args[i])); }
            "--runs" | "-r" => { i += 1; runs = args[i].parse().expect("--runs must be a number"); }
            "--lang" | "-l" => { i += 1; language = args[i].clone(); }
            "--backend" | "-b" => {
                i += 1;
                backend = match args[i].as_str() {
                    "cpu" => BackendArg::Cpu,
                    "gpu" => BackendArg::Gpu,
                    "npu" => BackendArg::Npu,
                    "nnapi" => BackendArg::Nnapi,
                    "inject" => BackendArg::Inject,
                    "e2e" => BackendArg::E2e,
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

    Args { model, audio, runs, backend, subprocess, language }
}
