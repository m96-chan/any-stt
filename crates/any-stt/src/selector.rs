use crate::config::{Backend, Model, Quantization, SttConfig};
use crate::hardware::{GpuVendor, HardwareInfo, NpuType, Platform};

/// Result of backend selection: which backend and quantization to use.
#[derive(Debug, Clone)]
pub struct Selection {
    pub backend: Backend,
    pub quantization: Quantization,
}

/// Select the best backend and quantization for the given hardware and config.
///
/// If the user has explicitly set `config.backend` or `config.quantization`,
/// those overrides take priority.
pub fn select(config: &SttConfig, hw: &HardwareInfo) -> Selection {
    let backend = config.backend.unwrap_or_else(|| select_backend(config, hw));
    let quantization = config
        .quantization
        .unwrap_or_else(|| select_quantization(&config.model, hw, backend));
    Selection {
        backend,
        quantization,
    }
}

/// Auto-select the best backend: NPU > GPU > CPU, with platform-specific rules.
fn select_backend(config: &SttConfig, hw: &HardwareInfo) -> Backend {
    let platform = hw.os.platform;

    // NPU check (highest priority).
    if let Some(ref npu) = hw.npu {
        if npu.available {
            match npu.npu_type {
                NpuType::CoreMl => return Backend::CoreMl,
                NpuType::QnnHtp => return Backend::Qnn,
                NpuType::Nnapi => return Backend::Nnapi,
            }
        }
    }

    // GPU check (second priority).
    if let Some(ref gpu) = hw.gpu {
        match platform {
            Platform::Linux | Platform::Windows => {
                if gpu.vendor == GpuVendor::Nvidia {
                    return Backend::Cuda;
                }
                // Vulkan: only if shader cache is warm (or user allows cold).
                if config.allow_cold_vulkan {
                    return Backend::Vulkan;
                }
                // TODO: check Vulkan shader cache existence.
            }
            Platform::MacOs | Platform::Ios => {
                return Backend::Metal;
            }
            Platform::Android => {
                // Vulkan is NEVER auto-selected on Android.
                // NNAPI was already handled above; fall through to CPU.
            }
        }
    }

    Backend::Cpu
}

/// Auto-select quantization based on available memory and model size.
fn select_quantization(model: &Model, hw: &HardwareInfo, backend: Backend) -> Quantization {
    let available_mb = match backend {
        // For GPU backends, use VRAM if available; otherwise fall back to RAM.
        Backend::Cuda | Backend::Metal | Backend::Vulkan => hw
            .gpu
            .as_ref()
            .map(|g| g.vram_mb)
            .filter(|&v| v > 0)
            .unwrap_or(hw.available_ram_mb),
        // NPU / CPU use system RAM.
        _ => hw.available_ram_mb,
    };

    let model_base_mb = model_size_estimate_f16_mb(model);

    // Pick the best quantization that fits in available memory.
    // Walk from highest quality to lowest, return first that fits.
    for &(quant, ratio) in QUANTIZATION_RATIOS {
        let estimated_mb = (model_base_mb as f64 * ratio) as u64;
        if estimated_mb <= available_mb {
            return quant;
        }
    }

    // If nothing fits, use the smallest.
    Quantization::Q4_0
}

/// Quantization formats ordered from highest quality to lowest,
/// with approximate size ratio relative to f16.
const QUANTIZATION_RATIOS: &[(Quantization, f64)] = &[
    (Quantization::F16, 1.0),
    (Quantization::Q8_0, 0.5),
    (Quantization::Q5_1, 0.35),
    (Quantization::Q5_0, 0.33),
    (Quantization::Q4_1, 0.28),
    (Quantization::Q4_0, 0.25),
];

/// Rough f16 model size estimate in MB.
fn model_size_estimate_f16_mb(model: &Model) -> u64 {
    match model {
        Model::Tiny | Model::TinyEn => 75,
        Model::Base | Model::BaseEn => 150,
        Model::Small | Model::SmallEn => 500,
        Model::DistilSmallEn => 400,
        Model::Medium | Model::MediumEn => 1500,
        Model::DistilMediumEn => 1200,
        Model::LargeV1 | Model::LargeV2 | Model::LargeV3 => 3100,
        Model::LargeV3Turbo => 1600,
        Model::DistilLargeV2 | Model::DistilLargeV3 => 1600,
        Model::KotobaV1 | Model::KotobaV2 => 1600,
        Model::Custom(_) => 1500, // Conservative default for unknown models.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::*;

    fn make_hw(
        platform: Platform,
        gpu: Option<GpuInfo>,
        npu: Option<NpuInfo>,
        ram_mb: u64,
    ) -> HardwareInfo {
        HardwareInfo {
            cpu: CpuInfo {
                arch: "x86_64".into(),
                features: vec![CpuFeature::Avx2],
                cores: 8,
            },
            gpu,
            npu,
            os: OsInfo {
                platform,
                version: String::new(),
            },
            available_ram_mb: ram_mb,
        }
    }

    fn nvidia_gpu(vram_mb: u64) -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: "RTX 4090".into(),
            vram_mb,
            driver: "555.42".into(),
        }
    }

    #[test]
    fn linux_nvidia_selects_cuda() {
        let hw = make_hw(
            Platform::Linux,
            Some(nvidia_gpu(24000)),
            None,
            32000,
        );
        let config = SttConfig::default();
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Cuda);
    }

    #[test]
    fn linux_no_gpu_selects_cpu() {
        let hw = make_hw(Platform::Linux, None, None, 16000);
        let config = SttConfig::default();
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Cpu);
    }

    #[test]
    fn macos_with_npu_selects_coreml() {
        let hw = make_hw(
            Platform::MacOs,
            Some(GpuInfo {
                vendor: GpuVendor::Apple,
                name: "Apple GPU".into(),
                vram_mb: 0,
                driver: String::new(),
            }),
            Some(NpuInfo {
                npu_type: NpuType::CoreMl,
                available: true,
            }),
            16000,
        );
        let config = SttConfig::default();
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::CoreMl);
    }

    #[test]
    fn macos_npu_unavailable_selects_metal() {
        let hw = make_hw(
            Platform::MacOs,
            Some(GpuInfo {
                vendor: GpuVendor::Apple,
                name: "Apple GPU".into(),
                vram_mb: 0,
                driver: String::new(),
            }),
            Some(NpuInfo {
                npu_type: NpuType::CoreMl,
                available: false,
            }),
            16000,
        );
        let config = SttConfig::default();
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Metal);
    }

    #[test]
    fn android_never_auto_selects_vulkan() {
        let hw = make_hw(
            Platform::Android,
            Some(GpuInfo {
                vendor: GpuVendor::Qualcomm,
                name: "Adreno 740".into(),
                vram_mb: 0,
                driver: String::new(),
            }),
            Some(NpuInfo {
                npu_type: NpuType::Nnapi,
                available: true,
            }),
            8000,
        );
        let config = SttConfig::default();
        let sel = select(&config, &hw);
        // Should pick NNAPI, never Vulkan.
        assert_eq!(sel.backend, Backend::Nnapi);
        assert_ne!(sel.backend, Backend::Vulkan);
    }

    #[test]
    fn android_without_nnapi_falls_to_cpu() {
        let hw = make_hw(
            Platform::Android,
            Some(GpuInfo {
                vendor: GpuVendor::Qualcomm,
                name: "Adreno 740".into(),
                vram_mb: 0,
                driver: String::new(),
            }),
            None,
            8000,
        );
        let config = SttConfig::default();
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Cpu);
    }

    #[test]
    fn user_override_backend_is_respected() {
        let hw = make_hw(Platform::Linux, Some(nvidia_gpu(24000)), None, 32000);
        let config = SttConfig {
            backend: Some(Backend::Cpu),
            ..Default::default()
        };
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Cpu);
    }

    #[test]
    fn user_override_quantization_is_respected() {
        let hw = make_hw(Platform::Linux, None, None, 32000);
        let config = SttConfig {
            quantization: Some(Quantization::Q4_0),
            ..Default::default()
        };
        let sel = select(&config, &hw);
        assert_eq!(sel.quantization, Quantization::Q4_0);
    }

    #[test]
    fn large_vram_selects_f16() {
        let hw = make_hw(
            Platform::Linux,
            Some(nvidia_gpu(32000)),
            None,
            64000,
        );
        let config = SttConfig {
            model: Model::LargeV3,
            ..Default::default()
        };
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Cuda);
        assert_eq!(sel.quantization, Quantization::F16);
    }

    #[test]
    fn limited_ram_selects_smaller_quant() {
        let hw = make_hw(Platform::Linux, None, None, 600);
        let config = SttConfig {
            model: Model::LargeV3,
            ..Default::default()
        };
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Cpu);
        // 3100 MB f16, 600 MB available -> need ratio ~0.19, nothing fits -> Q4_0
        assert_eq!(sel.quantization, Quantization::Q4_0);
    }

    #[test]
    fn tiny_model_fits_f16_even_low_ram() {
        let hw = make_hw(Platform::Linux, None, None, 200);
        let config = SttConfig {
            model: Model::Tiny,
            ..Default::default()
        };
        let sel = select(&config, &hw);
        // 75 MB f16 fits in 200 MB.
        assert_eq!(sel.quantization, Quantization::F16);
    }

    #[test]
    fn vulkan_cold_cache_falls_to_cpu() {
        let hw = make_hw(
            Platform::Linux,
            Some(GpuInfo {
                vendor: GpuVendor::Amd,
                name: "RX 7900 XTX".into(),
                vram_mb: 24000,
                driver: "mesa 24.1".into(),
            }),
            None,
            32000,
        );
        let config = SttConfig {
            allow_cold_vulkan: false,
            ..Default::default()
        };
        let sel = select(&config, &hw);
        // AMD on Linux without cold vulkan allowed -> CPU.
        assert_eq!(sel.backend, Backend::Cpu);
    }

    #[test]
    fn vulkan_allowed_when_cold_vulkan_true() {
        let hw = make_hw(
            Platform::Linux,
            Some(GpuInfo {
                vendor: GpuVendor::Amd,
                name: "RX 7900 XTX".into(),
                vram_mb: 24000,
                driver: "mesa 24.1".into(),
            }),
            None,
            32000,
        );
        let config = SttConfig {
            allow_cold_vulkan: true,
            ..Default::default()
        };
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Vulkan);
    }

    // --- iOS/macOS tests ---

    fn apple_gpu() -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Apple,
            name: "Apple GPU".into(),
            vram_mb: 0,
            driver: String::new(),
        }
    }

    #[test]
    fn ios_with_coreml_selects_coreml() {
        let hw = make_hw(
            Platform::Ios,
            Some(apple_gpu()),
            Some(NpuInfo { npu_type: NpuType::CoreMl, available: true }),
            6000,
        );
        let sel = select(&SttConfig::default(), &hw);
        assert_eq!(sel.backend, Backend::CoreMl);
    }

    #[test]
    fn ios_coreml_unavailable_selects_metal() {
        let hw = make_hw(
            Platform::Ios,
            Some(apple_gpu()),
            Some(NpuInfo { npu_type: NpuType::CoreMl, available: false }),
            6000,
        );
        let sel = select(&SttConfig::default(), &hw);
        assert_eq!(sel.backend, Backend::Metal);
    }

    #[test]
    fn ios_without_npu_selects_metal() {
        let hw = make_hw(
            Platform::Ios,
            Some(apple_gpu()),
            None,
            6000,
        );
        let sel = select(&SttConfig::default(), &hw);
        assert_eq!(sel.backend, Backend::Metal);
    }

    // --- Windows tests ---

    #[test]
    fn windows_nvidia_selects_cuda() {
        let hw = make_hw(Platform::Windows, Some(nvidia_gpu(24000)), None, 32000);
        let config = SttConfig::default();
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Cuda);
    }

    #[test]
    fn windows_no_gpu_selects_cpu() {
        let hw = make_hw(Platform::Windows, None, None, 16000);
        let config = SttConfig::default();
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Cpu);
    }

    #[test]
    fn windows_amd_vulkan_allowed() {
        let hw = make_hw(
            Platform::Windows,
            Some(GpuInfo {
                vendor: GpuVendor::Amd,
                name: "RX 7900 XTX".into(),
                vram_mb: 24000,
                driver: String::new(),
            }),
            None,
            32000,
        );
        let config = SttConfig {
            allow_cold_vulkan: true,
            ..Default::default()
        };
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Vulkan);
    }

    #[test]
    fn macos_with_coreml_selects_coreml() {
        let hw = make_hw(
            Platform::MacOs,
            Some(apple_gpu()),
            Some(NpuInfo { npu_type: NpuType::CoreMl, available: true }),
            16000,
        );
        let sel = select(&SttConfig::default(), &hw);
        assert_eq!(sel.backend, Backend::CoreMl);
    }

    // --- Additional quantization tests ---

    #[test]
    fn gpu_vram_used_for_quantization_selection() {
        // CUDA backend: should use VRAM (not system RAM) for quantization
        let hw = make_hw(
            Platform::Linux,
            Some(GpuInfo {
                vendor: GpuVendor::Nvidia,
                name: "RTX 3060".into(),
                vram_mb: 12000,
                driver: "555".into(),
            }),
            None,
            64000, // lots of RAM but only 12GB VRAM
        );
        let config = SttConfig {
            model: Model::LargeV3,
            ..Default::default()
        };
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Cuda);
        // 3100 MB f16, VRAM=12000 → f16 fits
        assert_eq!(sel.quantization, Quantization::F16);
    }

    #[test]
    fn gpu_zero_vram_falls_back_to_ram() {
        // Apple GPU: vram_mb=0, should fall back to system RAM
        let hw = make_hw(
            Platform::MacOs,
            Some(apple_gpu()),
            None, // no NPU → Metal
            8000,
        );
        let config = SttConfig {
            model: Model::LargeV3,
            ..Default::default()
        };
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::Metal);
        // 3100 MB f16, RAM=8000 → f16 fits
        assert_eq!(sel.quantization, Quantization::F16);
    }

    #[test]
    fn npu_uses_system_ram_for_quantization() {
        let hw = make_hw(
            Platform::MacOs,
            Some(apple_gpu()),
            Some(NpuInfo { npu_type: NpuType::CoreMl, available: true }),
            500, // limited RAM
        );
        let config = SttConfig {
            model: Model::LargeV3,
            ..Default::default()
        };
        let sel = select(&config, &hw);
        assert_eq!(sel.backend, Backend::CoreMl);
        // 3100 * ratio must fit in 500 MB → very aggressive quant
        assert_eq!(sel.quantization, Quantization::Q4_0);
    }

    #[test]
    fn android_qnn_htp_selects_qnn() {
        let hw = make_hw(
            Platform::Android,
            Some(GpuInfo {
                vendor: GpuVendor::Qualcomm,
                name: "Adreno 750".into(),
                vram_mb: 0,
                driver: String::new(),
            }),
            Some(NpuInfo { npu_type: NpuType::QnnHtp, available: true }),
            12000,
        );
        let sel = select(&SttConfig::default(), &hw);
        assert_eq!(sel.backend, Backend::Qnn);
    }

    #[test]
    fn all_model_sizes_are_nonzero() {
        let models = [
            Model::Tiny, Model::TinyEn, Model::Base, Model::BaseEn,
            Model::Small, Model::SmallEn, Model::Medium, Model::MediumEn,
            Model::LargeV1, Model::LargeV2, Model::LargeV3, Model::LargeV3Turbo,
            Model::DistilLargeV2, Model::DistilLargeV3, Model::DistilMediumEn,
            Model::DistilSmallEn, Model::KotobaV1, Model::KotobaV2,
            Model::Custom("unknown".into()),
        ];
        for m in &models {
            assert!(model_size_estimate_f16_mb(m) > 0, "model {m:?} has zero size estimate");
        }
    }

    #[test]
    fn model_sizes_are_ordered() {
        assert!(model_size_estimate_f16_mb(&Model::Tiny) < model_size_estimate_f16_mb(&Model::Base));
        assert!(model_size_estimate_f16_mb(&Model::Base) < model_size_estimate_f16_mb(&Model::Small));
        assert!(model_size_estimate_f16_mb(&Model::Small) < model_size_estimate_f16_mb(&Model::Medium));
        assert!(model_size_estimate_f16_mb(&Model::Medium) < model_size_estimate_f16_mb(&Model::LargeV3));
    }

    #[test]
    fn quantization_ratios_are_decreasing() {
        for window in QUANTIZATION_RATIOS.windows(2) {
            assert!(
                window[0].1 > window[1].1,
                "{:?} ratio {} should be > {:?} ratio {}",
                window[0].0, window[0].1, window[1].0, window[1].1
            );
        }
    }

    #[test]
    fn selection_debug_impl() {
        let sel = Selection {
            backend: Backend::Cpu,
            quantization: Quantization::Q8_0,
        };
        let dbg = format!("{sel:?}");
        assert!(dbg.contains("Cpu"));
        assert!(dbg.contains("Q8_0"));
    }
}
