/// Detected hardware information.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cpu: CpuInfo,
    pub gpu: Option<GpuInfo>,
    pub npu: Option<NpuInfo>,
    pub os: OsInfo,
    pub available_ram_mb: u64,
}

/// CPU information.
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub arch: String,
    pub features: Vec<CpuFeature>,
    pub cores: u32,
}

/// CPU SIMD / vector features relevant to inference performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuFeature {
    Avx2,
    Avx512,
    Neon,
    Sve,
}

/// GPU information.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    pub vram_mb: u64,
    pub driver: String,
}

/// GPU vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Qualcomm,
    Other,
}

/// NPU (Neural Processing Unit) information.
#[derive(Debug, Clone)]
pub struct NpuInfo {
    pub npu_type: NpuType,
    pub available: bool,
}

/// NPU type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuType {
    CoreMl,
    Nnapi,
    QnnHtp,
}

/// Operating system information.
#[derive(Debug, Clone)]
pub struct OsInfo {
    pub platform: Platform,
    pub version: String,
}

/// Target platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Linux,
    MacOs,
    Android,
    Ios,
    Windows,
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linux => write!(f, "Linux"),
            Self::MacOs => write!(f, "macOS"),
            Self::Android => write!(f, "Android"),
            Self::Ios => write!(f, "iOS"),
            Self::Windows => write!(f, "Windows"),
        }
    }
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvidia => write!(f, "NVIDIA"),
            Self::Amd => write!(f, "AMD"),
            Self::Intel => write!(f, "Intel"),
            Self::Apple => write!(f, "Apple"),
            Self::Qualcomm => write!(f, "Qualcomm"),
            Self::Other => write!(f, "Other"),
        }
    }
}

impl std::fmt::Display for NpuType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CoreMl => write!(f, "CoreML"),
            Self::Nnapi => write!(f, "NNAPI"),
            Self::QnnHtp => write!(f, "QNN HTP"),
        }
    }
}

impl std::fmt::Display for CpuFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Avx2 => write!(f, "AVX2"),
            Self::Avx512 => write!(f, "AVX-512"),
            Self::Neon => write!(f, "NEON"),
            Self::Sve => write!(f, "SVE"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_display() {
        assert_eq!(Platform::Linux.to_string(), "Linux");
        assert_eq!(Platform::MacOs.to_string(), "macOS");
        assert_eq!(Platform::Android.to_string(), "Android");
        assert_eq!(Platform::Ios.to_string(), "iOS");
        assert_eq!(Platform::Windows.to_string(), "Windows");
    }

    #[test]
    fn gpu_vendor_display() {
        assert_eq!(GpuVendor::Nvidia.to_string(), "NVIDIA");
        assert_eq!(GpuVendor::Amd.to_string(), "AMD");
        assert_eq!(GpuVendor::Intel.to_string(), "Intel");
        assert_eq!(GpuVendor::Apple.to_string(), "Apple");
        assert_eq!(GpuVendor::Qualcomm.to_string(), "Qualcomm");
        assert_eq!(GpuVendor::Other.to_string(), "Other");
    }

    #[test]
    fn npu_type_display() {
        assert_eq!(NpuType::CoreMl.to_string(), "CoreML");
        assert_eq!(NpuType::Nnapi.to_string(), "NNAPI");
        assert_eq!(NpuType::QnnHtp.to_string(), "QNN HTP");
    }

    #[test]
    fn cpu_feature_display() {
        assert_eq!(CpuFeature::Avx2.to_string(), "AVX2");
        assert_eq!(CpuFeature::Avx512.to_string(), "AVX-512");
        assert_eq!(CpuFeature::Neon.to_string(), "NEON");
        assert_eq!(CpuFeature::Sve.to_string(), "SVE");
    }

    #[test]
    fn hardware_info_construction() {
        let hw = HardwareInfo {
            cpu: CpuInfo {
                arch: "x86_64".into(),
                features: vec![CpuFeature::Avx2],
                cores: 16,
            },
            gpu: Some(GpuInfo {
                vendor: GpuVendor::Nvidia,
                name: "RTX 4090".into(),
                vram_mb: 24000,
                driver: "555.42".into(),
            }),
            npu: None,
            os: OsInfo {
                platform: Platform::Linux,
                version: "Ubuntu 24.04".into(),
            },
            available_ram_mb: 64000,
        };
        assert_eq!(hw.cpu.cores, 16);
        assert_eq!(hw.gpu.as_ref().unwrap().vram_mb, 24000);
        assert!(hw.npu.is_none());
    }

    #[test]
    fn enums_are_copy() {
        let p = Platform::Linux;
        let p2 = p; // Copy
        assert_eq!(p, p2);

        let v = GpuVendor::Nvidia;
        let v2 = v;
        assert_eq!(v, v2);
    }

    #[test]
    fn enums_debug_impl() {
        assert_eq!(format!("{:?}", Platform::Linux), "Linux");
        assert_eq!(format!("{:?}", GpuVendor::Nvidia), "Nvidia");
        assert_eq!(format!("{:?}", NpuType::QnnHtp), "QnnHtp");
        assert_eq!(format!("{:?}", CpuFeature::Avx2), "Avx2");
    }
}
