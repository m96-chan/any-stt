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
}
