use crate::hardware::{CpuFeature, CpuInfo, HardwareInfo, NpuInfo, OsInfo, Platform};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::hardware::{GpuInfo, GpuVendor, NpuType};

#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::hardware::GpuInfo;

#[cfg(target_os = "linux")]
use crate::hardware::GpuVendor;

#[cfg(target_os = "android")]
use crate::hardware::{GpuVendor, NpuType};

/// Detect hardware capabilities of the current system.
pub fn detect_hardware() -> HardwareInfo {
    HardwareInfo {
        cpu: detect_cpu(),
        gpu: detect_gpu(),
        npu: detect_npu(),
        os: detect_os(),
        available_ram_mb: detect_available_ram_mb(),
    }
}

fn detect_cpu() -> CpuInfo {
    CpuInfo {
        arch: std::env::consts::ARCH.into(),
        features: detect_cpu_features(),
        cores: detect_cpu_cores(),
    }
}

fn detect_cpu_features() -> Vec<CpuFeature> {
    let mut features = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            features.push(CpuFeature::Avx2);
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            features.push(CpuFeature::Avx512);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64.
        features.push(CpuFeature::Neon);

        // SVE detection: check via reading the OS-exposed feature bits.
        // On Linux/Android we can read /proc/cpuinfo or use libc's getauxval.
        if cfg!(target_os = "linux") || cfg!(target_os = "android") {
            if detect_sve_linux() {
                features.push(CpuFeature::Sve);
            }
        }
    }

    features
}

/// Detect SVE on Linux/Android via /proc/cpuinfo.
///
/// We avoid `getauxval(AT_HWCAP)` because on Android the NDK's static libc
/// implementation of `getauxval` can crash during early process init
/// (null deref in `__libc_shared_globals` on Android 15+).
#[cfg(target_arch = "aarch64")]
fn detect_sve_linux() -> bool {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            // Look for "sve" in the Features line
            for line in cpuinfo.lines() {
                if let Some(features) = line.strip_prefix("Features") {
                    if let Some(features) = features.strip_prefix(|c: char| c == ':' || c == '\t' || c == ' ') {
                        return features.split_whitespace().any(|f| f == "sve");
                    }
                }
            }
        }
        false
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        false
    }
}

fn detect_cpu_cores() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

fn detect_gpu() -> Option<GpuInfo> {
    // Linux: enumerate GPUs via /sys/class/drm, then probe details.
    #[cfg(target_os = "linux")]
    {
        // First try nvidia-smi for NVIDIA (gives VRAM + driver info)
        if let Some(gpu) = detect_nvidia_gpu() {
            return Some(gpu);
        }
        // Fall back to sysfs for AMD/Intel/other
        if let Some(gpu) = detect_gpu_sysfs() {
            return Some(gpu);
        }
    }

    // Android: GPU detection via Vulkan probing (done at runtime)
    #[cfg(target_os = "android")]
    {}

    // Apple (macOS/iOS): Apple GPU is always present on Apple Silicon.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        return Some(GpuInfo {
            vendor: GpuVendor::Apple,
            name: "Apple GPU".into(),
            vram_mb: 0,
            driver: String::new(),
        });
    }

    #[allow(unreachable_code)]
    None
}

/// Detect GPU via /sys/class/drm PCI vendor IDs.
/// Works for AMD, Intel, and NVIDIA (as fallback if nvidia-smi is missing).
#[cfg(target_os = "linux")]
fn detect_gpu_sysfs() -> Option<GpuInfo> {
    // PCI vendor IDs
    const VENDOR_NVIDIA: &str = "0x10de";
    const VENDOR_AMD: &str = "0x1002";
    const VENDOR_INTEL: &str = "0x8086";

    let drm_dir = std::path::Path::new("/sys/class/drm");
    if !drm_dir.exists() {
        return None;
    }

    let entries = std::fs::read_dir(drm_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Only look at cardN (not cardN-DP-1 etc.)
        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }

        let vendor_path = entry.path().join("device/vendor");
        let vendor = std::fs::read_to_string(&vendor_path)
            .ok()
            .map(|s| s.trim().to_lowercase());

        let driver_path = entry.path().join("device/uevent");
        let driver = std::fs::read_to_string(&driver_path)
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("DRIVER="))
                    .map(|l| l.trim_start_matches("DRIVER=").to_string())
            })
            .unwrap_or_default();

        if let Some(ref v) = vendor {
            let (gpu_vendor, gpu_name) = match v.as_str() {
                VENDOR_AMD => (GpuVendor::Amd, format!("AMD GPU ({})", driver)),
                VENDOR_INTEL => (GpuVendor::Intel, format!("Intel GPU ({})", driver)),
                VENDOR_NVIDIA => (GpuVendor::Nvidia, format!("NVIDIA GPU ({})", driver)),
                _ => continue,
            };

            // Try to get VRAM from /sys (AMD exposes this)
            let vram_mb = entry
                .path()
                .join("device/mem_info_vram_total")
                .pipe(|p| std::fs::read_to_string(p).ok())
                .and_then(|s| s.trim().parse::<u64>().ok())
                .map(|b| b / (1024 * 1024))
                .unwrap_or(0);

            return Some(GpuInfo {
                vendor: gpu_vendor,
                name: gpu_name,
                vram_mb,
                driver,
            });
        }
    }

    None
}

/// Helper: allow chaining on Path (for VRAM read)
#[cfg(target_os = "linux")]
trait PathPipe {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
        Self: Sized;
}

#[cfg(target_os = "linux")]
impl PathPipe for std::path::PathBuf {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        f(self)
    }
}

/// Try to detect NVIDIA GPU by running nvidia-smi.
#[cfg(target_os = "linux")]
fn detect_nvidia_gpu() -> Option<GpuInfo> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?;
    let parts: Vec<&str> = line.split(", ").collect();
    if parts.len() < 3 {
        return None;
    }

    Some(GpuInfo {
        vendor: GpuVendor::Nvidia,
        name: parts[0].trim().to_string(),
        vram_mb: parts[1].trim().parse().unwrap_or(0),
        driver: parts[2].trim().to_string(),
    })
}

fn detect_npu() -> Option<NpuInfo> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        return Some(NpuInfo {
            npu_type: NpuType::CoreMl,
            available: true,
        });
    }

    #[cfg(target_os = "android")]
    {
        // Prefer QNN HTP on Snapdragon devices (check for Hexagon DSP).
        if detect_qnn_htp_available() {
            return Some(NpuInfo {
                npu_type: NpuType::QnnHtp,
                available: true,
            });
        }
        // Fallback to NNAPI.
        return Some(NpuInfo {
            npu_type: NpuType::Nnapi,
            available: true,
        });
    }

    #[allow(unreachable_code)]
    None
}

/// Check if QNN HTP (Hexagon) is available.
///
/// On Android, checks for the presence of libQnnHtp.so in known paths.
/// On other platforms, returns false (QNN is Android/Snapdragon-only).
#[cfg(target_os = "android")]
fn detect_qnn_htp_available() -> bool {
    use std::path::Path;
    // Common locations for QNN libraries on Android
    let candidates = [
        "/vendor/lib64/libQnnHtp.so",
        "/system/lib64/libQnnHtp.so",
        "/system/vendor/lib64/libQnnHtp.so",
    ];
    candidates.iter().any(|p| Path::new(p).exists())
}

fn detect_os() -> OsInfo {
    let platform = if cfg!(target_os = "macos") {
        Platform::MacOs
    } else if cfg!(target_os = "ios") {
        Platform::Ios
    } else if cfg!(target_os = "android") {
        Platform::Android
    } else {
        Platform::Linux
    };

    let version = detect_os_version();

    OsInfo { platform, version }
}

fn detect_os_version() -> String {
    #[cfg(target_os = "linux")]
    {
        // Try /etc/os-release first.
        if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
            for line in content.lines() {
                if let Some(v) = line.strip_prefix("PRETTY_NAME=") {
                    return v.trim_matches('"').to_string();
                }
            }
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        if let Ok(output) = std::process::Command::new("sw_vers")
            .arg("-productVersion")
            .output()
        {
            if output.status.success() {
                return String::from_utf8_lossy(&output.stdout).trim().to_string();
            }
        }
    }

    String::new()
}

fn detect_available_ram_mb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Some(ram) = detect_available_ram_linux() {
            return ram;
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        if let Some(ram) = detect_available_ram_apple() {
            return ram;
        }
    }

    0
}

#[cfg(target_os = "linux")]
fn detect_available_ram_linux() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

/// Detect available RAM on Apple platforms (macOS/iOS) via sysctl.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn detect_available_ram_apple() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("hw.memsize")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let bytes: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    Some(bytes / (1024 * 1024))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_hardware_returns_valid_info() {
        let hw = detect_hardware();
        assert!(!hw.cpu.arch.is_empty());
        assert!(hw.cpu.cores > 0);
    }

    #[test]
    fn cpu_features_not_empty_on_known_arch() {
        let features = detect_cpu_features();
        // On x86_64 CI, at least AVX2 is expected.
        // On aarch64, NEON is mandatory.
        if cfg!(target_arch = "x86_64") {
            // AVX2 is present on virtually all x86_64 CPUs since ~2013.
            assert!(features.contains(&CpuFeature::Avx2));
        }
        if cfg!(target_arch = "aarch64") {
            assert!(features.contains(&CpuFeature::Neon));
        }
    }

    #[test]
    fn detect_os_returns_platform() {
        let os = detect_os();
        if cfg!(target_os = "linux") {
            assert_eq!(os.platform, Platform::Linux);
        }
        if cfg!(target_os = "macos") {
            assert_eq!(os.platform, Platform::MacOs);
        }
    }
}
