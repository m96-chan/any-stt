use crate::hardware::{CpuFeature, CpuInfo, HardwareInfo, NpuInfo, OsInfo, Platform};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::hardware::{GpuInfo, GpuVendor, NpuType};

#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::hardware::{GpuInfo, GpuVendor};

#[cfg(target_os = "android")]
use crate::hardware::NpuType;

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

/// Detect SVE on Linux/Android via getauxval(AT_HWCAP).
#[cfg(target_arch = "aarch64")]
fn detect_sve_linux() -> bool {
    // AT_HWCAP = 16, HWCAP_SVE bit = (1 << 22) on aarch64 Linux.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        const AT_HWCAP: libc::c_ulong = 16;
        const HWCAP_SVE: libc::c_ulong = 1 << 22;
        // SAFETY: getauxval is always safe to call with a valid type.
        let hwcap = unsafe { libc::getauxval(AT_HWCAP) };
        return hwcap & HWCAP_SVE != 0;
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
    // Linux: try parsing nvidia-smi for NVIDIA GPUs.
    #[cfg(target_os = "linux")]
    {
        if let Some(gpu) = detect_nvidia_gpu() {
            return Some(gpu);
        }
    }

    // macOS: Apple GPU is always present on Apple Silicon.
    #[cfg(target_os = "macos")]
    {
        return Some(GpuInfo {
            vendor: GpuVendor::Apple,
            name: "Apple GPU".into(),
            vram_mb: 0, // Unified memory — reported via available_ram_mb instead.
            driver: String::new(),
        });
    }

    #[allow(unreachable_code)]
    None
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
        return Some(NpuInfo {
            npu_type: NpuType::Nnapi,
            available: true,
        });
    }

    #[allow(unreachable_code)]
    None
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

    #[cfg(target_os = "macos")]
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

    #[cfg(target_os = "macos")]
    {
        if let Some(ram) = detect_available_ram_macos() {
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

#[cfg(target_os = "macos")]
fn detect_available_ram_macos() -> Option<u64> {
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
