use std::path::{Path, PathBuf};

use crate::vulkanr::device::GpuSelector;

const CONFIG_RELATIVE_PATH: &str = ".config/thyllore.toml";
const GPU_DEVICE_KEY: &str = "gpu_device";
const GPU_ENV_VAR: &str = "THYLLORE_GPU";

pub fn load_gpu_selector() -> GpuSelector {
    if let Ok(value) = std::env::var(GPU_ENV_VAR) {
        match value.parse::<GpuSelector>() {
            Ok(selector) => {
                log!("GPU selector from {}: {:?}", GPU_ENV_VAR, selector);
                return selector;
            }
            Err(error) => {
                log_warn!(
                    "Invalid {}='{}': {}; falling back",
                    GPU_ENV_VAR,
                    value,
                    error
                );
            }
        }
    }

    if let Some(path) = find_config_file() {
        match read_gpu_selector(&path) {
            Ok(Some(selector)) => {
                log!("GPU selector from {}: {:?}", path.display(), selector);
                return selector;
            }
            Ok(None) => {}
            Err(error) => log_warn!("Failed to read {}: {}", path.display(), error),
        }
    }

    GpuSelector::Auto
}

fn find_config_file() -> Option<PathBuf> {
    let mut search_roots = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            search_roots.push(dir.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        search_roots.push(cwd);
    }

    for root in search_roots {
        for ancestor in root.ancestors() {
            let candidate = ancestor.join(CONFIG_RELATIVE_PATH);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn read_gpu_selector(path: &Path) -> anyhow::Result<Option<GpuSelector>> {
    let contents = std::fs::read_to_string(path)?;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('[') {
            continue;
        }

        let Some((key, raw_value)) = trimmed.split_once('=') else {
            continue;
        };
        if key.trim() != GPU_DEVICE_KEY {
            continue;
        }

        return Ok(Some(parse_config_value(raw_value)?));
    }

    Ok(None)
}

fn parse_config_value(raw_value: &str) -> anyhow::Result<GpuSelector> {
    let value = raw_value.trim();
    let unquoted = if value.starts_with('"') {
        value
            .strip_prefix('"')
            .and_then(|rest| rest.split('"').next())
            .unwrap_or("")
    } else {
        value.split('#').next().unwrap_or("").trim()
    };

    unquoted.parse::<GpuSelector>()
}
