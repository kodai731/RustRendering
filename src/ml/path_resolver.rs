pub fn resolve_curve_copilot_model_path() -> Option<String> {
    if let Some(path) = find_latest_curve_copilot_model_in_shared_data() {
        log!("Using curve_copilot model: {}", path);
        return Some(path);
    }

    log_warn!(
        "curve_copilot model not found. For development, set {} to your SharedData directory. \
         HuggingFace download from {} is not yet implemented.",
        crate::paths::SHARED_DATA_ENV_VAR,
        crate::paths::HUGGINGFACE_CURVE_COPILOT_REPO
    );
    None
}

fn find_latest_curve_copilot_model_in_shared_data() -> Option<String> {
    let shared_data_dir = std::env::var(crate::paths::SHARED_DATA_ENV_VAR).ok()?;
    let exports_dir = std::path::Path::new(&shared_data_dir).join(crate::paths::EXPORTS_SUBDIR);
    let entries = std::fs::read_dir(&exports_dir).ok()?;

    entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            let matches_prefix = name.starts_with(crate::paths::CURVE_COPILOT_MODEL_PREFIX);
            let matches_suffix = name.ends_with(crate::paths::CURVE_COPILOT_MODEL_SUFFIX);
            if matches_prefix && matches_suffix {
                Some(entry.path().to_string_lossy().to_string())
            } else {
                None
            }
        })
        .max()
}
