pub fn resolve_curve_copilot_model_path() -> Option<String> {
    if let Some(path) = find_rawfuture_curve_copilot_model_in_shared_data() {
        log!("Using rawfuture curve_copilot model: {}", path);
        return Some(path);
    }

    log_warn!(
        "rawfuture curve_copilot model not found. For development, set {} to your SharedData \
         directory so that {}/{}/{} exists. HuggingFace download from {} is not yet implemented.",
        crate::paths::SHARED_DATA_ENV_VAR,
        crate::paths::EXPORTS_SUBDIR,
        crate::paths::RAWFUTURE_CURVE_COPILOT_SUBDIR,
        crate::paths::RAWFUTURE_CURVE_COPILOT_FILENAME,
        crate::paths::HUGGINGFACE_CURVE_COPILOT_REPO
    );
    None
}

fn find_rawfuture_curve_copilot_model_in_shared_data() -> Option<String> {
    let shared_data_dir = std::env::var(crate::paths::SHARED_DATA_ENV_VAR).ok()?;
    let model_path = std::path::Path::new(&shared_data_dir)
        .join(crate::paths::EXPORTS_SUBDIR)
        .join(crate::paths::RAWFUTURE_CURVE_COPILOT_SUBDIR)
        .join(crate::paths::RAWFUTURE_CURVE_COPILOT_FILENAME);

    model_path
        .exists()
        .then(|| model_path.to_string_lossy().to_string())
}
