pub fn resolve_curve_copilot_model_path() -> String {
    let local_path = std::path::Path::new(crate::paths::CURVE_COPILOT_LOCAL_MODEL);
    if local_path.exists() {
        log!(
            "Using local model: {}",
            crate::paths::CURVE_COPILOT_LOCAL_MODEL
        );
        return crate::paths::CURVE_COPILOT_LOCAL_MODEL.to_string();
    }

    let shared_path = std::path::Path::new(crate::paths::CURVE_COPILOT_SHARED_MODEL);
    if shared_path.exists() {
        log!(
            "Using SharedData model: {}",
            crate::paths::CURVE_COPILOT_SHARED_MODEL
        );
        return crate::paths::CURVE_COPILOT_SHARED_MODEL.to_string();
    }

    if let Some(latest) = find_latest_curve_copilot_model() {
        log!("Using dated SharedData model: {}", latest);
        return latest;
    }

    log!("No trained model found, falling back to dummy model");
    crate::paths::CURVE_COPILOT_DUMMY_MODEL.to_string()
}

fn find_latest_curve_copilot_model() -> Option<String> {
    let exports_dir = std::path::Path::new(crate::paths::SHARED_EXPORTS_DIR);
    let entries = std::fs::read_dir(exports_dir).ok()?;

    entries
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("curve_copilot_") && name.ends_with(".onnx") {
                Some(entry.path().to_string_lossy().to_string())
            } else {
                None
            }
        })
        .max()
}
