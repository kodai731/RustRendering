use std::path::Path;

pub const SHARED_DATA_ENV_VAR: &str = "THYLLORE_SHARED_DATA_DIR";
pub const EXPORTS_SUBDIR: &str = "exports";
pub const V1_CURVE_COPILOT_FILENAME: &str = "curve_copilot_20260531_v1_tangent.onnx";
pub const HUGGINGFACE_CURVE_COPILOT_REPO: &str = "kodai731/thyllore-curve-copilot";

pub fn resolve_v1_curve_copilot_model_path() -> Option<String> {
    let shared_data_dir = std::env::var(SHARED_DATA_ENV_VAR).ok()?;
    let model_path = Path::new(&shared_data_dir)
        .join(EXPORTS_SUBDIR)
        .join(V1_CURVE_COPILOT_FILENAME);

    model_path
        .exists()
        .then(|| model_path.to_string_lossy().to_string())
}
