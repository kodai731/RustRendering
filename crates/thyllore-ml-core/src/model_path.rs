//! Single source of truth for locating the rawfuture curve-copilot model.
//!
//! Both the engine (`src/ml/path_resolver.rs`) and the Blender addon (via the
//! PyO3 wheel) resolve the model through this module, so they always reference
//! the same file in the SharedData export tree.

use std::path::Path;

pub const SHARED_DATA_ENV_VAR: &str = "THYLLORE_SHARED_DATA_DIR";
pub const EXPORTS_SUBDIR: &str = "exports";
pub const RAWFUTURE_CURVE_COPILOT_SUBDIR: &str = "curve_copilot";
pub const RAWFUTURE_CURVE_COPILOT_FILENAME: &str = "rawfuture_v1_tangent.onnx";
pub const HUGGINGFACE_CURVE_COPILOT_REPO: &str = "kodai731/thyllore-curve-copilot";

/// Resolve `$THYLLORE_SHARED_DATA_DIR/exports/curve_copilot/rawfuture_v1_tangent.onnx`.
///
/// Returns `None` when the env var is unset or the file does not exist.
pub fn resolve_rawfuture_curve_copilot_model_path() -> Option<String> {
    let shared_data_dir = std::env::var(SHARED_DATA_ENV_VAR).ok()?;
    let model_path = Path::new(&shared_data_dir)
        .join(EXPORTS_SUBDIR)
        .join(RAWFUTURE_CURVE_COPILOT_SUBDIR)
        .join(RAWFUTURE_CURVE_COPILOT_FILENAME);

    model_path
        .exists()
        .then(|| model_path.to_string_lossy().to_string())
}
