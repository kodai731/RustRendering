pub const CURVE_COPILOT_MODEL_PREFIX: &str = "curve_copilot_";
pub const CURVE_COPILOT_MODEL_SUFFIX: &str = ".onnx";

// The rawfuture curve-copilot model location is the single source of truth in
// thyllore_ml_core::model_path, shared with the Blender addon. Re-exported here
// so engine code keeps using crate::paths::* without duplicating the values.
#[cfg(feature = "ml")]
pub use thyllore_ml_core::model_path::{
    EXPORTS_SUBDIR, HUGGINGFACE_CURVE_COPILOT_REPO, RAWFUTURE_CURVE_COPILOT_FILENAME,
    RAWFUTURE_CURVE_COPILOT_SUBDIR, SHARED_DATA_ENV_VAR,
};
