pub const CURVE_COPILOT_MODEL_PREFIX: &str = "curve_copilot_";
pub const CURVE_COPILOT_MODEL_SUFFIX: &str = ".onnx";

#[cfg(feature = "ml")]
pub use thyllore_ml_core::model_path::{
    EXPORTS_SUBDIR, HUGGINGFACE_CURVE_COPILOT_REPO, SHARED_DATA_ENV_VAR, V2_CURVE_COPILOT_FILENAME,
};
