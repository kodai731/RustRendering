use std::path::Path;

use crate::copilot::camera_direction::session::CameraDirectionOnnxPaths;

pub const SHARED_DATA_ENV_VAR: &str = "THYLLORE_SHARED_DATA_DIR";
pub const EXPORTS_SUBDIR: &str = "exports";
pub const V2_CURVE_COPILOT_FILENAME: &str = "curve_copilot_20260630_v2_k48opt.onnx";
pub const HUGGINGFACE_CURVE_COPILOT_REPO: &str = "kodai731/thyllore-curve-copilot";
pub const CAMERA_DIRECTION_EXPORT_DIR: &str = "camera_copilot_20260821";

pub fn resolve_v2_curve_copilot_model_path() -> Option<String> {
    let shared_data_dir = std::env::var(SHARED_DATA_ENV_VAR).ok()?;
    let model_path = Path::new(&shared_data_dir)
        .join(EXPORTS_SUBDIR)
        .join(V2_CURVE_COPILOT_FILENAME);

    model_path
        .exists()
        .then(|| model_path.to_string_lossy().to_string())
}

pub fn resolve_camera_direction_model_paths() -> Option<CameraDirectionOnnxPaths> {
    let shared_data_dir = std::env::var(SHARED_DATA_ENV_VAR).ok()?;
    let base = Path::new(&shared_data_dir)
        .join(EXPORTS_SUBDIR)
        .join(CAMERA_DIRECTION_EXPORT_DIR);

    let decoder_step0 = base.join("decoder_step0.onnx");
    let decoder_with_past = base.join("decoder_with_past.onnx");
    let embd_table = base.join("embd_table.bin");
    let text_encoder = base.join("text_encoder.onnx");
    let tokenizer = base.join("tokenizer.json");

    if decoder_step0.exists()
        && decoder_with_past.exists()
        && embd_table.exists()
        && text_encoder.exists()
        && tokenizer.exists()
    {
        Some(CameraDirectionOnnxPaths {
            decoder_step0,
            decoder_with_past,
            embd_table,
            text_encoder,
            tokenizer,
        })
    } else {
        None
    }
}
