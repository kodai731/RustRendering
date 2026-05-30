use std::env;
use std::path::Path;

use thyllore_ml_core::copilot::rawfuture::{
    RawFutureRequest, RawFutureSession, CONTEXT_LENGTH, MAX_HORIZON,
};
use thyllore_ml_core::model_path::resolve_rawfuture_curve_copilot_model_path;

const MODEL_PATH_ENV_VAR: &str = "THYLLORE_CURVE_MODEL";
const DEPLOY_FPS: f32 = 60.0;

fn resolve_model_path() -> Option<String> {
    if let Ok(explicit) = env::var(MODEL_PATH_ENV_VAR) {
        if Path::new(&explicit).exists() {
            return Some(explicit);
        }
    }
    resolve_rawfuture_curve_copilot_model_path()
}

fn ramp(len: usize) -> Vec<f32> {
    (0..len).map(|i| i as f32 * 0.01).collect()
}

#[test]
#[ignore]
fn rawfuture_inference_returns_finite_curve() {
    let Some(model_path) = resolve_model_path() else {
        eprintln!(
            "Skipping: rawfuture model not found. Set {MODEL_PATH_ENV_VAR} to an .onnx file, \
             or THYLLORE_SHARED_DATA_DIR to a SharedData dir containing the exported model. \
             The dummy model on HuggingFace is sufficient -- this checks only that inference \
             runs and returns finite values, not numeric parity with the production model."
        );
        return;
    };

    let context = ramp(CONTEXT_LENGTH);
    let future = ramp(MAX_HORIZON);
    let reveal_mask: Vec<bool> = (0..MAX_HORIZON).map(|i| i % 4 == 0).collect();

    let mut session = RawFutureSession::from_onnx_path(&model_path).expect("load rawfuture onnx");
    let predicted = session
        .predict_mean_curve(RawFutureRequest {
            context: &context,
            future: &future,
            reveal_mask: &reveal_mask,
            fps: DEPLOY_FPS,
        })
        .expect("rawfuture inference");

    assert_eq!(
        predicted.len(),
        MAX_HORIZON,
        "expected {MAX_HORIZON} mean-curve values, got {}",
        predicted.len()
    );
    assert!(
        predicted.iter().all(|value| value.is_finite()),
        "inference produced non-finite values: {predicted:?}"
    );

    eprintln!(
        "rawfuture smoke: model={model_path} produced {} finite mean-curve values",
        predicted.len()
    );
}
