use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use thyllore_ml_core::copilot::v1::rawfuture::{
    deploy_safe_stats, RawFutureRequest, RawFutureSession,
};

const SHARED_DATA_ENV_VAR: &str = "THYLLORE_SHARED_DATA_DIR";

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/python_parity/fixtures/rawfuture_v1_tangent_fixture.json")
}

fn read_fixture() -> Value {
    let raw = fs::read_to_string(fixture_path()).expect("read rawfuture fixture");
    serde_json::from_str(&raw).expect("parse rawfuture fixture")
}

fn f32_array(value: &Value, key: &str) -> Vec<f32> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("missing array field: {key}"))
        .iter()
        .map(|v| v.as_f64().expect("number") as f32)
        .collect()
}

fn bool_array(value: &Value, key: &str) -> Vec<bool> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("missing array field: {key}"))
        .iter()
        .map(|v| v.as_i64().expect("int mask") != 0)
        .collect()
}

#[test]
fn deploy_safe_stats_match_python() {
    let fixture = read_fixture();
    let context = f32_array(&fixture, "raw_context");
    let future = f32_array(&fixture, "raw_future");
    let reveal = bool_array(&fixture, "reveal_mask");

    let stats = deploy_safe_stats(&context, &future, &reveal);
    let expected_mu = fixture["expected_mu"].as_f64().unwrap() as f32;
    let expected_sd = fixture["expected_sd"].as_f64().unwrap() as f32;

    assert!(
        (stats.mu - expected_mu).abs() < 1e-4,
        "mu {} vs {expected_mu}",
        stats.mu
    );
    assert!(
        (stats.sd - expected_sd).abs() < 1e-4,
        "sd {} vs {expected_sd}",
        stats.sd
    );
}

#[test]
#[ignore]
fn predict_mean_curve_matches_python() {
    let fixture = read_fixture();
    let Some(onnx_path) = resolve_onnx(&fixture) else {
        eprintln!(
            "Skipping: ONNX not found. Set {SHARED_DATA_ENV_VAR} so that \
             $DIR/exports/curve_copilot/<onnx_filename> exists."
        );
        return;
    };

    let context = f32_array(&fixture, "raw_context");
    let future = f32_array(&fixture, "raw_future");
    let reveal = bool_array(&fixture, "reveal_mask");
    let fps = fixture["fps"].as_f64().unwrap() as f32;
    let expected = f32_array(&fixture, "expected_mean_curve");

    let mut session = RawFutureSession::from_onnx_path(&onnx_path).expect("load rawfuture onnx");
    let predicted = session
        .predict_mean_curve(RawFutureRequest {
            context: &context,
            future: &future,
            reveal_mask: &reveal,
            fps,
        })
        .expect("inference");

    assert_eq!(predicted.len(), expected.len());
    let max_abs = predicted
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs < 1e-3, "max|rust-python| = {max_abs}");
    eprintln!("rawfuture parity max|rust-python| = {max_abs:.3e}");
}

fn resolve_onnx(fixture: &Value) -> Option<String> {
    let filename = fixture["onnx_filename"].as_str()?;
    let shared = env::var(SHARED_DATA_ENV_VAR).ok()?;
    let path = Path::new(&shared)
        .join("exports/curve_copilot")
        .join(filename);
    path.exists().then(|| path.to_string_lossy().to_string())
}
