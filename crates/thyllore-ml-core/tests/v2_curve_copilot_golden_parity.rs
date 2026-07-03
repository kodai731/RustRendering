//! Golden-fixture verification of the deployed v2 curve_copilot ONNX against the
//! fixture exported by the training repo (`*_v2_k48opt_golden.json`).
//!
//! Two checks share the same fixture and inference path (`run_raw` fed the exact
//! preprocessed input tensors, f32 stored as u32 bits):
//! - `..._matches_golden_fixture_bits`: strict bit-exact parity of `mean_curve`.
//! - `..._match_rate_meets_threshold`: tolerance-based agreement rate over the
//!   forecast frames, a portable regression guard that survives ORT / platform
//!   f32 jitter that would break bit-exact parity.
//!
//! Both require THYLLORE_SHARED_DATA_DIR pointing to a SharedData directory that
//! contains both the golden JSON and the ONNX it names.

use std::path::{Path, PathBuf};

use serde_json::Value;
use thyllore_ml_core::copilot::v2::inference::V2CurveCopilotSession;
use thyllore_ml_core::model_path::{EXPORTS_SUBDIR, SHARED_DATA_ENV_VAR};

const GOLDEN_FILENAME: &str = "curve_copilot_20260630_v2_k48opt_golden.json";
const MATCH_TOLERANCE: f32 = 1e-3;
const MIN_MATCH_RATE: f32 = 0.99;

fn resolve_exports_dir() -> Option<PathBuf> {
    let shared_data_dir = std::env::var(SHARED_DATA_ENV_VAR).ok()?;
    let exports_dir = Path::new(&shared_data_dir).join(EXPORTS_SUBDIR);
    exports_dir.exists().then_some(exports_dir)
}

fn bits_to_f32(values: &Value, field: &str, label: &str) -> Vec<f32> {
    values[field]
        .as_array()
        .unwrap_or_else(|| panic!("case {label}: {field} must be an array"))
        .iter()
        .map(|v| {
            let bits = v
                .as_u64()
                .unwrap_or_else(|| panic!("case {label}: {field} must hold u32 bit values"));
            f32::from_bits(bits as u32)
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn load_golden_session_and_cases() -> Option<(V2CurveCopilotSession, Vec<Value>)> {
    let Some(exports_dir) = resolve_exports_dir() else {
        eprintln!(
            "Skipping: set {SHARED_DATA_ENV_VAR} to a SharedData dir containing \
             exports/{GOLDEN_FILENAME} and the ONNX it references."
        );
        return None;
    };

    let golden_path = exports_dir.join(GOLDEN_FILENAME);
    let golden_text = std::fs::read_to_string(&golden_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", golden_path.display()));
    let golden: Value = serde_json::from_str(&golden_text).expect("parse golden json");

    let onnx_filename = golden["onnx"].as_str().expect("golden must name its onnx");
    let model_path = exports_dir.join(onnx_filename);
    let session = V2CurveCopilotSession::from_onnx_path(&model_path.to_string_lossy())
        .unwrap_or_else(|e| panic!("load {}: {e}", model_path.display()));

    let cases = golden["cases"]
        .as_array()
        .expect("golden must have cases")
        .clone();
    assert!(!cases.is_empty(), "golden fixture has no cases");

    Some((session, cases))
}

fn run_case(session: &mut V2CurveCopilotSession, case: &Value) -> (String, Vec<f32>, Vec<f32>) {
    let label = case["label"].as_str().unwrap_or("<unlabeled>").to_string();
    let inputs = &case["inputs_bits"];
    let context = bits_to_f32(inputs, "context", &label);
    let context_times = bits_to_f32(inputs, "context_times", &label);
    let future_times = bits_to_f32(inputs, "future_times", &label);
    let context_tangent = bits_to_f32(inputs, "context_tangent", &label);
    let expected = bits_to_f32(case, "mean_curve_bits", &label);

    let actual = session
        .run_raw(&context, &context_times, &future_times, &context_tangent)
        .unwrap_or_else(|e| panic!("case {label}: inference failed: {e}"));

    assert_eq!(
        actual.len(),
        expected.len(),
        "case {label}: mean_curve length mismatch"
    );

    (label, actual, expected)
}

#[test]
#[ignore]
fn v2_curve_copilot_matches_golden_fixture_bits() {
    let Some((mut session, cases)) = load_golden_session_and_cases() else {
        return;
    };

    for case in &cases {
        let (label, actual, expected) = run_case(&mut session, case);

        let bit_equal = actual
            .iter()
            .zip(&expected)
            .all(|(a, e)| a.to_bits() == e.to_bits());
        assert!(
            bit_equal,
            "case {label}: mean_curve bits differ from golden (max |diff| = {:e})",
            max_abs_diff(&actual, &expected)
        );

        eprintln!("case {label}: bit-exact parity ({} values)", actual.len());
    }
}

#[test]
#[ignore]
fn v2_curve_copilot_match_rate_meets_threshold() {
    let Some((mut session, cases)) = load_golden_session_and_cases() else {
        return;
    };

    let mut total_matched = 0usize;
    let mut total_count = 0usize;

    for case in &cases {
        let (label, actual, expected) = run_case(&mut session, case);

        let matched = actual
            .iter()
            .zip(&expected)
            .filter(|(a, e)| (*a - *e).abs() <= MATCH_TOLERANCE)
            .count();
        let rate = matched as f32 / actual.len() as f32;
        total_matched += matched;
        total_count += actual.len();

        eprintln!(
            "case {label}: match rate {:.2}% ({matched}/{} within {MATCH_TOLERANCE:e}), \
             max |diff| = {:e}",
            rate * 100.0,
            actual.len(),
            max_abs_diff(&actual, &expected)
        );
        assert!(
            rate >= MIN_MATCH_RATE,
            "case {label}: match rate {:.2}% below threshold {:.2}%",
            rate * 100.0,
            MIN_MATCH_RATE * 100.0
        );
    }

    let overall = total_matched as f32 / total_count as f32;
    eprintln!(
        "overall match rate {:.2}% ({total_matched}/{total_count})",
        overall * 100.0
    );
    assert!(
        overall >= MIN_MATCH_RATE,
        "overall match rate {:.2}% below threshold {:.2}%",
        overall * 100.0,
        MIN_MATCH_RATE * 100.0
    );
}
