use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use thyllore_ml_core::copilot::context::flatten_context;
use thyllore_ml_core::copilot::query::generate_query_times;
use thyllore_ml_core::copilot::session::Session;
use thyllore_ml_core::copilot::window::sample_window;
use thyllore_ml_core::{compute_bone_name_tokens, compute_bone_topology};
use thyllore_model_core::Skeleton;

const SHARED_DATA_ENV_VAR: &str = "THYLLORE_SHARED_DATA_DIR";
const EXPORTS_SUBDIR: &str = "exports";
const ONNX_PREFIX: &str = "curve_copilot_";
const ONNX_SUFFIX: &str = ".onnx";

const TARGET_BONE_NAME: &str = "hip";
const PROPERTY_TYPE_ID: u32 = 2;
const MAX_CONTEXT_KEYFRAMES: usize = 8;
const CLIP_DURATION: f32 = 3.3;
const CURRENT_TIME: f32 = 2.5;

#[test]
#[ignore]
fn generate_curve_copilot_inference_fixture() {
    let Some(onnx_path) = find_curve_copilot_onnx() else {
        eprintln!(
            "Skipping: ONNX not found. Set {} and ensure {}/curve_copilot_*.onnx exists.",
            SHARED_DATA_ENV_VAR, EXPORTS_SUBDIR
        );
        return;
    };

    let inputs = build_synthetic_inputs();

    let mut session = Session::from_onnx_path(&onnx_path).expect("load onnx");
    let max_steps = session
        .max_steps()
        .expect("ONNX missing max_steps metadata");
    assert_eq!(
        max_steps,
        inputs.query_times.len(),
        "fixture must use the same max_steps as the model"
    );

    let steps = session
        .run_curve_copilot(
            &inputs.context_flat,
            PROPERTY_TYPE_ID,
            &inputs.topology_features,
            &inputs.bone_name_tokens,
            &inputs.query_times,
            &inputs.curve_window,
        )
        .expect("inference failed");

    let onnx_filename = Path::new(&onnx_path)
        .file_name()
        .expect("onnx filename")
        .to_string_lossy()
        .to_string();

    let fixture_json = serialize_fixture(&onnx_filename, &inputs, &steps);

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/python_parity/fixtures");
    fs::create_dir_all(&out_dir).expect("create fixture dir");
    let out_path = out_dir.join("curve_copilot_inference_fixture.json");
    fs::write(&out_path, fixture_json).expect("write fixture");

    eprintln!("Wrote fixture: {}", out_path.display());
    eprintln!("Rust predictions (first 3):");
    for (i, s) in steps.iter().take(3).enumerate() {
        eprintln!(
            "  step {}: value={:.4} confidence={:.4}",
            i + 1,
            s.value,
            s.confidence
        );
    }
}

struct SyntheticInputs {
    context_flat: Vec<f32>,
    context_mean: f32,
    context_std: f32,
    topology_features: Vec<f32>,
    bone_name_tokens: Vec<i64>,
    query_times: Vec<f32>,
    curve_window: Vec<f32>,
    raw_times: Vec<f32>,
    raw_values: Vec<f32>,
}

fn build_synthetic_inputs() -> SyntheticInputs {
    let times: Vec<f32> = (0..MAX_CONTEXT_KEYFRAMES).map(|i| i as f32 * 0.3).collect();
    let values: Vec<f32> = times
        .iter()
        .map(|&t| (t * std::f32::consts::PI).sin() * 3.0)
        .collect();
    let zero = vec![0.0_f32; times.len()];

    let ctx = flatten_context(
        &times,
        &values,
        &zero,
        &zero,
        &zero,
        &zero,
        MAX_CONTEXT_KEYFRAMES,
        CLIP_DURATION,
    );

    let mut skeleton = Skeleton::new("parity");
    skeleton.add_bone("root", None);
    let hip_id = skeleton.add_bone(TARGET_BONE_NAME, Some(0));
    skeleton.add_bone("spine", Some(hip_id));

    let topology_map = compute_bone_topology(&skeleton);
    let tokens_map = compute_bone_name_tokens(&skeleton);

    let topology_features = topology_map.get(&hip_id).expect("hip topology").to_vec();
    let bone_name_tokens = tokens_map.get(&hip_id).expect("hip tokens").to_vec();

    let max_steps = 8usize;
    let query_times = generate_query_times(&times, CURRENT_TIME, CLIP_DURATION, max_steps);

    let context_start_time = times
        .iter()
        .rev()
        .take(MAX_CONTEXT_KEYFRAMES)
        .last()
        .copied()
        .unwrap_or(0.0);
    let last_query_time = query_times
        .last()
        .map(|t| t * CLIP_DURATION.max(0.001))
        .unwrap_or(CLIP_DURATION);
    let curve_window = sample_window(
        &times,
        &values,
        context_start_time,
        last_query_time,
        ctx.curve_mean,
        ctx.curve_std,
    );

    SyntheticInputs {
        context_flat: ctx.flat,
        context_mean: ctx.curve_mean,
        context_std: ctx.curve_std,
        topology_features,
        bone_name_tokens,
        query_times,
        curve_window: curve_window.to_vec(),
        raw_times: times,
        raw_values: values,
    }
}

fn serialize_fixture(
    onnx_filename: &str,
    inputs: &SyntheticInputs,
    steps: &[thyllore_ml_core::copilot::types::CopilotStepPrediction],
) -> String {
    let mut json = String::from("{\n");

    json.push_str(&format!("  \"onnx_filename\": \"{}\",\n", onnx_filename));
    json.push_str(&format!(
        "  \"clip_duration\": {},\n",
        f32_bits(CLIP_DURATION)
    ));
    json.push_str(&format!(
        "  \"current_time\": {},\n",
        f32_bits(CURRENT_TIME)
    ));
    json.push_str(&format!("  \"property_type_id\": {},\n", PROPERTY_TYPE_ID));

    json.push_str(&format!(
        "  \"raw_times\": [{}],\n",
        bits_csv(&inputs.raw_times)
    ));
    json.push_str(&format!(
        "  \"raw_values\": [{}],\n",
        bits_csv(&inputs.raw_values)
    ));

    json.push_str(&format!(
        "  \"context_flat\": [{}],\n",
        bits_csv(&inputs.context_flat)
    ));
    json.push_str(&format!(
        "  \"context_mean\": {},\n",
        f32_bits(inputs.context_mean)
    ));
    json.push_str(&format!(
        "  \"context_std\": {},\n",
        f32_bits(inputs.context_std)
    ));

    json.push_str(&format!(
        "  \"topology_features\": [{}],\n",
        bits_csv(&inputs.topology_features)
    ));

    let tokens_csv: Vec<String> = inputs
        .bone_name_tokens
        .iter()
        .map(|v| v.to_string())
        .collect();
    json.push_str(&format!(
        "  \"bone_name_tokens\": [{}],\n",
        tokens_csv.join(", ")
    ));

    json.push_str(&format!(
        "  \"query_times\": [{}],\n",
        bits_csv(&inputs.query_times)
    ));
    json.push_str(&format!(
        "  \"curve_window\": [{}],\n",
        bits_csv(&inputs.curve_window)
    ));

    let pred_rows: Vec<String> = steps
        .iter()
        .map(|s| {
            format!(
                "[{}, {}, {}, {}, {}]",
                f32_bits(s.value),
                f32_bits(s.tangent_in.0),
                f32_bits(s.tangent_in.1),
                f32_bits(s.tangent_out.0),
                f32_bits(s.tangent_out.1)
            )
        })
        .collect();
    json.push_str(&format!(
        "  \"rust_predictions\": [{}],\n",
        pred_rows.join(", ")
    ));

    let conf_csv: Vec<String> = steps
        .iter()
        .map(|s| f32_bits(s.confidence).to_string())
        .collect();
    json.push_str(&format!(
        "  \"rust_confidences\": [{}]\n",
        conf_csv.join(", ")
    ));

    json.push_str("}\n");
    json
}

fn f32_bits(v: f32) -> u32 {
    v.to_bits()
}

fn bits_csv(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| f32_bits(*v).to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn find_curve_copilot_onnx() -> Option<String> {
    let shared_dir = env::var(SHARED_DATA_ENV_VAR).ok()?;
    let exports = Path::new(&shared_dir).join(EXPORTS_SUBDIR);
    let entries = fs::read_dir(&exports).ok()?;
    entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(ONNX_PREFIX) && name.ends_with(ONNX_SUFFIX) {
                Some(entry.path().to_string_lossy().to_string())
            } else {
                None
            }
        })
        .max()
}
