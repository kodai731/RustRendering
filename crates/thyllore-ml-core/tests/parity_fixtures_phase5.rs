//! Phase 5 — Tier B parity fixture generator.
//!
//! Run from WSL2 (recommended) for native ext4 I/O speed:
//!     export THYLLORE_PHASE5_FIXTURE_OUTPUT=/home/kodai/Projects/SharedData/fixtures/ml_parity
//!     cargo test -p thyllore-ml-core --test parity_fixtures_phase5 \
//!         generate_phase5_curve_copilot_fixtures -- --ignored --nocapture
//!
//! Run from Windows native cargo (fallback, slower):
//!     $env:THYLLORE_PHASE5_FIXTURE_OUTPUT = "//wsl.localhost/Ubuntu/home/kodai/Projects/SharedData/fixtures/ml_parity"
//!     cargo test -p thyllore-ml-core --test parity_fixtures_phase5 `
//!         generate_phase5_curve_copilot_fixtures -- --ignored --nocapture
//!
//! Output schema (one JSON per case, paired input + golden):
//!     numpy/curve_copilot_input_<case>.json
//!         {
//!           "model_path": "...",
//!           "property_type": 0,
//!           "context_bits":          [u32; 48],   // 8 keyframes * 6 features, f32::to_bits()
//!           "topology_features_bits":[u32; 6],
//!           "bone_name_tokens":      [i64; 32],
//!           "query_times_bits":      [u32; Q],
//!           "curve_window_bits":     [u32; 64]
//!         }
//!     numpy/curve_copilot_golden_<case>.json
//!         {
//!           "predictions": [
//!             { "value_bits": u32,
//!               "tangent_in_bits":  [u32; 2],
//!               "tangent_out_bits": [u32; 2],
//!               "confidence_bits":  u32 },
//!             ...
//!           ]
//!         }
//!
//! Bit-identical preservation: all f32 values are serialized as `to_bits() -> u32`
//! to round-trip through JSON without precision loss (Phase 2 established pattern).
//!
//! No `python` feature gate is needed — the generator only uses `ort` (always
//! linked) and writes JSON manually.

use std::env;
use std::fs;
use std::path::PathBuf;

use thyllore_ml_api::{CopilotRequest, CopilotResponse, MlOps};
use thyllore_ml_core::MlCoreImpl;

// All shapes are fixed by the curve_copilot.onnx model.
// query_times in particular is hardcoded to 4 in the trained model
// (`MAX_STEPS = 4` in src/ecs/systems/curve_suggestion_systems.rs);
// fixtures only vary in input values, not dimensions.
const CONTEXT_LEN: usize = 8 * 6;
const TOPOLOGY_LEN: usize = 6;
const BONE_NAME_TOKENS: usize = 32;
const QUERY_TIMES: usize = 4;
const CURVE_WINDOW_LEN: usize = 64;

fn fixture_root() -> PathBuf {
    PathBuf::from(
        env::var("THYLLORE_PHASE5_FIXTURE_OUTPUT").expect(
            "set THYLLORE_PHASE5_FIXTURE_OUTPUT to the fixtures/ml_parity root \
             (e.g. /home/kodai/Projects/SharedData/fixtures/ml_parity)",
        ),
    )
}

fn onnx_model_path() -> String {
    fixture_root()
        .join("onnx")
        .join("curve_copilot.onnx")
        .to_string_lossy()
        .replace('\\', "/")
}

fn f32_bits(v: f32) -> u32 {
    v.to_bits()
}

#[test]
#[ignore]
fn generate_phase5_curve_copilot_fixtures() {
    let out_dir = fixture_root().join("numpy");
    fs::create_dir_all(&out_dir).expect("create numpy fixture dir");

    let core = MlCoreImpl::new();
    let model_path = onnx_model_path();

    for case in [Case::Short, Case::Medium, Case::Long] {
        let request = build_request(case, &model_path);
        write_input_json(&out_dir, case, &request);

        let response = core
            .run_curve_copilot(request)
            .unwrap_or_else(|e| panic!("run_curve_copilot failed for {case:?}: {e:?}"));
        write_golden_json(&out_dir, case, &response);

        eprintln!(
            "wrote fixtures for {case:?}: {} predictions",
            response.predictions.len()
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum Case {
    Short,
    Medium,
    Long,
}

impl Case {
    fn label(self) -> &'static str {
        match self {
            // Cases differ in input value patterns, not in array dimensions
            // (curve_copilot.onnx fixes all shapes including query_times = 4).
            Case::Short => "short",   // smooth low-frequency curve, translation
            Case::Medium => "medium", // mid-frequency, rotation property
            Case::Long => "long",     // high-frequency + larger context, scale
        }
    }

    fn property_type(self) -> u32 {
        // Maps to PropertyType enum order in the proto:
        //   0: TRANSLATION_X, 3: ROTATION_X, 6: SCALE_X (approximate IDs)
        match self {
            Case::Short => 0,
            Case::Medium => 3,
            Case::Long => 6,
        }
    }

    fn frequency_hz(self) -> f32 {
        match self {
            Case::Short => 0.5,
            Case::Medium => 1.5,
            Case::Long => 3.0,
        }
    }

    fn seed(self) -> u32 {
        match self {
            Case::Short => 0xA1A1_A1A1,
            Case::Medium => 0xB2B2_B2B2,
            Case::Long => 0xC3C3_C3C3,
        }
    }
}

fn build_request(case: Case, model_path: &str) -> CopilotRequest {
    let mut rng = LinearRng::new(case.seed());

    let context: Vec<f32> = (0..CONTEXT_LEN).map(|_| rng.next_f32_signed()).collect();
    let topology_features: Vec<f32> = (0..TOPOLOGY_LEN).map(|_| rng.next_f32()).collect();
    let bone_name_tokens: Vec<i64> = (0..BONE_NAME_TOKENS)
        .map(|i| (i as i64) % 31 + 1)
        .collect();

    // Fixed-size query times: evenly spaced over a 4 second clip.
    let query_times: Vec<f32> = (0..QUERY_TIMES)
        .map(|i| (i as f32 + 1.0) * 4.0 / (QUERY_TIMES as f32 + 1.0))
        .collect();

    // Synthetic sin curve at the case's frequency.
    let freq = case.frequency_hz();
    let curve_window: Vec<f32> = (0..CURVE_WINDOW_LEN)
        .map(|i| {
            let t = i as f32 / (CURVE_WINDOW_LEN as f32 - 1.0);
            (t * std::f32::consts::TAU * freq).sin() * 0.5
        })
        .collect();

    CopilotRequest {
        model_path: model_path.to_string(),
        property_type: case.property_type(),
        context,
        topology_features,
        bone_name_tokens,
        query_times,
        curve_window,
    }
}

fn write_input_json(out_dir: &std::path::Path, case: Case, request: &CopilotRequest) {
    let mut json = String::from("{\n");
    json.push_str(&format!(
        "  \"model_path\": {},\n",
        json_string(&request.model_path)
    ));
    json.push_str(&format!(
        "  \"property_type\": {},\n",
        request.property_type
    ));
    json.push_str(&format!(
        "  \"context_bits\": {},\n",
        f32_array_to_bits_json(&request.context)
    ));
    json.push_str(&format!(
        "  \"topology_features_bits\": {},\n",
        f32_array_to_bits_json(&request.topology_features)
    ));
    json.push_str(&format!(
        "  \"bone_name_tokens\": {},\n",
        i64_array_to_json(&request.bone_name_tokens)
    ));
    json.push_str(&format!(
        "  \"query_times_bits\": {},\n",
        f32_array_to_bits_json(&request.query_times)
    ));
    json.push_str(&format!(
        "  \"curve_window_bits\": {}\n",
        f32_array_to_bits_json(&request.curve_window)
    ));
    json.push_str("}\n");

    let path = out_dir.join(format!("curve_copilot_input_{}.json", case.label()));
    fs::write(&path, json).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
}

fn write_golden_json(out_dir: &std::path::Path, case: Case, response: &CopilotResponse) {
    let mut json = String::from("{\n  \"predictions\": [");
    for (i, step) in response.predictions.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str("\n    {");
        json.push_str(&format!(" \"value_bits\": {}", f32_bits(step.value)));
        json.push_str(&format!(
            ", \"tangent_in_bits\": [{}, {}]",
            f32_bits(step.tangent_in[0]),
            f32_bits(step.tangent_in[1])
        ));
        json.push_str(&format!(
            ", \"tangent_out_bits\": [{}, {}]",
            f32_bits(step.tangent_out[0]),
            f32_bits(step.tangent_out[1])
        ));
        json.push_str(&format!(
            ", \"confidence_bits\": {}",
            f32_bits(step.confidence)
        ));
        json.push_str(" }");
    }
    json.push_str("\n  ]\n}\n");

    let path = out_dir.join(format!("curve_copilot_golden_{}.json", case.label()));
    fs::write(&path, json).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
}

fn f32_array_to_bits_json(values: &[f32]) -> String {
    let parts: Vec<String> = values.iter().map(|v| f32_bits(*v).to_string()).collect();
    format!("[{}]", parts.join(", "))
}

fn i64_array_to_json(values: &[i64]) -> String {
    let parts: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Linear congruential generator for deterministic fixture content.
/// Not for cryptographic use; only so all test runs produce identical fixtures.
struct LinearRng {
    state: u32,
}

impl LinearRng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes constants
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        // mantissa-only 24 bit fraction in [0, 1)
        (self.next_u32() & 0x00FF_FFFF) as f32 / ((1u32 << 24) as f32)
    }

    fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}
