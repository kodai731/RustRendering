use std::env;
use std::fs;
use std::path::PathBuf;

use thyllore_ml_api::{CopilotRequest, CopilotResponse, MlOps};
use thyllore_ml_core::MlCoreImpl;

const CONTEXT_LEN: usize = 8 * 6;
const TOPOLOGY_LEN: usize = 6;
const BONE_NAME_TOKENS: usize = 32;
const QUERY_TIMES: usize = 4;
const CURVE_WINDOW_LEN: usize = 64;

fn fixture_root_from_env() -> PathBuf {
    PathBuf::from(
        env::var("THYLLORE_PHASE5_FIXTURE_OUTPUT")
            .expect("THYLLORE_PHASE5_FIXTURE_OUTPUT must be set"),
    )
}

fn canonical_onnx_path() -> String {
    fixture_root_from_env()
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
    let numpy_dir = fixture_root_from_env().join("numpy");
    fs::create_dir_all(&numpy_dir).expect("create numpy fixture dir");

    let core = MlCoreImpl::new();
    let model_path = canonical_onnx_path();

    for case in [Case::Short, Case::Medium, Case::Long] {
        let request = build_request(case, &model_path);
        write_input_json(&numpy_dir, case, &request);

        let response = core
            .run_curve_copilot(request)
            .unwrap_or_else(|e| panic!("run_curve_copilot failed for {case:?}: {e:?}"));
        write_golden_json(&numpy_dir, case, &response);

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
            Case::Short => "short",
            Case::Medium => "medium",
            Case::Long => "long",
        }
    }

    fn property_type_id(self) -> u32 {
        match self {
            Case::Short => 0,
            Case::Medium => 3,
            Case::Long => 6,
        }
    }

    fn curve_frequency_hz(self) -> f32 {
        match self {
            Case::Short => 0.5,
            Case::Medium => 1.5,
            Case::Long => 3.0,
        }
    }

    fn rng_seed(self) -> u32 {
        match self {
            Case::Short => 0xA1A1_A1A1,
            Case::Medium => 0xB2B2_B2B2,
            Case::Long => 0xC3C3_C3C3,
        }
    }
}

fn build_request(case: Case, model_path: &str) -> CopilotRequest {
    let mut rng = LinearRng::new(case.rng_seed());

    let context: Vec<f32> = (0..CONTEXT_LEN).map(|_| rng.next_f32_signed()).collect();
    let topology_features: Vec<f32> = (0..TOPOLOGY_LEN).map(|_| rng.next_f32()).collect();
    let bone_name_tokens: Vec<i64> = (0..BONE_NAME_TOKENS)
        .map(|i| (i as i64) % 31 + 1)
        .collect();

    let query_times: Vec<f32> = (0..QUERY_TIMES)
        .map(|i| (i as f32 + 1.0) * 4.0 / (QUERY_TIMES as f32 + 1.0))
        .collect();

    let frequency = case.curve_frequency_hz();
    let curve_window: Vec<f32> = (0..CURVE_WINDOW_LEN)
        .map(|i| {
            let t = i as f32 / (CURVE_WINDOW_LEN as f32 - 1.0);
            (t * std::f32::consts::TAU * frequency).sin() * 0.5
        })
        .collect();

    CopilotRequest {
        model_path: model_path.to_string(),
        property_type: case.property_type_id(),
        context,
        topology_features,
        bone_name_tokens,
        query_times,
        curve_window,
    }
}

fn write_input_json(numpy_dir: &std::path::Path, case: Case, request: &CopilotRequest) {
    let mut json = String::from("{\n");
    json.push_str(&format!(
        "  \"model_path\": {},\n",
        json_escape_string(&request.model_path)
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

    let path = numpy_dir.join(format!("curve_copilot_input_{}.json", case.label()));
    fs::write(&path, json).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
}

fn write_golden_json(numpy_dir: &std::path::Path, case: Case, response: &CopilotResponse) {
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

    let path = numpy_dir.join(format!("curve_copilot_golden_{}.json", case.label()));
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

fn json_escape_string(s: &str) -> String {
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

struct LinearRng {
    state: u32,
}

impl LinearRng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() & 0x00FF_FFFF) as f32 / ((1u32 << 24) as f32)
    }

    fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}
