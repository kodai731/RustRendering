use std::env;
use std::path::{Path, PathBuf};

use thyllore_ml_api::{
    CopilotRequest, CopilotRequestWire, CopilotResponseWire, MlOps, OP_RUN_CURVE_COPILOT,
};
use thyllore_ml_core::MlCoreImpl;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn resolve_fixture_root() -> PathBuf {
    if let Ok(p) = env::var("THYLLORE_PARITY_FIXTURE_OUTPUT") {
        return PathBuf::from(p);
    }
    workspace_root().join("fixtures").join("ml_parity")
}

fn canonical_onnx_path(fixture_root: &std::path::Path) -> String {
    fixture_root
        .join("onnx")
        .join("curve_copilot.onnx")
        .to_string_lossy()
        .replace('\\', "/")
}

fn build_synthetic_request(model_path: String, seed: u32, frequency: f32) -> CopilotRequest {
    let mut state = seed;
    let next = |state: &mut u32| {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((*state & 0x00FF_FFFF) as f32 / ((1u32 << 24) as f32)) * 2.0 - 1.0
    };

    let context: Vec<f32> = (0..32 * 6).map(|_| next(&mut state)).collect();
    let topology_features: Vec<f32> = (0..6).map(|_| (next(&mut state) + 1.0) * 0.5).collect();
    let bone_name_tokens: Vec<i64> = (0..32).map(|i| (i as i64) % 31 + 1).collect();
    let query_times: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.4).collect();
    let curve_window: Vec<f32> = (0..64)
        .map(|i| {
            let t = i as f32 / 63.0;
            (t * std::f32::consts::TAU * frequency).sin() * 0.5
        })
        .collect();

    let bone_context_keyframes: Vec<f32> = (0..32 * 32 * 6).map(|_| next(&mut state)).collect();
    let bone_context_topology: Vec<f32> = (0..32 * 6)
        .map(|_| (next(&mut state) + 1.0) * 0.5)
        .collect();
    let bone_context_rest_positions: Vec<f32> = (0..32 * 3).map(|_| next(&mut state)).collect();
    let bone_context_mask: Vec<bool> = (0..32).map(|i| i < 16).collect();

    CopilotRequest {
        model_path,
        property_type: 0,
        context,
        topology_features,
        bone_name_tokens,
        query_times,
        curve_window,
        bone_context_keyframes,
        bone_context_topology,
        bone_context_rest_positions,
        bone_context_mask,
    }
}

#[test]
#[ignore]
fn typed_and_call_op_paths_are_bit_identical() {
    let fixture_root = resolve_fixture_root();
    let onnx_path = canonical_onnx_path(&fixture_root);
    if !std::path::Path::new(&onnx_path).exists() {
        eprintln!("skip: onnx not found at {onnx_path}");
        return;
    }

    let core = MlCoreImpl::new();

    for (label, seed, frequency) in [
        ("short", 0xA1A1_A1A1u32, 0.5_f32),
        ("medium", 0xB2B2_B2B2, 1.5),
        ("long", 0xC3C3_C3C3, 3.0),
    ] {
        let request = build_synthetic_request(onnx_path.clone(), seed, frequency);

        let typed_response = core
            .run_curve_copilot(request.clone())
            .unwrap_or_else(|e| panic!("typed call failed for {label}: {e:?}"));

        let wire_request = CopilotRequestWire::from_typed(&request);
        let payload = serde_json::to_vec(&wire_request).expect("encode wire request");
        let response_bytes = core
            .call_op(OP_RUN_CURVE_COPILOT, &payload)
            .unwrap_or_else(|e| panic!("call_op failed for {label}: {e:?}"));
        let wire_response: CopilotResponseWire =
            serde_json::from_slice(&response_bytes).expect("decode wire response");
        let call_op_response = wire_response.into_typed();

        assert_eq!(
            typed_response.predictions.len(),
            call_op_response.predictions.len(),
            "[{label}] prediction count differs"
        );
        for (i, (typed, via_call_op)) in typed_response
            .predictions
            .iter()
            .zip(call_op_response.predictions.iter())
            .enumerate()
        {
            assert_eq!(
                typed.value.to_bits(),
                via_call_op.value.to_bits(),
                "[{label}] prediction[{i}].value bits differ"
            );
            assert_eq!(
                typed.tangent_in[0].to_bits(),
                via_call_op.tangent_in[0].to_bits(),
                "[{label}] prediction[{i}].tangent_in[0] bits differ"
            );
            assert_eq!(
                typed.tangent_in[1].to_bits(),
                via_call_op.tangent_in[1].to_bits(),
                "[{label}] prediction[{i}].tangent_in[1] bits differ"
            );
            assert_eq!(
                typed.tangent_out[0].to_bits(),
                via_call_op.tangent_out[0].to_bits(),
                "[{label}] prediction[{i}].tangent_out[0] bits differ"
            );
            assert_eq!(
                typed.tangent_out[1].to_bits(),
                via_call_op.tangent_out[1].to_bits(),
                "[{label}] prediction[{i}].tangent_out[1] bits differ"
            );
            assert_eq!(
                typed.confidence.to_bits(),
                via_call_op.confidence.to_bits(),
                "[{label}] prediction[{i}].confidence bits differ"
            );
        }
    }
}
