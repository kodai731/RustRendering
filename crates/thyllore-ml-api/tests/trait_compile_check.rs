//! Compile-time signature lock for the L2 stable contract.
//!
//! This test does not exercise any runtime behavior. Its sole job is to fail
//! to compile when the public surface of `thyllore-ml-api` drifts in a way
//! that should also bump `ABI_MARKER` and require an addon reinstall.
//!
//! When this file no longer compiles, the changer must:
//! 1. Decide whether the change is genuinely breaking (yes for most
//!    rename/remove/type-change operations, no for pure additions with
//!    `#[serde(default)]`).
//! 2. If breaking, bump `ABI_MARKER` in `src/lib.rs` and update the
//!    `addon_expected_abi_marker` constant inside this test in lockstep.
//! 3. Update `blender_addon/__init__.py::EXPECTED_ABI_MARKER` to the same
//!    value so the addon and wheel agree.
//! 4. Update the assertions in this file to reflect the new signatures.

use thyllore_ml_api::{
    CopilotRequest, CopilotResponse, CopilotStepPrediction, CurvePredictRequest,
    CurvePredictResponse, MlError, MlOps, SkeletonSnapshot, TopologyResult, ABI_MARKER,
};

#[test]
fn abi_marker_is_locked() {
    let addon_expected_abi_marker: u32 = 3;
    assert_eq!(
        ABI_MARKER, addon_expected_abi_marker,
        "ABI_MARKER changed without updating the matching constant in this test (and \
         blender_addon/__init__.py::EXPECTED_ABI_MARKER must move in lockstep)."
    );
}

#[test]
fn ml_error_variants_are_addressable() {
    let cases: [MlError; 5] = [
        MlError::invalid("x"),
        MlError::inference("x"),
        MlError::unsupported("x"),
        MlError::schema("op", "msg"),
        MlError::internal("x"),
    ];

    for err in &cases {
        match err {
            MlError::InvalidRequest(_)
            | MlError::InferenceFailed(_)
            | MlError::UnsupportedOp(_)
            | MlError::SchemaMismatch { .. }
            | MlError::Internal(_) => {}
        }
    }
}

#[test]
fn schemas_round_trip_through_serde_json() {
    let skeleton = SkeletonSnapshot {
        bone_names: vec!["root".into(), "hip".into()],
        parent_indices: vec![-1, 0],
        local_matrices: vec![[[0.0; 4]; 4]; 2],
    };
    let req = CopilotRequest {
        model_path: "/tmp/model.onnx".into(),
        property_type: 0,
        context: vec![0.0; 48],
        topology_features: vec![0.0; 6],
        bone_name_tokens: vec![0; 32],
        query_times: vec![1.0, 2.0],
        curve_window: vec![0.0; 64],
        bone_context_keyframes: vec![0.0; 32 * 8 * 6],
        bone_context_topology: vec![0.0; 32 * 6],
        bone_context_rest_positions: vec![0.0; 32 * 3],
        bone_context_mask: vec![false; 32],
    };
    let json = serde_json::to_string(&req).expect("serialize CopilotRequest");
    let _: CopilotRequest = serde_json::from_str(&json).expect("deserialize CopilotRequest");

    let resp = CopilotResponse {
        predictions: vec![CopilotStepPrediction {
            value: 0.0,
            tangent_in: [0.0, 0.0],
            tangent_out: [0.0, 0.0],
            confidence: 0.9,
        }],
    };
    let _ = serde_json::to_string(&resp).expect("serialize CopilotResponse");
    let _ = serde_json::to_string(&CurvePredictRequest {
        model_path: "p".into(),
        flat_input: vec![0.0],
    })
    .expect("serialize CurvePredictRequest");
    let _ = serde_json::to_string(&CurvePredictResponse {
        flat_output: vec![0.0],
    })
    .expect("serialize CurvePredictResponse");
    let _ = serde_json::to_string(&TopologyResult {
        flat_features: vec![0.0],
        features_per_bone: 1,
        bone_count: skeleton.bone_count() as u32,
    })
    .expect("serialize TopologyResult");
}

/// If this fails to compile, an `MlOps` method has been removed or its
/// signature has drifted in a way that breaks downstream callers.
#[test]
fn ml_ops_trait_object_is_callable() {
    fn _accept(boxed: &dyn MlOps) {
        let _ = boxed.capabilities();
        let _ = boxed.run_curve_copilot(CopilotRequest {
            model_path: String::new(),
            property_type: 0,
            context: vec![],
            topology_features: vec![],
            bone_name_tokens: vec![],
            query_times: vec![],
            curve_window: vec![],
            bone_context_keyframes: vec![],
            bone_context_topology: vec![],
            bone_context_rest_positions: vec![],
            bone_context_mask: vec![],
        });
        let _ = boxed.run_curve_predict(CurvePredictRequest {
            model_path: String::new(),
            flat_input: vec![],
        });
        let _ = boxed.compute_topology(&SkeletonSnapshot {
            bone_names: vec![],
            parent_indices: vec![],
            local_matrices: vec![],
        });
        let _ = boxed.call_op("noop", &[]);
    }
}
