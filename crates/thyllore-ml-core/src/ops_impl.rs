//! L1 implementation of the L2 [`thyllore_ml_api::MlOps`] contract.
//!
//! This module wires the existing `copilot::session::Session`,
//! `topology::compute_bone_topology`, and `tokenizer` helpers behind a single
//! [`MlCoreImpl`] type. The trait methods only adapt request/response shapes;
//! all actual inference logic still lives in the existing modules so the
//! Phase 1-2 parity tests remain intact.

use std::collections::HashMap;
use std::sync::Mutex;

use thyllore_ml_api::{
    CopilotRequest, CopilotRequestWire, CopilotResponse, CopilotResponseWire,
    CopilotStepPrediction, CurvePredictRequest, CurvePredictResponse, MlError, MlOps,
    SkeletonSnapshot, TopologyResult, OP_RUN_CURVE_COPILOT,
};
use thyllore_model_core::{BoneId, Skeleton};

use crate::copilot::input::BoneContextInput;
use crate::copilot::session::{CurveCopilotRequest, Session};
use crate::topology::{compute_bone_topology, BoneTopologyFeatures};

/// Concrete implementation of [`MlOps`].
///
/// Sessions are cached per-`model_path` because building an `ort::Session`
/// from disk takes hundreds of milliseconds; subsequent calls with the same
/// path reuse the cached session. The cache is `Mutex`-protected because
/// `ort::Session::run` requires `&mut`, but the trait exposes `&self` to
/// callers (the Blender wheel holds a single shared instance).
pub struct MlCoreImpl {
    sessions: Mutex<HashMap<String, Session>>,
}

impl MlCoreImpl {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    fn with_session<R, F>(&self, model_path: &str, f: F) -> Result<R, MlError>
    where
        F: FnOnce(&mut Session) -> Result<R, MlError>,
    {
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|e| MlError::internal(format!("session cache poisoned: {e}")))?;

        if !sessions.contains_key(model_path) {
            let new_session = Session::from_onnx_path(model_path)
                .map_err(|e| MlError::inference(format!("load {model_path}: {e}")))?;
            sessions.insert(model_path.to_string(), new_session);
        }

        let session = sessions
            .get_mut(model_path)
            .expect("session was just inserted if missing");
        f(session)
    }
}

impl Default for MlCoreImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl MlOps for MlCoreImpl {
    fn run_curve_copilot(&self, request: CopilotRequest) -> Result<CopilotResponse, MlError> {
        validate_copilot_request(&request)?;

        let CopilotRequest {
            model_path,
            property_type,
            context,
            topology_features,
            bone_name_tokens,
            query_times,
            curve_window,
            bone_context_keyframes,
            bone_context_topology,
            bone_context_rest_positions,
            bone_context_mask,
        } = request;

        let predictions = self.with_session(&model_path, |session| {
            session
                .run_curve_copilot(CurveCopilotRequest {
                    context: &context,
                    property_type_id: property_type,
                    topology_features: &topology_features,
                    bone_name_tokens: &bone_name_tokens,
                    query_times: &query_times,
                    curve_window: &curve_window,
                    bone_context: BoneContextInput {
                        keyframes: &bone_context_keyframes,
                        topology: &bone_context_topology,
                        rest_positions: &bone_context_rest_positions,
                        mask: &bone_context_mask,
                    },
                })
                .map_err(|e| MlError::inference(format!("run_curve_copilot: {e}")))
        })?;

        Ok(CopilotResponse {
            predictions: predictions
                .into_iter()
                .map(|step| CopilotStepPrediction {
                    value: step.value,
                    tangent_in: [step.tangent_in.0, step.tangent_in.1],
                    tangent_out: [step.tangent_out.0, step.tangent_out.1],
                    confidence: step.confidence,
                })
                .collect(),
        })
    }

    fn run_curve_predict(
        &self,
        request: CurvePredictRequest,
    ) -> Result<CurvePredictResponse, MlError> {
        let CurvePredictRequest {
            model_path,
            flat_input,
        } = request;

        let flat_output = self.with_session(&model_path, |session| {
            session
                .run_curve_predict(&flat_input)
                .map_err(|e| MlError::inference(format!("run_curve_predict: {e}")))
        })?;

        Ok(CurvePredictResponse { flat_output })
    }

    fn compute_topology(&self, skeleton: &SkeletonSnapshot) -> Result<TopologyResult, MlError> {
        if !skeleton.is_well_formed() {
            return Err(MlError::invalid(
                "SkeletonSnapshot has mismatched parent_indices/local_matrices length",
            ));
        }

        let internal_skeleton = build_internal_skeleton(skeleton)?;
        let features_map = compute_bone_topology(&internal_skeleton);

        let bone_count = internal_skeleton.bones.len();
        let features_per_bone = 6u32;
        let mut flat_features = Vec::with_capacity(bone_count * features_per_bone as usize);

        for bone in &internal_skeleton.bones {
            let features = features_map
                .get(&bone.id)
                .cloned()
                .unwrap_or(BoneTopologyFeatures::default());
            flat_features.extend_from_slice(&features.to_vec());
        }

        Ok(TopologyResult {
            flat_features,
            features_per_bone,
            bone_count: bone_count as u32,
        })
    }

    fn call_op(&self, op_name: &str, payload: &[u8]) -> Result<Vec<u8>, MlError> {
        match op_name {
            OP_RUN_CURVE_COPILOT => self.dispatch_run_curve_copilot(payload),
            _ => Err(MlError::unsupported(op_name)),
        }
    }

    fn capabilities(&self) -> Vec<&'static str> {
        vec!["curve_copilot", "curve_predict", "compute_topology"]
    }
}

impl MlCoreImpl {
    fn dispatch_run_curve_copilot(&self, payload: &[u8]) -> Result<Vec<u8>, MlError> {
        let wire_request: CopilotRequestWire = serde_json::from_slice(payload).map_err(|e| {
            MlError::schema(OP_RUN_CURVE_COPILOT, format!("payload decode failed: {e}"))
        })?;
        let typed_request = wire_request.into_typed();
        let typed_response = self.run_curve_copilot(typed_request)?;
        let wire_response = CopilotResponseWire::from_typed(&typed_response);
        serde_json::to_vec(&wire_response).map_err(|e| {
            MlError::internal(format!(
                "response encode failed for {OP_RUN_CURVE_COPILOT}: {e}"
            ))
        })
    }
}

fn validate_copilot_request(request: &CopilotRequest) -> Result<(), MlError> {
    use crate::copilot::input::{
        BONE_CONTEXT_N_MAX, BONE_CONTEXT_REST_POSITION_DIM, BONE_NAME_TOKEN_DIM,
        CONTEXT_FEATURE_DIM, CONTEXT_KEYFRAME_COUNT, TOPOLOGY_FEATURE_DIM,
    };

    let expected_context = CONTEXT_KEYFRAME_COUNT * CONTEXT_FEATURE_DIM;
    if request.context.len() != expected_context {
        return Err(MlError::invalid(format!(
            "context: expected {} floats, got {}",
            expected_context,
            request.context.len()
        )));
    }
    if request.topology_features.len() != TOPOLOGY_FEATURE_DIM {
        return Err(MlError::invalid(format!(
            "topology_features: expected {} floats, got {}",
            TOPOLOGY_FEATURE_DIM,
            request.topology_features.len()
        )));
    }
    if request.bone_name_tokens.len() != BONE_NAME_TOKEN_DIM {
        return Err(MlError::invalid(format!(
            "bone_name_tokens: expected {} ids, got {}",
            BONE_NAME_TOKEN_DIM,
            request.bone_name_tokens.len()
        )));
    }
    if request.query_times.is_empty() {
        return Err(MlError::invalid("query_times must not be empty"));
    }
    if request.curve_window.is_empty() {
        return Err(MlError::invalid("curve_window must not be empty"));
    }

    let n = BONE_CONTEXT_N_MAX;
    let expected_bc_keyframes = n * CONTEXT_KEYFRAME_COUNT * CONTEXT_FEATURE_DIM;
    if request.bone_context_keyframes.len() != expected_bc_keyframes {
        return Err(MlError::invalid(format!(
            "bone_context_keyframes: expected {} floats, got {}",
            expected_bc_keyframes,
            request.bone_context_keyframes.len()
        )));
    }
    let expected_bc_topology = n * TOPOLOGY_FEATURE_DIM;
    if request.bone_context_topology.len() != expected_bc_topology {
        return Err(MlError::invalid(format!(
            "bone_context_topology: expected {} floats, got {}",
            expected_bc_topology,
            request.bone_context_topology.len()
        )));
    }
    let expected_bc_rest = n * BONE_CONTEXT_REST_POSITION_DIM;
    if request.bone_context_rest_positions.len() != expected_bc_rest {
        return Err(MlError::invalid(format!(
            "bone_context_rest_positions: expected {} floats, got {}",
            expected_bc_rest,
            request.bone_context_rest_positions.len()
        )));
    }
    if request.bone_context_mask.len() != n {
        return Err(MlError::invalid(format!(
            "bone_context_mask: expected {} bools, got {}",
            n,
            request.bone_context_mask.len()
        )));
    }
    Ok(())
}

fn build_internal_skeleton(snapshot: &SkeletonSnapshot) -> Result<Skeleton, MlError> {
    let mut skeleton = Skeleton::new("from_snapshot");
    for (index, name) in snapshot.bone_names.iter().enumerate() {
        let parent_index = snapshot.parent_indices[index];
        let parent = if parent_index < 0 {
            None
        } else {
            let pid = parent_index as usize;
            if pid >= index {
                return Err(MlError::invalid(format!(
                    "bone {index} references parent {pid} which is not yet defined"
                )));
            }
            Some(pid as BoneId)
        };
        skeleton.add_bone(name, parent);
    }
    Ok(skeleton)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn three_bone_snapshot() -> SkeletonSnapshot {
        SkeletonSnapshot {
            bone_names: vec!["root".into(), "child".into(), "leaf".into()],
            parent_indices: vec![-1, 0, 1],
            local_matrices: vec![[[0.0; 4]; 4]; 3],
        }
    }

    #[test]
    fn capabilities_lists_the_advertised_ops() {
        let impl_ = MlCoreImpl::new();
        let caps = impl_.capabilities();
        assert!(caps.contains(&"curve_copilot"));
        assert!(caps.contains(&"curve_predict"));
        assert!(caps.contains(&"compute_topology"));
    }

    #[test]
    fn call_op_rejects_unknown_ops_with_unsupported() {
        let impl_ = MlCoreImpl::new();
        let err = impl_.call_op("nonexistent", &[]).unwrap_err();
        assert!(matches!(err, MlError::UnsupportedOp(name) if name == "nonexistent"));
    }

    #[test]
    fn compute_topology_produces_flat_features() {
        let impl_ = MlCoreImpl::new();
        let result = impl_.compute_topology(&three_bone_snapshot()).unwrap();

        assert_eq!(result.bone_count, 3);
        assert_eq!(result.features_per_bone, 6);
        assert_eq!(result.flat_features.len(), 3 * 6);
    }

    #[test]
    fn compute_topology_rejects_malformed_snapshot() {
        let impl_ = MlCoreImpl::new();
        let bad = SkeletonSnapshot {
            bone_names: vec!["a".into(), "b".into()],
            parent_indices: vec![-1],
            local_matrices: vec![[[0.0; 4]; 4]; 2],
        };
        let err = impl_.compute_topology(&bad).unwrap_err();
        assert!(matches!(err, MlError::InvalidRequest(_)));
    }

    #[test]
    fn compute_topology_rejects_forward_parent_reference() {
        let impl_ = MlCoreImpl::new();
        let bad = SkeletonSnapshot {
            bone_names: vec!["a".into(), "b".into()],
            parent_indices: vec![1, -1], // bone 0 claims bone 1 as parent
            local_matrices: vec![[[0.0; 4]; 4]; 2],
        };
        let err = impl_.compute_topology(&bad).unwrap_err();
        assert!(matches!(err, MlError::InvalidRequest(_)));
    }

    fn well_shaped_bone_context_fields() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>) {
        use crate::copilot::input::{
            BONE_CONTEXT_N_MAX, BONE_CONTEXT_REST_POSITION_DIM, CONTEXT_FEATURE_DIM,
            CONTEXT_KEYFRAME_COUNT, TOPOLOGY_FEATURE_DIM,
        };
        let n = BONE_CONTEXT_N_MAX;
        (
            vec![0.0; n * CONTEXT_KEYFRAME_COUNT * CONTEXT_FEATURE_DIM],
            vec![0.0; n * TOPOLOGY_FEATURE_DIM],
            vec![0.0; n * BONE_CONTEXT_REST_POSITION_DIM],
            vec![false; n],
        )
    }

    #[test]
    fn run_curve_copilot_validates_input_shapes() {
        let impl_ = MlCoreImpl::new();
        let (bc_keyframes, bc_topology, bc_rest, bc_mask) = well_shaped_bone_context_fields();

        let bad_request = CopilotRequest {
            model_path: "/tmp/does-not-matter.onnx".into(),
            property_type: 0,
            context: vec![0.0; 7],
            topology_features: vec![0.0; 6],
            bone_name_tokens: vec![0; 32],
            query_times: vec![1.0],
            curve_window: vec![0.0; 64],
            bone_context_keyframes: bc_keyframes,
            bone_context_topology: bc_topology,
            bone_context_rest_positions: bc_rest,
            bone_context_mask: bc_mask,
        };
        let err = impl_.run_curve_copilot(bad_request).unwrap_err();
        assert!(matches!(err, MlError::InvalidRequest(_)));
    }

    #[test]
    fn run_curve_copilot_rejects_malformed_bone_context() {
        let impl_ = MlCoreImpl::new();
        let (_, bc_topology, bc_rest, bc_mask) = well_shaped_bone_context_fields();

        let bad_request = CopilotRequest {
            model_path: "/tmp/does-not-matter.onnx".into(),
            property_type: 0,
            context: vec![0.0; 32 * 6],
            topology_features: vec![0.0; 6],
            bone_name_tokens: vec![0; 32],
            query_times: vec![1.0],
            curve_window: vec![0.0; 64],
            bone_context_keyframes: vec![0.0; 10],
            bone_context_topology: bc_topology,
            bone_context_rest_positions: bc_rest,
            bone_context_mask: bc_mask,
        };
        let err = impl_.run_curve_copilot(bad_request).unwrap_err();
        match err {
            MlError::InvalidRequest(msg) => {
                assert!(msg.contains("bone_context_keyframes"), "got: {msg}")
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn call_op_run_curve_copilot_rejects_malformed_payload() {
        let impl_ = MlCoreImpl::new();
        let err = impl_
            .call_op(OP_RUN_CURVE_COPILOT, b"not valid json")
            .unwrap_err();
        assert!(matches!(err, MlError::SchemaMismatch { .. }));
    }

    #[test]
    fn call_op_run_curve_copilot_validates_input_shapes() {
        let impl_ = MlCoreImpl::new();
        let (bc_keyframes, bc_topology, bc_rest, bc_mask) = well_shaped_bone_context_fields();
        let to_bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

        let wire_request = CopilotRequestWire {
            model_path: "/tmp/does-not-matter.onnx".into(),
            property_type: 0,
            context_bits: vec![0; 7],
            topology_features_bits: vec![0; 6],
            bone_name_tokens: vec![0; 32],
            query_times_bits: vec![1u32.to_le()],
            curve_window_bits: vec![0; 64],
            bone_context_keyframes_bits: to_bits(&bc_keyframes),
            bone_context_topology_bits: to_bits(&bc_topology),
            bone_context_rest_positions_bits: to_bits(&bc_rest),
            bone_context_mask: bc_mask,
        };
        let payload = serde_json::to_vec(&wire_request).expect("encode wire request");
        let err = impl_.call_op(OP_RUN_CURVE_COPILOT, &payload).unwrap_err();
        assert!(matches!(err, MlError::InvalidRequest(_)));
    }
}
