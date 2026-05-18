use serde::{Deserialize, Serialize};

/// Skeleton in a flat, serializable form so it can travel through both typed
/// methods and `call_op` envelopes without depending on `thyllore-model-core`.
///
/// Implementations are responsible for converting this into their internal
/// representation; the addon never sees the internal type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonSnapshot {
    pub bone_names: Vec<String>,
    /// One entry per bone. `-1` marks a root.
    pub parent_indices: Vec<i32>,
    /// Row-major 4x4 local transform per bone. Length must equal
    /// `bone_names.len()`.
    pub local_matrices: Vec<[[f32; 4]; 4]>,
}

impl SkeletonSnapshot {
    pub fn bone_count(&self) -> usize {
        self.bone_names.len()
    }

    pub fn is_well_formed(&self) -> bool {
        let n = self.bone_names.len();
        self.parent_indices.len() == n && self.local_matrices.len() == n
    }
}

/// Request for the curve copilot inference pipeline.
///
/// Each field is sized to the model's expected input shape:
///
/// - `context`: `32 * 6 = 192` flat values (32 keyframes × 6 features each).
/// - `topology_features`: 6 features describing the target bone's place in the
///   skeleton hierarchy.
/// - `bone_name_tokens`: 32 token ids derived from the bone name.
/// - `query_times`: arbitrary `Q` predictions.
/// - `curve_window`: densely sampled curve values around the context window.
/// - `bone_context_keyframes`: `32 * 32 * 6 = 6144` flat values feeding the
///   ISAB cross-bone encoder (one keyframe block per neighbor slot).
/// - `bone_context_topology`: `32 * 6 = 192` per-neighbor topology features.
/// - `bone_context_rest_positions`: `32 * 3 = 96` per-neighbor normalized
///   rest positions.
/// - `bone_context_mask`: 32 booleans marking which neighbor slots carry
///   valid context (vs zero padding).
///
/// Adding new fields requires either an `ABI_MARKER` bump or `#[serde(default)]`
/// annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotRequest {
    /// Onnx model path. Implementations may cache the loaded session by path.
    pub model_path: String,
    pub property_type: u32,
    pub context: Vec<f32>,
    pub topology_features: Vec<f32>,
    pub bone_name_tokens: Vec<i64>,
    pub query_times: Vec<f32>,
    pub curve_window: Vec<f32>,
    pub bone_context_keyframes: Vec<f32>,
    pub bone_context_topology: Vec<f32>,
    pub bone_context_rest_positions: Vec<f32>,
    pub bone_context_mask: Vec<bool>,
}

/// Request for the legacy single-tensor curve predictor used by the existing
/// Phase 1 pipeline. Kept as a typed entry so the wheel can call it without
/// going through `call_op`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvePredictRequest {
    pub model_path: String,
    pub flat_input: Vec<f32>,
}
