use serde::{Deserialize, Serialize};

/// Single predicted keyframe paired with its confidence.
///
/// The time at which this prediction applies is supplied by the caller through
/// `CopilotRequest::query_times`; the response is index-aligned with that
/// vector, so the time is not repeated here.
///
/// `tangent_in` and `tangent_out` follow Blender's bezier handle convention:
/// `[handle_x_offset, handle_y_offset]` relative to the keyframe origin.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CopilotStepPrediction {
    pub value: f32,
    pub tangent_in: [f32; 2],
    pub tangent_out: [f32; 2],
    pub confidence: f32,
}

/// Response for [`crate::CopilotRequest`]. The `predictions` slice has the same
/// length as the request's `query_times`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotResponse {
    pub predictions: Vec<CopilotStepPrediction>,
}

impl CopilotResponse {
    pub fn len(&self) -> usize {
        self.predictions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.predictions.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvePredictResponse {
    pub flat_output: Vec<f32>,
}

/// Output of [`crate::MlOps::compute_topology`] for a given skeleton.
///
/// Shape and ordering follow the existing `topology` module in
/// `thyllore-ml-core`; the field is exposed flat so wheel callers can drop it
/// straight into a numpy buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyResult {
    /// Flat features per bone. The number of features per bone is implementation
    /// defined and reported via `features_per_bone`.
    pub flat_features: Vec<f32>,
    pub features_per_bone: u32,
    pub bone_count: u32,
}
