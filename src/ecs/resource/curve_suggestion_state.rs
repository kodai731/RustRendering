use crate::animation::editable::PropertyType;
use crate::animation::BoneId;
use crate::ml::InferenceRequestId;

#[derive(Clone)]
pub struct GhostCurveSuggestion {
    pub bone_id: BoneId,
    pub property_type: PropertyType,
    pub predicted_time: f32,
    pub predicted_value: f32,
    pub tangent_in: (f32, f32),
    pub tangent_out: (f32, f32),
    pub confidence: f32,
    pub request_id: InferenceRequestId,
}

pub struct CurveSuggestionPendingDump {
    pub context: Vec<f32>,
    pub fps: f32,
    pub anchor_time: f32,
}

pub struct PendingSuggestionRequest {
    pub request_id: InferenceRequestId,
    pub bone_id: BoneId,
    pub property_type: PropertyType,
    pub anchor_time: f32,
    pub origin_value: f32,
    pub dt: f32,
    pub dump: Option<CurveSuggestionPendingDump>,
    pub feedback_context: Option<Vec<f32>>,
}

pub struct CurveSuggestionState {
    pub suggestions: Vec<GhostCurveSuggestion>,
    pub pending: Option<PendingSuggestionRequest>,
    pub enabled: bool,
    pub dump_inference: bool,
}

impl Default for CurveSuggestionState {
    fn default() -> Self {
        Self {
            suggestions: Vec::new(),
            pending: None,
            enabled: true,
            dump_inference: false,
        }
    }
}
