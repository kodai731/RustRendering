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
    pub context_keyframes: Vec<f32>,
    pub property_type_id: u32,
    pub topology_features: Vec<f32>,
    pub bone_name_tokens: Vec<i64>,
    pub query_times_normalized: Vec<f32>,
    pub curve_window: Vec<f32>,
    pub bone_context_keyframes: Vec<f32>,
    pub bone_context_topology: Vec<f32>,
    pub bone_context_rest_positions: Vec<f32>,
    pub bone_context_mask: Vec<bool>,
}

pub struct CurveSuggestionState {
    pub suggestions: Vec<GhostCurveSuggestion>,
    pub pending_request_id: Option<InferenceRequestId>,
    pub pending_bone_id: Option<BoneId>,
    pub pending_property_type: Option<PropertyType>,
    pub pending_clip_duration: Option<f32>,
    pub pending_curve_mean: Option<f32>,
    pub pending_curve_std: Option<f32>,
    pub pending_query_times: Option<Vec<f32>>,
    pub pending_real_curve_samples: Option<Vec<Option<f32>>>,
    pub pending_nearest_keyframes: Option<Vec<Option<(f32, f32)>>>,
    pub pending_dump: Option<CurveSuggestionPendingDump>,
    pub enabled: bool,
    pub dump_inference: bool,
}

impl Default for CurveSuggestionState {
    fn default() -> Self {
        Self {
            suggestions: Vec::new(),
            pending_request_id: None,
            pending_bone_id: None,
            pending_property_type: None,
            pending_clip_duration: None,
            pending_curve_mean: None,
            pending_curve_std: None,
            pending_query_times: None,
            pending_real_curve_samples: None,
            pending_nearest_keyframes: None,
            pending_dump: None,
            enabled: true,
            dump_inference: false,
        }
    }
}
