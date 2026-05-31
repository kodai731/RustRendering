use thyllore_anim_core::editable::PropertyType;

use crate::{BoneId, InferenceRequestId};

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
    pub future: Vec<f32>,
    pub reveal_mask: Vec<bool>,
    pub fps: f32,
    pub anchor_time: f32,
}

pub struct CurveSuggestionState {
    pub suggestions: Vec<GhostCurveSuggestion>,
    pub pending_request_id: Option<InferenceRequestId>,
    pub pending_bone_id: Option<BoneId>,
    pub pending_property_type: Option<PropertyType>,
    pub pending_anchor_time: Option<f32>,
    pub pending_origin_value: Option<f32>,
    pub pending_dt: Option<f32>,
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
            pending_anchor_time: None,
            pending_origin_value: None,
            pending_dt: None,
            pending_dump: None,
            enabled: true,
            dump_inference: false,
        }
    }
}
