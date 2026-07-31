use std::collections::HashSet;

use super::timeline_state::CurveTrackRef;
use crate::animation::editable::{BezierHandle, KeyframeId, PropertyType};
use crate::animation::BoneId;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CurveEditorTarget {
    Bone(BoneId),
    /// Clip-level scalar curves (`PropertyType::Custom`), e.g. flame parameters.
    Scalars,
}

#[derive(Clone, Debug)]
pub struct CurveSelectedKeyframe {
    pub property_type: PropertyType,
    pub keyframe_id: KeyframeId,
    pub original_time: f32,
    pub original_value: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TangentHandleType {
    In,
    Out,
}

#[derive(Clone, Debug)]
pub struct DraggingTangent {
    pub property_type: PropertyType,
    pub keyframe_id: KeyframeId,
    pub handle_type: TangentHandleType,
    pub original_handle: BezierHandle,
}

#[derive(Clone, Debug)]
pub enum CurveInteractionMode {
    Idle,
    DraggingKeyframe,
    ScrubbingRuler,
    Panning {
        start_mouse_pos: [f32; 2],
        start_offset: [f32; 2],
    },
    DraggingTangent(DraggingTangent),
}

impl Default for CurveInteractionMode {
    fn default() -> Self {
        Self::Idle
    }
}

pub struct CurveEditorState {
    pub is_open: bool,
    pub selected_target: Option<CurveEditorTarget>,
    pub visible_curves: HashSet<PropertyType>,
    pub window_size: [f32; 2],
    pub selected_keyframes: Vec<CurveSelectedKeyframe>,
    pub selection_anchor: Option<(PropertyType, KeyframeId)>,
    pub interaction: CurveInteractionMode,
    pub drag_start_mouse_pos: [f32; 2],
    pub zoom_x: f32,
    pub zoom_y: f32,
    pub view_time_offset: f32,
    pub view_value_offset: f32,
    pub view_val_range: f32,
    pub view_duration: f32,
    pub view_initialized: bool,
    pub context_menu_keyframe: Option<CurveSelectedKeyframe>,
    pub context_menu_click_time: f32,
    pub context_menu_click_value: f32,
    pub needs_focus: bool,
}

impl CurveEditorState {
    pub fn selected_bone_id(&self) -> Option<BoneId> {
        match self.selected_target {
            Some(CurveEditorTarget::Bone(id)) => Some(id),
            _ => None,
        }
    }

    pub fn select_bone(&mut self, bone_id: BoneId) {
        self.selected_target = Some(CurveEditorTarget::Bone(bone_id));
    }

    pub fn select_scalars(&mut self) {
        self.selected_target = Some(CurveEditorTarget::Scalars);
    }

    pub fn selected_track_ref(&self) -> Option<CurveTrackRef> {
        match self.selected_target {
            Some(CurveEditorTarget::Bone(id)) => Some(CurveTrackRef::Bone(id)),
            Some(CurveEditorTarget::Scalars) => Some(CurveTrackRef::Scalar),
            None => None,
        }
    }
}

impl Default for CurveEditorState {
    fn default() -> Self {
        let mut visible_curves = HashSet::new();
        visible_curves.insert(PropertyType::TranslationX);
        visible_curves.insert(PropertyType::TranslationY);
        visible_curves.insert(PropertyType::TranslationZ);
        visible_curves.insert(PropertyType::RotationX);
        visible_curves.insert(PropertyType::RotationY);
        visible_curves.insert(PropertyType::RotationZ);

        Self {
            is_open: false,
            selected_target: None,
            visible_curves,
            window_size: [800.0, 500.0],
            selected_keyframes: Vec::new(),
            selection_anchor: None,
            interaction: CurveInteractionMode::Idle,
            drag_start_mouse_pos: [0.0, 0.0],
            zoom_x: 1.0,
            zoom_y: 1.0,
            view_time_offset: 0.0,
            view_value_offset: 0.0,
            view_val_range: 2.0,
            view_duration: 2.0,
            view_initialized: false,
            context_menu_keyframe: None,
            context_menu_click_time: 0.0,
            context_menu_click_value: 0.0,
            needs_focus: false,
        }
    }
}
