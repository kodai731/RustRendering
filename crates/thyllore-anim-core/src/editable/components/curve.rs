use serde::{Deserialize, Serialize};

use super::keyframe::{
    BezierHandle, CurveId, EditableKeyframe, InterpolationType, KeyframeId, TangentWeightMode,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyType {
    TranslationX,
    TranslationY,
    TranslationZ,
    RotationX,
    RotationY,
    RotationZ,
    ScaleX,
    ScaleY,
    ScaleZ,
}

impl PropertyType {
    pub fn display_name(&self) -> &'static str {
        match self {
            PropertyType::TranslationX => "Position X",
            PropertyType::TranslationY => "Position Y",
            PropertyType::TranslationZ => "Position Z",
            PropertyType::RotationX => "Rotation X",
            PropertyType::RotationY => "Rotation Y",
            PropertyType::RotationZ => "Rotation Z",
            PropertyType::ScaleX => "Scale X",
            PropertyType::ScaleY => "Scale Y",
            PropertyType::ScaleZ => "Scale Z",
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            PropertyType::TranslationX => "Pos.X",
            PropertyType::TranslationY => "Pos.Y",
            PropertyType::TranslationZ => "Pos.Z",
            PropertyType::RotationX => "Rot.X",
            PropertyType::RotationY => "Rot.Y",
            PropertyType::RotationZ => "Rot.Z",
            PropertyType::ScaleX => "Scl.X",
            PropertyType::ScaleY => "Scl.Y",
            PropertyType::ScaleZ => "Scl.Z",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PropertyCurve {
    pub id: CurveId,
    pub property_type: PropertyType,
    pub keyframes: Vec<EditableKeyframe>,
    next_keyframe_id: KeyframeId,
}

impl PropertyCurve {
    pub fn new(id: CurveId, property_type: PropertyType) -> Self {
        Self {
            id,
            property_type,
            keyframes: Vec::new(),
            next_keyframe_id: 1,
        }
    }

    pub(crate) fn allocate_keyframe_id(&mut self) -> KeyframeId {
        let id = self.next_keyframe_id;
        self.next_keyframe_id += 1;
        id
    }

    pub fn get_keyframe(&self, keyframe_id: KeyframeId) -> Option<&EditableKeyframe> {
        self.keyframes.iter().find(|k| k.id == keyframe_id)
    }

    pub fn get_keyframe_mut(&mut self, keyframe_id: KeyframeId) -> Option<&mut EditableKeyframe> {
        self.keyframes.iter_mut().find(|k| k.id == keyframe_id)
    }

    pub fn set_keyframe_value(&mut self, keyframe_id: KeyframeId, value: f32) {
        if let Some(kf) = self.get_keyframe_mut(keyframe_id) {
            kf.value = value;
        }
    }

    pub fn set_keyframe_interpolation(
        &mut self,
        keyframe_id: KeyframeId,
        interpolation: InterpolationType,
    ) {
        if let Some(kf) = self.get_keyframe_mut(keyframe_id) {
            kf.interpolation = interpolation;
        }
    }

    pub fn set_keyframe_tangents(
        &mut self,
        keyframe_id: KeyframeId,
        in_tangent: BezierHandle,
        out_tangent: BezierHandle,
    ) {
        if let Some(kf) = self.get_keyframe_mut(keyframe_id) {
            kf.in_tangent = in_tangent;
            kf.out_tangent = out_tangent;
        }
    }

    pub fn set_keyframe_weight_mode(
        &mut self,
        keyframe_id: KeyframeId,
        weight_mode: TangentWeightMode,
    ) {
        if let Some(kf) = self.get_keyframe_mut(keyframe_id) {
            kf.weight_mode = weight_mode;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    pub fn has_bezier_keyframes(&self) -> bool {
        self.keyframes
            .iter()
            .any(|k| k.interpolation == InterpolationType::Bezier)
    }
}

impl Default for PropertyCurve {
    fn default() -> Self {
        Self {
            id: 0,
            property_type: PropertyType::TranslationX,
            keyframes: Vec::new(),
            next_keyframe_id: 1,
        }
    }
}
