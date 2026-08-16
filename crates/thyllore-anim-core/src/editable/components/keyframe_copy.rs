use crate::BoneId;

use super::curve::PropertyType;
use super::keyframe::{BezierHandle, InterpolationType, SourceClipId, TangentWeightMode};

#[derive(Clone, Debug)]
pub struct CopiedKeyframe {
    /// `Some` for bone-track curves; `None` for clip-level scalar curves
    /// (`PropertyType::Custom`), which have no bone and are never mirrored.
    pub bone_id: Option<BoneId>,
    pub property_type: PropertyType,
    pub relative_time: f32,
    pub value: f32,
    pub interpolation: InterpolationType,
    pub in_tangent: BezierHandle,
    pub out_tangent: BezierHandle,
    pub weight_mode: TangentWeightMode,
}

#[derive(Clone, Debug, Default)]
pub struct KeyframeCopyBuffer {
    pub entries: Vec<CopiedKeyframe>,
    pub base_time: f32,
    pub source_clip_id: Option<SourceClipId>,
}

impl KeyframeCopyBuffer {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.base_time = 0.0;
        self.source_clip_id = None;
    }
}
