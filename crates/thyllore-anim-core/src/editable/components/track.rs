use serde::{Deserialize, Serialize};

use crate::BoneId;

use super::curve::{PropertyCurve, PropertyType};
use super::keyframe::CurveId;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoneTrack {
    pub bone_id: BoneId,
    pub bone_name: String,
    pub translation_x: PropertyCurve,
    pub translation_y: PropertyCurve,
    pub translation_z: PropertyCurve,
    pub rotation_x: PropertyCurve,
    pub rotation_y: PropertyCurve,
    pub rotation_z: PropertyCurve,
    pub scale_x: PropertyCurve,
    pub scale_y: PropertyCurve,
    pub scale_z: PropertyCurve,
}

impl BoneTrack {
    pub fn new(bone_id: BoneId, bone_name: String, base_curve_id: CurveId) -> Self {
        Self {
            bone_id,
            bone_name,
            translation_x: PropertyCurve::new(base_curve_id, PropertyType::TranslationX),
            translation_y: PropertyCurve::new(base_curve_id + 1, PropertyType::TranslationY),
            translation_z: PropertyCurve::new(base_curve_id + 2, PropertyType::TranslationZ),
            rotation_x: PropertyCurve::new(base_curve_id + 3, PropertyType::RotationX),
            rotation_y: PropertyCurve::new(base_curve_id + 4, PropertyType::RotationY),
            rotation_z: PropertyCurve::new(base_curve_id + 5, PropertyType::RotationZ),
            scale_x: PropertyCurve::new(base_curve_id + 6, PropertyType::ScaleX),
            scale_y: PropertyCurve::new(base_curve_id + 7, PropertyType::ScaleY),
            scale_z: PropertyCurve::new(base_curve_id + 8, PropertyType::ScaleZ),
        }
    }

    pub fn get_curve(&self, property_type: PropertyType) -> &PropertyCurve {
        match property_type {
            PropertyType::TranslationX => &self.translation_x,
            PropertyType::TranslationY => &self.translation_y,
            PropertyType::TranslationZ => &self.translation_z,
            PropertyType::RotationX => &self.rotation_x,
            PropertyType::RotationY => &self.rotation_y,
            PropertyType::RotationZ => &self.rotation_z,
            PropertyType::ScaleX => &self.scale_x,
            PropertyType::ScaleY => &self.scale_y,
            PropertyType::ScaleZ => &self.scale_z,
        }
    }

    pub fn get_curve_mut(&mut self, property_type: PropertyType) -> &mut PropertyCurve {
        match property_type {
            PropertyType::TranslationX => &mut self.translation_x,
            PropertyType::TranslationY => &mut self.translation_y,
            PropertyType::TranslationZ => &mut self.translation_z,
            PropertyType::RotationX => &mut self.rotation_x,
            PropertyType::RotationY => &mut self.rotation_y,
            PropertyType::RotationZ => &mut self.rotation_z,
            PropertyType::ScaleX => &mut self.scale_x,
            PropertyType::ScaleY => &mut self.scale_y,
            PropertyType::ScaleZ => &mut self.scale_z,
        }
    }

    pub fn all_curves(&self) -> [&PropertyCurve; 9] {
        [
            &self.translation_x,
            &self.translation_y,
            &self.translation_z,
            &self.rotation_x,
            &self.rotation_y,
            &self.rotation_z,
            &self.scale_x,
            &self.scale_y,
            &self.scale_z,
        ]
    }

    pub fn has_any_keyframes(&self) -> bool {
        self.all_curves().iter().any(|c| !c.is_empty())
    }

    pub fn total_keyframe_count(&self) -> usize {
        self.all_curves().iter().map(|c| c.keyframe_count()).sum()
    }

    pub fn has_translation_keyframes(&self) -> bool {
        !self.translation_x.is_empty()
            || !self.translation_y.is_empty()
            || !self.translation_z.is_empty()
    }

    pub fn has_rotation_keyframes(&self) -> bool {
        !self.rotation_x.is_empty() || !self.rotation_y.is_empty() || !self.rotation_z.is_empty()
    }

    pub fn has_scale_keyframes(&self) -> bool {
        !self.scale_x.is_empty() || !self.scale_y.is_empty() || !self.scale_z.is_empty()
    }

    pub fn collect_all_keyframe_times(&self) -> Vec<f32> {
        let mut times: Vec<f32> = self
            .all_curves()
            .iter()
            .flat_map(|c| c.keyframes.iter().map(|kf| kf.time))
            .collect();

        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        times.dedup_by(|a, b| (*a - *b).abs() < 0.001);
        times
    }
}
