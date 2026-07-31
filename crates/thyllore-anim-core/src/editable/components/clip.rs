use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::BoneId;

use super::curve::{PropertyCurve, PropertyType};
use super::keyframe::SourceClipId;
use super::track::BoneTrack;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditableAnimationClip {
    pub id: SourceClipId,
    pub name: String,
    pub duration: f32,
    pub tracks: HashMap<BoneId, BoneTrack>,
    #[serde(default)]
    pub scalar_curves: Vec<PropertyCurve>,
    pub source_path: Option<String>,
    next_curve_id: u64,
}

impl EditableAnimationClip {
    pub fn new(id: SourceClipId, name: String) -> Self {
        Self {
            id,
            name,
            duration: 0.0,
            tracks: HashMap::new(),
            scalar_curves: Vec::new(),
            source_path: None,
            next_curve_id: 1,
        }
    }

    pub fn add_track(&mut self, bone_id: BoneId, bone_name: String) -> &mut BoneTrack {
        let base_curve_id = self.next_curve_id;
        self.next_curve_id += 10;

        let track = BoneTrack::new(bone_id, bone_name, base_curve_id);
        self.tracks.insert(bone_id, track);
        self.tracks
            .get_mut(&bone_id)
            .expect("track was just inserted above")
    }

    pub fn remove_track(&mut self, bone_id: BoneId) -> Option<BoneTrack> {
        self.tracks.remove(&bone_id)
    }

    pub fn get_track(&self, bone_id: BoneId) -> Option<&BoneTrack> {
        self.tracks.get(&bone_id)
    }

    pub fn get_track_mut(&mut self, bone_id: BoneId) -> Option<&mut BoneTrack> {
        self.tracks.get_mut(&bone_id)
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    pub fn total_keyframe_count(&self) -> usize {
        let bone_keys: usize = self.tracks.values().map(|t| t.total_keyframe_count()).sum();
        let scalar_keys: usize = self.scalar_curves.iter().map(|c| c.keyframe_count()).sum();
        bone_keys + scalar_keys
    }

    pub fn get_scalar_curve(&self, property_type: PropertyType) -> Option<&PropertyCurve> {
        self.scalar_curves
            .iter()
            .find(|c| c.property_type == property_type)
    }

    pub fn get_scalar_curve_mut(
        &mut self,
        property_type: PropertyType,
    ) -> Option<&mut PropertyCurve> {
        self.scalar_curves
            .iter_mut()
            .find(|c| c.property_type == property_type)
    }

    pub fn get_or_add_scalar_curve(&mut self, property_type: PropertyType) -> &mut PropertyCurve {
        if let Some(idx) = self
            .scalar_curves
            .iter()
            .position(|c| c.property_type == property_type)
        {
            return &mut self.scalar_curves[idx];
        }
        let id = self.next_curve_id;
        self.next_curve_id += 1;
        self.scalar_curves
            .push(PropertyCurve::new(id, property_type));
        self.scalar_curves.last_mut().expect("curve just pushed")
    }

    pub fn remove_empty_scalar_curves(&mut self) {
        self.scalar_curves.retain(|c| !c.is_empty());
    }

    pub fn has_scalar_keyframes(&self) -> bool {
        self.scalar_curves.iter().any(|c| !c.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_curve_ids_and_lookup() {
        let mut clip = EditableAnimationClip::new(1, "fx".to_string());
        let a = clip.get_or_add_scalar_curve(PropertyType::Custom(3)).id;
        let b = clip.get_or_add_scalar_curve(PropertyType::Custom(7)).id;
        assert_ne!(a, b);
        // Re-request returns the same curve, no new allocation
        assert_eq!(clip.get_or_add_scalar_curve(PropertyType::Custom(3)).id, a);
        assert_eq!(clip.scalar_curves.len(), 2);
        assert!(clip.get_scalar_curve(PropertyType::Custom(7)).is_some());
        assert!(clip.get_scalar_curve(PropertyType::Custom(99)).is_none());
    }

    #[test]
    fn test_legacy_ron_without_scalar_curves_deserializes() {
        let clip = EditableAnimationClip::new(5, "legacy".to_string());
        let ron = ron::to_string(&clip).expect("serialize");
        // Simulate a pre-scalar_curves file by stripping the field
        let legacy = ron
            .replace("scalar_curves:[],", "")
            .replace("scalar_curves: [],", "");
        assert_ne!(ron, legacy, "field must have been present to strip");
        let parsed: EditableAnimationClip = ron::from_str(&legacy).expect("legacy deserialize");
        assert!(parsed.scalar_curves.is_empty());
        assert_eq!(parsed.name, "legacy");
    }

    #[test]
    fn test_scalar_curves_ron_roundtrip() {
        let mut clip = EditableAnimationClip::new(2, "fx".to_string());
        let curve = clip.get_or_add_scalar_curve(PropertyType::Custom(4));
        let id = curve.allocate_keyframe_id();
        curve
            .keyframes
            .push(crate::editable::EditableKeyframe::new(id, 1.5, 2.5));
        let ron = ron::to_string(&clip).expect("serialize");
        let parsed: EditableAnimationClip = ron::from_str(&ron).expect("deserialize");
        let restored = parsed
            .get_scalar_curve(PropertyType::Custom(4))
            .expect("scalar curve");
        assert_eq!(restored.keyframes.len(), 1);
        assert!((restored.keyframes[0].time - 1.5).abs() < 1e-6);
        assert!((restored.keyframes[0].value - 2.5).abs() < 1e-6);
    }
}

impl Default for EditableAnimationClip {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            duration: 0.0,
            tracks: HashMap::new(),
            scalar_curves: Vec::new(),
            source_path: None,
            next_curve_id: 1,
        }
    }
}
