use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::BoneId;

use super::keyframe::SourceClipId;
use super::track::BoneTrack;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditableAnimationClip {
    pub id: SourceClipId,
    pub name: String,
    pub duration: f32,
    pub tracks: HashMap<BoneId, BoneTrack>,
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
        self.tracks.values().map(|t| t.total_keyframe_count()).sum()
    }
}

impl Default for EditableAnimationClip {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            duration: 0.0,
            tracks: HashMap::new(),
            source_path: None,
            next_curve_id: 1,
        }
    }
}
