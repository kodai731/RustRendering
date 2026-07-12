use std::collections::HashSet;

use crate::animation::editable::{ClipInstanceId, KeyframeId, PropertyType, SourceClipId};
use crate::animation::BoneId;
use crate::ecs::world::Entity;

pub use thyllore_anim_core::editable::SnapSettings;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionModifier {
    Replace,
    Add,
    Toggle,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SelectedKeyframe {
    pub bone_id: BoneId,
    pub property_type: PropertyType,
    pub keyframe_id: KeyframeId,
}

impl SelectedKeyframe {
    pub fn new(bone_id: BoneId, property_type: PropertyType, keyframe_id: KeyframeId) -> Self {
        Self {
            bone_id,
            property_type,
            keyframe_id,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClipDragState {
    pub entity: Entity,
    pub instance_id: ClipInstanceId,
    pub drag_type: ClipDragType,
    pub original_value: f32,
    pub drag_start_x: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClipDragType {
    Move,
    TrimStart,
    TrimEnd,
}

#[derive(Clone, Debug)]
pub struct TimelineState {
    pub current_clip_id: Option<SourceClipId>,
    pub current_time: f32,
    pub playing: bool,
    pub looping: bool,
    pub speed: f32,
    pub zoom_level: f32,
    pub scroll_offset: f32,
    pub last_visible_width: f32,
    pub pan_pending_delta_x: f32,
    pub selected_keyframes: HashSet<SelectedKeyframe>,
    pub expanded_tracks: HashSet<BoneId>,
    pub show_translation: bool,
    pub show_rotation: bool,
    pub show_scale: bool,
    pub target_entity: Option<Entity>,
    pub selected_clip_instance: Option<(Entity, ClipInstanceId)>,
    pub snap_settings: SnapSettings,
    pub baked_bone_ids: Vec<BoneId>,
}

impl TimelineState {
    pub fn new() -> Self {
        Self {
            current_clip_id: None,
            current_time: 0.0,
            playing: false,
            looping: true,
            speed: 1.0,
            zoom_level: 1.0,
            scroll_offset: 0.0,
            last_visible_width: 0.0,
            pan_pending_delta_x: 0.0,
            selected_keyframes: HashSet::new(),
            expanded_tracks: HashSet::new(),
            show_translation: true,
            show_rotation: true,
            show_scale: true,
            target_entity: None,
            selected_clip_instance: None,
            snap_settings: SnapSettings::default(),
            baked_bone_ids: Vec::new(),
        }
    }

    pub fn select_keyframe(&mut self, keyframe: SelectedKeyframe) {
        self.selected_keyframes.clear();
        self.selected_keyframes.insert(keyframe);
    }

    pub fn add_keyframe_to_selection(&mut self, keyframe: SelectedKeyframe) {
        self.selected_keyframes.insert(keyframe);
    }

    pub fn remove_keyframe_from_selection(&mut self, keyframe: &SelectedKeyframe) {
        self.selected_keyframes.remove(keyframe);
    }

    pub fn clear_selection(&mut self) {
        self.selected_keyframes.clear();
    }

    pub fn is_keyframe_selected(&self, keyframe: &SelectedKeyframe) -> bool {
        self.selected_keyframes.contains(keyframe)
    }

    pub fn is_track_expanded(&self, bone_id: BoneId) -> bool {
        self.expanded_tracks.contains(&bone_id)
    }

    pub fn expand_track(&mut self, bone_id: BoneId) {
        self.expanded_tracks.insert(bone_id);
    }

    pub fn collapse_track(&mut self, bone_id: BoneId) {
        self.expanded_tracks.remove(&bone_id);
    }

    pub fn set_time(&mut self, time: f32) {
        self.current_time = time.max(0.0);
    }
}

impl Default for TimelineState {
    fn default() -> Self {
        Self::new()
    }
}
