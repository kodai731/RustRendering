use crate::animation::editable::{BlendMode, ClipGroupId, ClipInstanceId, SourceClipId};
use crate::animation::BoneId;
use crate::ecs::systems::CameraSwitchMarker;
use crate::ecs::world::Entity;

pub struct ClipTrackSnapshot {
    pub entries: Vec<ClipTrackEntry>,
    pub camera_switch_markers: Vec<CameraSwitchMarker>,
}

pub struct ClipTrackEntry {
    pub entity: Entity,
    pub entity_name: String,
    pub mesh_bone_id: Option<BoneId>,
    pub instances: Vec<ClipInstanceSnapshot>,
    pub groups: Vec<ClipGroupSnapshot>,
}

pub struct ClipInstanceSnapshot {
    pub instance_id: ClipInstanceId,
    pub source_id: SourceClipId,
    pub clip_name: String,
    pub start_time: f32,
    pub end_time: f32,
    pub clip_in: f32,
    pub clip_out: f32,
    pub muted: bool,
    pub weight: f32,
    pub blend_mode: BlendMode,
    pub group_id: Option<ClipGroupId>,
}

pub struct ClipGroupSnapshot {
    pub id: ClipGroupId,
    pub name: String,
    pub muted: bool,
    pub weight: f32,
    pub instance_ids: Vec<ClipInstanceId>,
}
