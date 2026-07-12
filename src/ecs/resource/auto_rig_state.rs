use std::time::Instant;

use crate::ecs::world::Entity;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AutoRigStatus {
    Idle,
    WaitingForServer,
    Rigging,
    Previewing,
    Error,
}

pub struct AutoRigState {
    pub status: AutoRigStatus,
    pub rigged_glb_data: Option<Vec<u8>>,
    pub error_message: Option<String>,
    pub joint_count: Option<u32>,
    pub bone_count: Option<u32>,
    pub generation_time_ms: Option<f32>,
    pub source_glb_data: Option<Vec<u8>>,
    pub original_glb_backup: Option<Vec<u8>>,
    pub target_entity: Option<Entity>,
    pub last_status_check: Option<Instant>,
}

impl Default for AutoRigState {
    fn default() -> Self {
        Self {
            status: AutoRigStatus::Idle,
            rigged_glb_data: None,
            error_message: None,
            joint_count: None,
            bone_count: None,
            generation_time_ms: None,
            source_glb_data: None,
            original_glb_backup: None,
            target_entity: None,
            last_status_check: None,
        }
    }
}
