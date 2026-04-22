use cgmath::Vector3;

use crate::BoneId;

pub type SpringChainId = u32;
pub type SpringColliderId = u32;
pub type SpringColliderGroupId = u32;

#[derive(Clone, Debug)]
pub struct SpringJointParam {
    pub bone_id: BoneId,
    pub stiffness: f32,
    pub drag_force: f32,
    pub gravity_power: f32,
    pub gravity_dir: Vector3<f32>,
    pub hit_radius: f32,
}

impl Default for SpringJointParam {
    fn default() -> Self {
        Self {
            bone_id: 0,
            stiffness: 0.5,
            drag_force: 0.4,
            gravity_power: 1.0,
            gravity_dir: Vector3::new(0.0, -1.0, 0.0),
            hit_radius: 0.02,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpringChain {
    pub id: SpringChainId,
    pub name: String,
    pub joints: Vec<SpringJointParam>,
    pub collider_group_ids: Vec<SpringColliderGroupId>,
    pub center_bone_id: Option<BoneId>,
    pub enabled: bool,
}

impl Default for SpringChain {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            joints: Vec::new(),
            collider_group_ids: Vec::new(),
            center_bone_id: None,
            enabled: true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ColliderShape {
    Sphere { radius: f32 },
    Capsule { radius: f32, tail: Vector3<f32> },
}

#[derive(Clone, Debug)]
pub struct SpringColliderDef {
    pub id: SpringColliderId,
    pub bone_id: BoneId,
    pub offset: Vector3<f32>,
    pub shape: ColliderShape,
}

#[derive(Clone, Debug)]
pub struct SpringColliderGroup {
    pub id: SpringColliderGroupId,
    pub name: String,
    pub collider_ids: Vec<SpringColliderId>,
}

#[derive(Clone, Debug, Default)]
pub struct SpringBoneSetup {
    pub chains: Vec<SpringChain>,
    pub colliders: Vec<SpringColliderDef>,
    pub collider_groups: Vec<SpringColliderGroup>,
    pub next_chain_id: SpringChainId,
    pub next_collider_id: SpringColliderId,
    pub next_group_id: SpringColliderGroupId,
}
