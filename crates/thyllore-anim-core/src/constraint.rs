use cgmath::{Quaternion, Vector3};

use crate::BoneId;

pub type ConstraintId = u64;

pub const PRIORITY_PARENT: u32 = 100;
pub const PRIORITY_POSITION: u32 = 200;
pub const PRIORITY_ROTATION: u32 = 200;
pub const PRIORITY_SCALE: u32 = 200;
pub const PRIORITY_AIM: u32 = 300;
pub const PRIORITY_IK: u32 = 400;

#[derive(Clone, Debug)]
pub struct IkConstraintData {
    pub chain_length: u32,
    pub target_bone: BoneId,
    pub effector_bone: BoneId,
    pub pole_vector: Option<Vector3<f32>>,
    pub pole_target: Option<BoneId>,
    pub twist: f32,
    pub enabled: bool,
    pub weight: f32,
}

impl Default for IkConstraintData {
    fn default() -> Self {
        Self {
            chain_length: 2,
            target_bone: 0,
            effector_bone: 0,
            pole_vector: None,
            pole_target: None,
            twist: 0.0,
            enabled: true,
            weight: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AimConstraintData {
    pub source_bone: BoneId,
    pub target_bone: BoneId,
    pub aim_axis: Vector3<f32>,
    pub up_axis: Vector3<f32>,
    pub up_target: Option<BoneId>,
    pub enabled: bool,
    pub weight: f32,
}

impl Default for AimConstraintData {
    fn default() -> Self {
        Self {
            source_bone: 0,
            target_bone: 0,
            aim_axis: Vector3::new(0.0, 0.0, 1.0),
            up_axis: Vector3::new(0.0, 1.0, 0.0),
            up_target: None,
            enabled: true,
            weight: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParentConstraintData {
    pub constrained_bone: BoneId,
    pub sources: Vec<(BoneId, f32)>,
    pub affect_translation: bool,
    pub affect_rotation: bool,
    pub enabled: bool,
    pub weight: f32,
}

impl Default for ParentConstraintData {
    fn default() -> Self {
        Self {
            constrained_bone: 0,
            sources: Vec::new(),
            affect_translation: true,
            affect_rotation: true,
            enabled: true,
            weight: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PositionConstraintData {
    pub constrained_bone: BoneId,
    pub target_bone: BoneId,
    pub offset: Vector3<f32>,
    pub affect_axes: [bool; 3],
    pub enabled: bool,
    pub weight: f32,
}

impl Default for PositionConstraintData {
    fn default() -> Self {
        Self {
            constrained_bone: 0,
            target_bone: 0,
            offset: Vector3::new(0.0, 0.0, 0.0),
            affect_axes: [true, true, true],
            enabled: true,
            weight: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RotationConstraintData {
    pub constrained_bone: BoneId,
    pub target_bone: BoneId,
    pub offset: Quaternion<f32>,
    pub affect_axes: [bool; 3],
    pub enabled: bool,
    pub weight: f32,
}

impl Default for RotationConstraintData {
    fn default() -> Self {
        Self {
            constrained_bone: 0,
            target_bone: 0,
            offset: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            affect_axes: [true, true, true],
            enabled: true,
            weight: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScaleConstraintData {
    pub constrained_bone: BoneId,
    pub target_bone: BoneId,
    pub offset: Vector3<f32>,
    pub affect_axes: [bool; 3],
    pub enabled: bool,
    pub weight: f32,
}

impl Default for ScaleConstraintData {
    fn default() -> Self {
        Self {
            constrained_bone: 0,
            target_bone: 0,
            offset: Vector3::new(1.0, 1.0, 1.0),
            affect_axes: [true, true, true],
            enabled: true,
            weight: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ConstraintType {
    Ik(IkConstraintData),
    Aim(AimConstraintData),
    Parent(ParentConstraintData),
    Position(PositionConstraintData),
    Rotation(RotationConstraintData),
    Scale(ScaleConstraintData),
}

impl ConstraintType {
    pub fn is_enabled(&self) -> bool {
        match self {
            Self::Ik(d) => d.enabled,
            Self::Aim(d) => d.enabled,
            Self::Parent(d) => d.enabled,
            Self::Position(d) => d.enabled,
            Self::Rotation(d) => d.enabled,
            Self::Scale(d) => d.enabled,
        }
    }

    pub fn constrained_bone_id(&self) -> BoneId {
        match self {
            Self::Ik(d) => d.effector_bone,
            Self::Aim(d) => d.source_bone,
            Self::Parent(d) => d.constrained_bone,
            Self::Position(d) => d.constrained_bone,
            Self::Rotation(d) => d.constrained_bone,
            Self::Scale(d) => d.constrained_bone,
        }
    }

    pub fn default_priority(&self) -> u32 {
        match self {
            Self::Ik(_) => PRIORITY_IK,
            Self::Aim(_) => PRIORITY_AIM,
            Self::Parent(_) => PRIORITY_PARENT,
            Self::Position(_) => PRIORITY_POSITION,
            Self::Rotation(_) => PRIORITY_ROTATION,
            Self::Scale(_) => PRIORITY_SCALE,
        }
    }
}
