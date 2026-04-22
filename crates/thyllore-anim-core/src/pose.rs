use cgmath::{Quaternion, Vector3};

use crate::SkeletonId;

#[derive(Clone, Debug)]
pub struct BoneLocalPose {
    pub translation: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale: Vector3<f32>,
}

impl Default for BoneLocalPose {
    fn default() -> Self {
        Self {
            translation: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SkeletonPose {
    pub skeleton_id: SkeletonId,
    pub bone_poses: Vec<BoneLocalPose>,
}
