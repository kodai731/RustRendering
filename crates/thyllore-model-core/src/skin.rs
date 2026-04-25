use cgmath::{Vector3, Vector4};

use crate::SkeletonId;

#[derive(Clone, Debug)]
pub struct SkinData {
    pub skeleton_id: SkeletonId,
    pub bone_indices: Vec<Vector4<u32>>,
    pub bone_weights: Vec<Vector4<f32>>,
    pub base_positions: Vec<Vector3<f32>>,
    pub base_normals: Vec<Vector3<f32>>,
}

impl Default for SkinData {
    fn default() -> Self {
        Self {
            skeleton_id: 0,
            bone_indices: Vec::new(),
            bone_weights: Vec::new(),
            base_positions: Vec::new(),
            base_normals: Vec::new(),
        }
    }
}
