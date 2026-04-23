use cgmath::{Matrix4, Vector3};

use crate::DynamicMesh;

#[repr(C)]
#[derive(Clone, Debug, Copy, Default)]
pub struct BillboardVertex {
    pub pos: [f32; 3],
    pub tex_coord: [f32; 2],
}

pub type BillboardMesh = DynamicMesh<BillboardVertex>;

#[derive(Clone, Debug)]
pub struct BillboardTransform {
    pub position: Vector3<f32>,
    pub model_matrix: Matrix4<f32>,
}
