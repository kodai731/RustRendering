use std::collections::HashMap;

use cgmath::Matrix4;

#[derive(Default)]
pub struct PoseApplyCache {
    pub skinned_cache: HashMap<usize, Vec<Matrix4<f32>>>,
    pub node_cache: HashMap<usize, (Matrix4<f32>, f32)>,
}
