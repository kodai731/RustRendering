use crate::vulkanr::resource::graphics_resource::{
    FrameDescriptorSet, MaterialId, MaterialManager, MeshBuffer, NodeData, ObjectDescriptorSet,
};

#[derive(Clone, Debug, Default)]
pub struct GpuDescriptors {
    pub frame_set: FrameDescriptorSet,
    pub objects: ObjectDescriptorSet,
}

impl GpuDescriptors {
    pub fn new(frame_set: FrameDescriptorSet, objects: ObjectDescriptorSet) -> Self {
        Self { frame_set, objects }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MaterialRegistry {
    pub materials: MaterialManager,
    pub mesh_material_ids: Vec<MaterialId>,
}

impl MaterialRegistry {
    pub fn new(materials: MaterialManager) -> Self {
        Self {
            materials,
            mesh_material_ids: Vec::new(),
        }
    }

    pub fn get_material_id(&self, mesh_index: usize) -> Option<MaterialId> {
        self.mesh_material_ids.get(mesh_index).copied()
    }
}

pub use thyllore_anim_core::AnimationType;

#[derive(Clone, Debug)]
pub struct ModelState {
    pub has_skinned_meshes: bool,
    pub model_path: String,
    pub load_status: String,
    pub flame_preset_index: usize,
    pub texture_fit_path: String,
    pub texture_fit_blend: f32,
    pub texture_fit_groups: [bool; 4],
}

impl Default for ModelState {
    fn default() -> Self {
        Self {
            has_skinned_meshes: false,
            model_path: String::new(),
            load_status: String::from("No model loaded"),
            flame_preset_index: 0,
            texture_fit_path: String::new(),
            texture_fit_blend: 1.0,
            texture_fit_groups: [true; 4],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MeshAssets {
    pub meshes: Vec<MeshBuffer>,
}

impl MeshAssets {
    pub fn new() -> Self {
        Self { meshes: Vec::new() }
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct NodeAssets {
    pub nodes: Vec<NodeData>,
}

impl NodeAssets {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }
}
