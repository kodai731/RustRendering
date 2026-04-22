use crate::core::device::RRDevice;
use crate::vulkan::*;
use cgmath::{Matrix4, SquareMatrix, Vector3};
use thyllore_render_core::ObjectUBO;

pub use crate::descriptor::{
    FrameDescriptorSet, Material, MaterialId, MaterialManager, ObjectDescriptorSet, ObjectId,
};
pub use crate::resource::MeshBuffer;

#[derive(Clone, Debug)]
pub struct NodeData {
    pub index: usize,
    pub name: String,
    pub parent_index: Option<usize>,
    pub local_transform: Matrix4<f32>,
    pub global_transform: Matrix4<f32>,
}

impl Default for NodeData {
    fn default() -> Self {
        Self {
            index: 0,
            name: String::new(),
            parent_index: None,
            local_transform: Matrix4::identity(),
            global_transform: Matrix4::identity(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GraphicsResources {
    pub frame_set: FrameDescriptorSet,
    pub materials: MaterialManager,
    pub objects: ObjectDescriptorSet,
    pub meshes: Vec<MeshBuffer>,
    pub mesh_material_ids: Vec<MaterialId>,
}

impl GraphicsResources {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        swapchain_image_count: usize,
        max_materials: u32,
        max_objects: usize,
    ) -> anyhow::Result<Self> {
        let frame_set = FrameDescriptorSet::new(instance, rrdevice, swapchain_image_count)?;
        let materials = MaterialManager::new(rrdevice, max_materials)?;
        let objects =
            ObjectDescriptorSet::new(instance, rrdevice, swapchain_image_count, max_objects)?;

        Ok(Self {
            frame_set,
            materials,
            objects,
            meshes: Vec::new(),
            mesh_material_ids: Vec::new(),
        })
    }

    pub unsafe fn update_objects(
        &self,
        rrdevice: &RRDevice,
        image_index: usize,
        model: Matrix4<f32>,
    ) -> anyhow::Result<()> {
        for mesh in &self.meshes {
            let object_ubo = ObjectUBO { model };
            self.objects
                .update(rrdevice, image_index, mesh.object_index, &object_ubo)?;
        }
        Ok(())
    }

    pub fn get_material_id(&self, mesh_index: usize) -> Option<MaterialId> {
        self.mesh_material_ids.get(mesh_index).copied()
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    pub fn get_layouts(&self) -> [vk::DescriptorSetLayout; 3] {
        [
            self.frame_set.layout,
            self.materials.layout,
            self.objects.layout,
        ]
    }

    pub fn get_layouts_without_material(&self) -> [vk::DescriptorSetLayout; 2] {
        [self.frame_set.layout, self.objects.layout]
    }

    pub fn calculate_model_bounds(&self) -> Option<(Vector3<f32>, Vector3<f32>, Vector3<f32>)> {
        if self.meshes.is_empty() {
            return None;
        }

        let mut min = Vector3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vector3::new(f32::MIN, f32::MIN, f32::MIN);
        let mut has_vertices = false;

        for mesh in &self.meshes {
            for vertex in &mesh.vertex_data.vertices {
                has_vertices = true;
                min.x = min.x.min(vertex.pos.x);
                min.y = min.y.min(vertex.pos.y);
                min.z = min.z.min(vertex.pos.z);
                max.x = max.x.max(vertex.pos.x);
                max.y = max.y.max(vertex.pos.y);
                max.z = max.z.max(vertex.pos.z);
            }
        }

        if !has_vertices {
            return None;
        }

        let center = Vector3::new(
            (min.x + max.x) * 0.5,
            (min.y + max.y) * 0.5,
            (min.z + max.z) * 0.5,
        );

        log!("Model bounds: min=({:.2}, {:.2}, {:.2}), max=({:.2}, {:.2}, {:.2}), center=({:.2}, {:.2}, {:.2})",
            min.x, min.y, min.z, max.x, max.y, max.z, center.x, center.y, center.z);

        Some((min, max, center))
    }

    pub unsafe fn destroy(&mut self, rrdevice: &RRDevice) {
        for mesh in &mut self.meshes {
            mesh.destroy(rrdevice);
        }
        self.meshes.clear();
        self.mesh_material_ids.clear();

        self.frame_set.destroy(&rrdevice.device);
        self.materials.destroy(&rrdevice.device);
        self.objects.destroy(&rrdevice.device);
    }

    pub unsafe fn clear_meshes(&mut self, rrdevice: &RRDevice) {
        for mesh in &mut self.meshes {
            mesh.destroy(rrdevice);
        }
        self.meshes.clear();
        self.mesh_material_ids.clear();
    }

    pub fn create_pipeline_key(
        &self,
        vertex_shader: &str,
        fragment_shader: &str,
        topology: vk::PrimitiveTopology,
        polygon_mode: vk::PolygonMode,
        cull_mode: vk::CullModeFlags,
        depth_test_enable: bool,
        blend_enable: bool,
        render_pass: vk::RenderPass,
    ) -> crate::pipeline::PipelineKey {
        crate::pipeline::PipelineKey::new(
            vertex_shader,
            fragment_shader,
            topology,
            polygon_mode,
            cull_mode,
            depth_test_enable,
            depth_test_enable,
            blend_enable,
            vk::SampleCountFlags::_1,
            1,
            render_pass,
        )
    }
}
