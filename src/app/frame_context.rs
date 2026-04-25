use std::rc::Rc;

use anyhow::Result;
use cgmath::{Matrix4, Vector3};

use crate::asset::AssetStorage;
use crate::ecs::resource::billboard::BillboardData;
use crate::ecs::resource::gizmo::{
    BoneGizmoData, GridGizmoData, LightGizmoData, TransformGizmoData,
};
use crate::ecs::resource::DebugViewState;
use crate::ecs::resource::{Camera, GridMeshData, LightState};
use crate::ecs::world::{ResMut, ResRef, World};
use crate::render::{BillboardBackend, RenderBackend};
use crate::vulkanr::command::RRCommandPool;
use crate::vulkanr::device::RRDevice;
use crate::vulkanr::renderer::onion_skin_buffers::OnionSkinGpuState;
use crate::vulkanr::resource::graphics_resource::GraphicsResources;
use crate::vulkanr::resource::GpuBufferRegistry;
use crate::vulkanr::vulkan::Instance;
use crate::vulkanr::VulkanBackend;
use thyllore_vulkan_core::resource::raytracing_data::RayTracingData;

pub struct FrameContext<'a> {
    pub instance: &'a Instance,
    pub device: &'a RRDevice,
    pub command_pool: Rc<RRCommandPool>,

    pub time: f32,
    pub delta_time: f32,
    pub image_index: usize,
    pub swapchain_extent: (u32, u32),

    pub graphics: &'a mut GraphicsResources,
    pub raytracing: &'a mut RayTracingData,
    pub buffer_registry: &'a mut GpuBufferRegistry,

    pub world: &'a mut World,
    pub assets: &'a mut AssetStorage,

    pub onion_skin_gpu: &'a mut Option<OnionSkinGpuState>,
}

impl<'a> FrameContext<'a> {
    pub fn create_backend(&mut self) -> VulkanBackend<'_> {
        VulkanBackend::new(
            self.instance,
            self.device,
            self.command_pool.clone(),
            self.graphics,
            self.raytracing,
            self.buffer_registry,
        )
    }

    pub fn camera(&self) -> ResRef<Camera> {
        self.world.resource::<Camera>()
    }

    pub fn camera_mut(&self) -> ResMut<Camera> {
        self.world.resource_mut::<Camera>()
    }

    pub fn light_state(&self) -> ResRef<LightState> {
        self.world.resource::<LightState>()
    }

    pub fn light_state_mut(&self) -> ResMut<LightState> {
        self.world.resource_mut::<LightState>()
    }

    pub fn debug_view_state(&self) -> ResRef<DebugViewState> {
        self.world.resource::<DebugViewState>()
    }

    pub fn camera_position(&self) -> Vector3<f32> {
        use crate::ecs::systems::camera_systems::compute_camera_position;
        compute_camera_position(&self.camera())
    }

    pub fn camera_direction(&self) -> Vector3<f32> {
        use crate::ecs::systems::camera_systems::compute_camera_direction;
        compute_camera_direction(&self.camera())
    }

    pub fn camera_up(&self) -> Vector3<f32> {
        use crate::ecs::systems::camera_systems::compute_camera_up;
        compute_camera_up(&self.camera())
    }

    pub fn light_position(&self) -> Vector3<f32> {
        self.light_state().light_position
    }

    pub fn grid_mesh(&self) -> ResRef<GridMeshData> {
        self.world.resource::<GridMeshData>()
    }

    pub fn grid_mesh_mut(&self) -> ResMut<GridMeshData> {
        self.world.resource_mut::<GridMeshData>()
    }

    pub fn gizmo(&self) -> ResRef<GridGizmoData> {
        self.world.resource::<GridGizmoData>()
    }

    pub fn gizmo_mut(&self) -> ResMut<GridGizmoData> {
        self.world.resource_mut::<GridGizmoData>()
    }

    pub fn light_gizmo(&self) -> ResRef<LightGizmoData> {
        self.world.resource::<LightGizmoData>()
    }

    pub fn light_gizmo_mut(&self) -> ResMut<LightGizmoData> {
        self.world.resource_mut::<LightGizmoData>()
    }

    pub fn bone_gizmo(&self) -> ResRef<BoneGizmoData> {
        self.world.resource::<BoneGizmoData>()
    }

    pub fn bone_gizmo_mut(&self) -> ResMut<BoneGizmoData> {
        self.world.resource_mut::<BoneGizmoData>()
    }

    pub fn billboard(&self) -> ResRef<BillboardData> {
        self.world.resource::<BillboardData>()
    }

    pub fn billboard_mut(&self) -> ResMut<BillboardData> {
        self.world.resource_mut::<BillboardData>()
    }

    pub fn transform_gizmo(&self) -> ResRef<TransformGizmoData> {
        self.world.resource::<TransformGizmoData>()
    }

    pub fn transform_gizmo_mut(&self) -> ResMut<TransformGizmoData> {
        self.world.resource_mut::<TransformGizmoData>()
    }

    pub unsafe fn update_billboard_ubo_internal(
        &mut self,
        model: Matrix4<f32>,
        view: Matrix4<f32>,
        proj: Matrix4<f32>,
        image_index: usize,
    ) -> Result<()> {
        let mut billboard = self.world.resource_mut::<BillboardData>();
        let mut backend = VulkanBackend::new(
            self.instance,
            self.device,
            self.command_pool.clone(),
            self.graphics,
            self.raytracing,
            self.buffer_registry,
        );
        backend.update_billboard_ubo(&mut billboard, model, view, proj, image_index)
    }
}
