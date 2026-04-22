use anyhow::Result;
use cgmath::{Matrix4, Vector3};

use crate::ecs::component::LineMesh;
use crate::ecs::resource::billboard::BillboardData;
use crate::ecs::resource::DistanceAttenuation;
use crate::ecs::resource::ProjectionData;

pub use thyllore_render_core::{BufferMemoryType, MeshId};

pub trait RenderBackend {
    unsafe fn upload_mesh_vertices(&mut self, mesh_id: MeshId) -> Result<()>;

    unsafe fn update_acceleration_structure(&mut self, mesh_ids: &[MeshId]) -> Result<()>;

    unsafe fn rebuild_tlas(&mut self) -> Result<()>;

    unsafe fn create_gizmo_buffers(
        &mut self,
        mesh: &mut LineMesh,
        memory_type: BufferMemoryType,
    ) -> Result<()>;

    unsafe fn update_gizmo_vertex_buffer(&self, mesh: &LineMesh) -> Result<()>;

    unsafe fn destroy_gizmo_buffers(&mut self, mesh: &mut LineMesh);

    unsafe fn update_or_create_line_buffers(&mut self, mesh: &mut LineMesh) -> Result<()>;

    unsafe fn destroy_line_buffers(&mut self, mesh: &mut LineMesh);

    unsafe fn create_billboard_buffers(&mut self, billboard: &mut BillboardData) -> Result<()>;

    unsafe fn update_frame_ubo(
        &mut self,
        proj_data: &ProjectionData,
        camera_pos: Vector3<f32>,
        light_pos: Vector3<f32>,
        light_color: Vector3<f32>,
        image_index: usize,
    ) -> Result<()>;

    unsafe fn update_object_ubo(
        &mut self,
        model_matrix: Matrix4<f32>,
        object_index: usize,
        image_index: usize,
    ) -> Result<()>;

    unsafe fn update_scene_uniform(
        &mut self,
        view: Matrix4<f32>,
        proj: Matrix4<f32>,
        light_pos: Vector3<f32>,
        light_color: Vector3<f32>,
        debug_mode: i32,
        shadow_strength: f32,
        distance_attenuation: DistanceAttenuation,
        exposure_value: f32,
    ) -> Result<()>;

    unsafe fn update_billboard_ubo(
        &mut self,
        billboard: &mut BillboardData,
        model: Matrix4<f32>,
        view: Matrix4<f32>,
        proj: Matrix4<f32>,
        image_index: usize,
    ) -> Result<()>;
}
