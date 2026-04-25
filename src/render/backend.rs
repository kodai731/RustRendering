use anyhow::Result;
use cgmath::Matrix4;

use crate::ecs::resource::billboard::BillboardData;

pub use thyllore_render_core::backend::RenderBackend;
pub use thyllore_render_core::{BufferMemoryType, MeshId};

pub trait BillboardBackend {
    unsafe fn create_billboard_buffers(&mut self, billboard: &mut BillboardData) -> Result<()>;

    unsafe fn update_billboard_ubo(
        &mut self,
        billboard: &mut BillboardData,
        model: Matrix4<f32>,
        view: Matrix4<f32>,
        proj: Matrix4<f32>,
        image_index: usize,
    ) -> Result<()>;
}
