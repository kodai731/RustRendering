use anyhow::Result;
use cgmath::Matrix4;
use vulkanalia::prelude::v1_0::*;

use crate::ecs::resource::billboard::BillboardData;
use crate::render::BillboardBackend;
use crate::vulkanr::image::RRImage;

pub use thyllore_vulkan_core::backend::VulkanBackend;

impl<'a> BillboardBackend for VulkanBackend<'a> {
    unsafe fn create_billboard_buffers(&mut self, billboard: &mut BillboardData) -> Result<()> {
        billboard.mesh.vertex_buffer_handles[0] =
            self.buffer_registry.create_host_visible_vertex_buffer(
                self.instance,
                self.device,
                &billboard.mesh.vertices,
                256,
            )?;
        billboard.mesh.last_written_slot = 0;

        billboard.mesh.index_buffer_handles[0] =
            self.buffer_registry.create_host_visible_index_buffer(
                self.instance,
                self.device,
                &billboard.mesh.indices,
            )?;

        let white_pixel: [u8; 4] = [255, 255, 255, 255];
        billboard.render_state.texture = Some(
            RRImage::new_from_pixels(
                self.instance,
                self.device,
                &self.command_pool,
                &white_pixel,
                1,
                1,
            )
            .map_err(|e| anyhow::anyhow!("Failed to create billboard texture: {}", e))?,
        );

        Ok(())
    }

    unsafe fn update_billboard_ubo(
        &mut self,
        billboard: &mut BillboardData,
        model: Matrix4<f32>,
        view: Matrix4<f32>,
        proj: Matrix4<f32>,
        image_index: usize,
    ) -> Result<()> {
        use crate::vulkanr::data::UniformBufferObject;

        for i in 0..billboard.render_state.descriptor_set.rrdata.len() {
            let rrdata = &mut billboard.render_state.descriptor_set.rrdata[i];

            let ubo = UniformBufferObject { model, view, proj };
            rrdata.rruniform_buffers[image_index].update(self.device, &ubo)?;
        }

        Ok(())
    }
}
