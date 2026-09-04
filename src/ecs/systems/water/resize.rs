use crate::app::App;
use crate::vulkanr::context::{CommandState, RenderTargets};
use crate::vulkanr::vulkan::{vk, Handle};
use anyhow::Result;

impl App {
    pub(crate) unsafe fn recreate_water_on_resize(&mut self) -> Result<()> {
        let depth_view = {
            let rt = self.resource::<RenderTargets>();
            rt.render.gbuffer_depth_image_view
        };
        let hdr_view = match &self.data.viewport.hdr_buffer {
            Some(hdr) => hdr.color_image_view,
            None => return Ok(()),
        };
        if depth_view == vk::ImageView::null() {
            return Ok(());
        }

        let width = self.data.viewport.width;
        let height = self.data.viewport.height;

        // Destroy old buffer if it exists
        if let Some(mut old_buffer) = self.data.viewport.water_buffer.take() {
            old_buffer.destroy(&self.rrdevice.device);
        }

        // Recreate with new dimensions
        let command_pool = self.resource::<CommandState>().pool.command_pool;
        let water_buffer = thyllore_vulkan_core::resource::WaterBuffer::new(
            &self.instance,
            &self.rrdevice,
            command_pool,
            width,
            height,
            hdr_view,
            depth_view,
        )?;
        let (scene_color_view, scene_color_sampler) = water_buffer.scene_color_binding();
        self.data.viewport.water_buffer = Some(water_buffer);

        if let Some(descriptor) = &self.data.raytracing.water_descriptor {
            descriptor.update_scene_color(&self.rrdevice, scene_color_view, scene_color_sampler)?;
        }

        if let Some(trace_descriptor) = &self.data.raytracing.water_trace_descriptor {
            if let Some(accel) = self.data.raytracing.acceleration_structure.as_ref() {
                if let Some(tlas) = accel.tlas.acceleration_structure {
                    if let Some(hit_table) = accel.hit_shading_table.as_ref() {
                        if let Some(water_buffer) = self.data.viewport.water_buffer.as_ref() {
                            if let Some(water_ubo) = self.data.raytracing.water_ubo.as_ref() {
                                trace_descriptor.write_all(
                                    &self.rrdevice,
                                    tlas,
                                    water_buffer.trace_image_view,
                                    water_ubo,
                                    hit_table.buffer,
                                )?;
                            }
                        }
                    }
                }
            }
        }

        self.update_water_caustic_descriptor()?;

        Ok(())
    }

    /// The caustic descriptor binds the accumulation, G-buffer position and HDR views
    /// directly, so every resize that recreates them leaves it stale.
    pub(crate) unsafe fn update_water_caustic_descriptor(&mut self) -> Result<()> {
        let (Some(caustic_accum_view), Some(hdr_color_view)) = (
            self.data
                .viewport
                .water_buffer
                .as_ref()
                .map(|water_buffer| water_buffer.caustic_accum_view),
            self.data
                .viewport
                .hdr_buffer
                .as_ref()
                .map(|hdr_buffer| hdr_buffer.color_image_view),
        ) else {
            return Ok(());
        };

        let rrdevice = &self.rrdevice;
        let raytracing = &mut self.data.raytracing;
        let tlas = raytracing
            .acceleration_structure
            .as_ref()
            .and_then(|accel| accel.tlas.acceleration_structure);
        let (Some(position_image_view), Some(scene_buffer), Some(water_ubo)) = (
            raytracing
                .gbuffer
                .as_ref()
                .map(|gbuffer| gbuffer.position_image_view),
            raytracing.scene_uniform_buffer,
            raytracing.water_ubo.as_ref().map(|ubo| ubo.handle()),
        ) else {
            return Ok(());
        };

        let Some(descriptor) = raytracing.water_caustic_descriptor.as_mut() else {
            return Ok(());
        };

        descriptor.allocate_and_update(
            rrdevice,
            caustic_accum_view,
            position_image_view,
            tlas,
            scene_buffer,
            water_ubo,
            hdr_color_view,
        )
    }
}
