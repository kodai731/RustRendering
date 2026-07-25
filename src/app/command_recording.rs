use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::App;
use crate::ecs::resource::GpuPassTimings;
use crate::vulkanr::context::FrameSync;
use crate::vulkanr::renderer::deferred;

impl App {
    pub unsafe fn record_command_buffer(
        &mut self,
        image_index: usize,
        draw_data: &imgui::DrawData,
        frame_slot: usize,
    ) -> Result<()> {
        let command_buffer = self
            .resource::<crate::vulkanr::context::CommandState>()
            .buffers
            .command_buffers[image_index];

        self.rrdevice
            .device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.rrdevice
            .device
            .begin_command_buffer(command_buffer, &begin_info)?;

        // Collect GPU timestamp profiler results from previous frame and write to ECS resource.
        // Derive next_frame from the current GpuPassTimings resource (frame + 1), defaulting to 1
        // if the resource is absent. The borrow must be dropped before insert_resource.
        let next_frame = self.data.ecs_world.get_resource::<GpuPassTimings>()
            .map(|t| t.frame + 1)
            .unwrap_or(1);

        if let Some(passes) = self.gpu_timestamp_profiler.collect(&self.rrdevice.device, image_index) {
            self.data.ecs_world.insert_resource(GpuPassTimings {
                frame: next_frame,
                passes,
            });
        }

        self.gpu_timestamp_profiler.begin_frame(&self.rrdevice.device, command_buffer, image_index);

        let use_gbuffer =
            self.data.raytracing.is_available() && self.data.viewport.offscreen.is_some();

        if use_gbuffer {
            self.gpu_timestamp_profiler.begin_scope(
                &self.rrdevice.device,
                command_buffer,
                image_index,
                "gbuffer".to_string(),
            );
            deferred::record_gbuffer_pass(self, command_buffer, image_index)?;
            self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);

            self.gpu_timestamp_profiler.begin_scope(
                &self.rrdevice.device,
                command_buffer,
                image_index,
                "object_id_copy".to_string(),
            );
            self.record_object_id_copy(command_buffer);
            self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);

            if self.data.raytracing.has_valid_tlas() {
                self.gpu_timestamp_profiler.begin_scope(
                    &self.rrdevice.device,
                    command_buffer,
                    image_index,
                    "ray_query".to_string(),
                );
                deferred::record_ray_query_pass(self, command_buffer)?;
                self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);
            } else {
                self.prepare_empty_shadow_mask(command_buffer);
            }

            let has_hdr_pipeline = self.data.viewport.hdr_buffer.is_some()
                && self.data.raytracing.tonemap_pipeline.is_some();

            if has_hdr_pipeline {
                self.gpu_timestamp_profiler.begin_scope(
                    &self.rrdevice.device,
                    command_buffer,
                    image_index,
                    "composite_hdr".to_string(),
                );
                deferred::record_composite_to_hdr(self, command_buffer)?;
                self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);

                self.gpu_timestamp_profiler.begin_scope(
                    &self.rrdevice.device,
                    command_buffer,
                    image_index,
                    "onion_skin".to_string(),
                );
                deferred::record_onion_skin_pass(self, command_buffer, image_index)?;
                self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);

                self.gpu_timestamp_profiler.begin_scope(
                    &self.rrdevice.device,
                    command_buffer,
                    image_index,
                    "flame".to_string(),
                );
                deferred::record_flame_passes(self, command_buffer, image_index)?;
                self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);

                self.gpu_timestamp_profiler.begin_scope(
                    &self.rrdevice.device,
                    command_buffer,
                    image_index,
                    "bloom".to_string(),
                );
                deferred::record_bloom(self, command_buffer)?;
                self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);

                self.gpu_timestamp_profiler.begin_scope(
                    &self.rrdevice.device,
                    command_buffer,
                    image_index,
                    "dof".to_string(),
                );
                deferred::record_dof(self, command_buffer)?;
                self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);

                self.gpu_timestamp_profiler.begin_scope(
                    &self.rrdevice.device,
                    command_buffer,
                    image_index,
                    "auto_exposure".to_string(),
                );
                deferred::record_auto_exposure(self, command_buffer, self.resource::<FrameSync>().current_frame)?;
                self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);

                self.gpu_timestamp_profiler.begin_scope(
                    &self.rrdevice.device,
                    command_buffer,
                    image_index,
                    "tonemap".to_string(),
                );
                deferred::record_tonemap_to_offscreen(self, command_buffer, image_index)?;
                self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);

                self.gpu_timestamp_profiler.begin_scope(
                    &self.rrdevice.device,
                    command_buffer,
                    image_index,
                    "onion_composite".to_string(),
                );
                deferred::record_onion_skin_composite(self, command_buffer)?;
                self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);
            } else {
                self.gpu_timestamp_profiler.begin_scope(
                    &self.rrdevice.device,
                    command_buffer,
                    image_index,
                    "composite_offscreen".to_string(),
                );
                deferred::record_composite_to_offscreen(self, command_buffer, image_index)?;
                self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);
            }

            self.gpu_timestamp_profiler.begin_scope(
                &self.rrdevice.device,
                command_buffer,
                image_index,
                "imgui".to_string(),
            );
            self.begin_main_render_pass(command_buffer, image_index);
            self.record_imgui_rendering(command_buffer, draw_data)?;
            self.rrdevice.device.cmd_end_render_pass(command_buffer);
            self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);
        } else {
            if let Some(ref offscreen) = self.data.viewport.offscreen {
                self.begin_offscreen_render_pass(command_buffer, offscreen);
                self.gpu_timestamp_profiler.begin_scope(
                    &self.rrdevice.device,
                    command_buffer,
                    image_index,
                    "offscreen_3d".to_string(),
                );
                self.record_3d_rendering_to_offscreen(command_buffer, image_index, offscreen)?;
                self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);
                self.rrdevice.device.cmd_end_render_pass(command_buffer);
            }

            self.gpu_timestamp_profiler.begin_scope(
                &self.rrdevice.device,
                command_buffer,
                image_index,
                "imgui".to_string(),
            );
            self.begin_main_render_pass(command_buffer, image_index);
            self.record_imgui_rendering(command_buffer, draw_data)?;
            self.rrdevice.device.cmd_end_render_pass(command_buffer);
            self.gpu_timestamp_profiler.end_scope(&self.rrdevice.device, command_buffer, image_index);
        }

        self.rrdevice.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    unsafe fn prepare_empty_shadow_mask(&self, command_buffer: vk::CommandBuffer) {
        let Some(ref gbuffer) = self.data.raytracing.gbuffer else {
            return;
        };

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let to_transfer = vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(gbuffer.shadow_mask_image)
            .subresource_range(subresource_range)
            .build();

        self.rrdevice.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[to_transfer],
        );

        let clear_value = vk::ClearColorValue {
            float32: [1.0, 1.0, 1.0, 1.0],
        };
        self.rrdevice.device.cmd_clear_color_image(
            command_buffer,
            gbuffer.shadow_mask_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &clear_value,
            &[subresource_range],
        );

        let to_shader_read = vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(gbuffer.shadow_mask_image)
            .subresource_range(subresource_range)
            .build();

        self.rrdevice.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[to_shader_read],
        );
    }

    unsafe fn record_object_id_copy(&mut self, command_buffer: vk::CommandBuffer) {
        use crate::ecs::resource::ObjectIdReadback;

        if !self.data.ecs_world.contains_resource::<ObjectIdReadback>() {
            return;
        }

        let readback = self.data.ecs_world.resource::<ObjectIdReadback>();
        if readback.copy_in_flight {
            drop(readback);
            return;
        }
        let Some((px, py)) = readback.pending_pixel else {
            drop(readback);
            return;
        };
        drop(readback);

        let Some(ref gbuffer) = self.data.raytracing.gbuffer else {
            return;
        };

        let object_id_image = gbuffer.object_id_image;
        let staging_buffer = gbuffer.readback_staging_buffer;

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let barrier_to_transfer = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(object_id_image)
            .subresource_range(subresource_range)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

        self.rrdevice.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier_to_transfer.build()],
        );

        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D {
                x: px as i32,
                y: py as i32,
                z: 0,
            })
            .image_extent(vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            });

        self.rrdevice.device.cmd_copy_image_to_buffer(
            command_buffer,
            object_id_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging_buffer,
            &[region.build()],
        );

        let barrier_to_shader = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(object_id_image)
            .subresource_range(subresource_range)
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        self.rrdevice.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier_to_shader.build()],
        );

        let mut readback = self.data.ecs_world.resource_mut::<ObjectIdReadback>();
        readback.pending_pixel = None;
        readback.copy_in_flight = true;
    }
}
