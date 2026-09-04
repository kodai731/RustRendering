use crate::app::App;
use crate::vulkanr::command::RRCommandBuffer;
use crate::vulkanr::context::{CommandState, RenderTargets, SwapchainState};
use crate::vulkanr::render::framebuffer::{create_color_objects, create_framebuffers};
use crate::vulkanr::render::pass::create_depth_objects;
use crate::vulkanr::swapchain::RRSwapchain;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use winit::window::Window;

impl App {
    pub unsafe fn destroy(&mut self) {
        log!("Destroying application resources...");

        let _ = self.rrdevice.device.device_wait_idle();

        if let Some(sampler) = self.data.raytracing.gbuffer_sampler {
            self.rrdevice.device.destroy_sampler(sampler, None);
            log!("Destroyed G-Buffer sampler");
        }

        if let Some(onion_skin_pass) = self.data.raytracing.onion_skin_pass.take() {
            onion_skin_pass.destroy(&self.rrdevice.device);
        }

        if let Some(gbuffer_pipeline) = self.data.raytracing.gbuffer_pipeline.take() {
            gbuffer_pipeline.destroy(&self.rrdevice.device);
            log!("Destroyed G-Buffer pipeline");
        }

        if let Some(mut dof_descriptor) = self.data.raytracing.dof_descriptor.take() {
            dof_descriptor.destroy(&self.rrdevice.device);
        }

        if let Some(dof_pipeline) = self.data.raytracing.dof_pipeline.take() {
            dof_pipeline.destroy(&self.rrdevice.device);
        }

        if let Some(mut desc) = self
            .data
            .raytracing
            .auto_exposure_histogram_descriptor
            .take()
        {
            desc.destroy(&self.rrdevice.device);
        }

        if let Some(mut desc) = self.data.raytracing.auto_exposure_average_descriptor.take() {
            desc.destroy(&self.rrdevice.device);
        }

        if let Some(pipeline) = self.data.raytracing.auto_exposure_histogram_pipeline.take() {
            pipeline.destroy(&self.rrdevice.device);
        }

        if let Some(pipeline) = self.data.raytracing.auto_exposure_average_pipeline.take() {
            pipeline.destroy(&self.rrdevice.device);
        }

        if let Some(mut bloom_descriptors) = self.data.raytracing.bloom_descriptors.take() {
            bloom_descriptors.destroy(&self.rrdevice.device);
        }

        if let Some(bloom_downsample_pipeline) =
            self.data.raytracing.bloom_downsample_pipeline.take()
        {
            bloom_downsample_pipeline.destroy(&self.rrdevice.device);
        }

        if let Some(bloom_upsample_pipeline) = self.data.raytracing.bloom_upsample_pipeline.take() {
            bloom_upsample_pipeline.destroy(&self.rrdevice.device);
        }

        if let Some(mut tonemap_descriptor) = self.data.raytracing.tonemap_descriptor.take() {
            tonemap_descriptor.destroy(&self.rrdevice.device);
        }

        if let Some(tonemap_pipeline) = self.data.raytracing.tonemap_pipeline.take() {
            tonemap_pipeline.destroy(&self.rrdevice.device);
        }

        if let Some(mut composite_descriptor) = self.data.raytracing.composite_descriptor.take() {
            composite_descriptor.destroy(&self.rrdevice.device);
        }

        if let Some(composite_pipeline) = self.data.raytracing.composite_pipeline.take() {
            composite_pipeline.destroy(&self.rrdevice.device);
        }

        if let Some(mut flame_descriptor) = self.data.raytracing.flame_descriptor.take() {
            flame_descriptor.destroy(&self.rrdevice.device);
        }

        if let Some(flame_shading_pipeline) = self.data.raytracing.flame_shading_pipeline.take() {
            flame_shading_pipeline.destroy(&self.rrdevice.device);
        }

        if let Some(mut flame_ubo) = self.data.raytracing.flame_ubo.take() {
            flame_ubo.destroy(&self.rrdevice.device);
            log!("Destroyed flame uniform buffer");
        }

        if let Some(mut water_descriptor) = self.data.raytracing.water_descriptor.take() {
            water_descriptor.destroy(&self.rrdevice.device);
        }

        if let Some(water_shading_pipeline) = self.data.raytracing.water_shading_pipeline.take() {
            water_shading_pipeline.destroy(&self.rrdevice.device);
        }

        if let Some(mut water_ubo) = self.data.raytracing.water_ubo.take() {
            water_ubo.destroy(&self.rrdevice.device);
            log!("Destroyed water uniform buffer");
        }

        if let Some(mut effect_trace_descriptor) =
            self.data.raytracing.effect_trace_descriptor.take()
        {
            effect_trace_descriptor.destroy(&self.rrdevice.device);
        }

        if let Some(effect_trace_pipeline) = self.data.raytracing.effect_trace_pipeline.take() {
            effect_trace_pipeline.destroy(&self.rrdevice.device);
        }

        if let Some(mut water_caustic_descriptor) =
            self.data.raytracing.water_caustic_descriptor.take()
        {
            water_caustic_descriptor.destroy(&self.rrdevice.device);
        }

        if let Some(pipeline) = self.data.raytracing.water_caustic_splat_pipeline.take() {
            pipeline.destroy(&self.rrdevice.device);
        }

        if let Some(pipeline) = self.data.raytracing.water_caustic_apply_pipeline.take() {
            pipeline.destroy(&self.rrdevice.device);
        }

        if let (Some(buffer), Some(memory)) = (
            self.data.raytracing.scene_uniform_buffer,
            self.data.raytracing.scene_uniform_buffer_memory,
        ) {
            self.rrdevice.device.destroy_buffer(buffer, None);
            self.rrdevice.device.free_memory(memory, None);
            log!("Destroyed scene uniform buffer");
        }

        if let Some(mut ray_query_descriptor) = self.data.raytracing.ray_query_descriptor.take() {
            ray_query_descriptor.destroy(&self.rrdevice.device);
            log!("Destroyed ray query descriptor set");
        }

        if let Some(ray_query_pipeline) = self.data.raytracing.ray_query_pipeline.take() {
            ray_query_pipeline.destroy(&self.rrdevice.device);
            log!("Destroyed ray query pipeline");
        }

        if let Some(mut acceleration_structure) = self.data.raytracing.acceleration_structure.take()
        {
            acceleration_structure.destroy(&self.rrdevice.device);
            log!("Destroyed acceleration structure");
        }

        if let Some(mut gbuffer) = self.data.raytracing.gbuffer.take() {
            gbuffer.destroy(&self.rrdevice);
            log!("Destroyed G-Buffer");
        }

        self.data.buffer_registry.destroy_all(&self.rrdevice);

        thyllore_vulkan_core::GpuTimestampProfiler::destroy(
            &mut self.gpu_timestamp_profiler,
            &self.rrdevice.device,
        );
        log!("Destroyed GPU timestamp profiler");

        self.data.graphics_resources.destroy(&self.rrdevice);
        log!("Destroyed render resources");

        self.rrdevice.destroy_descriptor_pools();
        log!("Destroyed descriptor pools");

        log!("All application resources destroyed");
    }

    pub unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.rrdevice.device.device_wait_idle()?;

        {
            let render_targets = self.resource::<RenderTargets>();
            render_targets
                .render
                .destroy_size_dependent(&self.rrdevice.device);
        }

        {
            let swapchain_state = self.resource::<SwapchainState>();
            swapchain_state.swapchain.destroy(&self.rrdevice.device);
        }

        let command_pool_handle = {
            let command_state = self.resource::<CommandState>();
            let pool_handle = command_state.pool.command_pool;
            self.rrdevice
                .device
                .free_command_buffers(pool_handle, &command_state.buffers.command_buffers);
            pool_handle
        };

        let surface = self
            .resource::<crate::vulkanr::context::SurfaceState>()
            .surface;
        let new_swapchain = RRSwapchain::new(window, &self.instance, &surface, &self.rrdevice)?;
        let image_count = new_swapchain.swapchain_images.len();

        {
            let mut render_targets = self.resource_mut::<RenderTargets>();
            create_depth_objects(
                &self.instance,
                &self.rrdevice,
                &new_swapchain,
                &crate::vulkanr::command::RRCommandPool {
                    command_pool: command_pool_handle,
                },
                &mut render_targets.render,
            )?;
            create_color_objects(
                &self.instance,
                &self.rrdevice,
                &new_swapchain,
                &mut render_targets.render,
            )?;
            create_framebuffers(&self.rrdevice, &new_swapchain, &mut render_targets.render)?;
        }

        {
            let mut command_state = self.resource_mut::<CommandState>();
            let render_targets = self.resource::<RenderTargets>();
            RRCommandBuffer::allocate_command_buffers(
                &self.rrdevice,
                &render_targets.render,
                &mut command_state.buffers,
            )?;
        }

        {
            let mut swapchain_state = self.resource_mut::<SwapchainState>();
            swapchain_state.swapchain = new_swapchain;
            swapchain_state.images_in_flight = vec![vk::Fence::null(); image_count];
        }

        log!("Swapchain recreated successfully");
        Ok(())
    }
}
