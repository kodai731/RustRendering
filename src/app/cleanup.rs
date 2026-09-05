use crate::app::App;
use crate::vulkanr::command::RRCommandBuffer;
use crate::vulkanr::context::{CommandState, RenderTargets, SwapchainState};
use crate::vulkanr::render::framebuffer::{create_color_objects, create_framebuffers};
use crate::vulkanr::render::pass::create_depth_objects;
use crate::vulkanr::resource::{destroy_all_in_reverse, GpuResource};
use crate::vulkanr::swapchain::RRSwapchain;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use winit::window::Window;

impl App {
    pub unsafe fn destroy(&mut self) {
        log!("Destroying application resources...");

        let _ = self.rrdevice.device.device_wait_idle();

        let mut resources: [&mut dyn GpuResource; 5] = [
            &mut self.data.graphics_resources,
            &mut self.gpu_timestamp_profiler,
            &mut self.data.buffer_registry,
            &mut self.data.pipeline_storage,
            &mut self.data.raytracing,
        ];
        destroy_all_in_reverse(&mut resources, &self.rrdevice);

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
