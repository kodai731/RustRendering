use crate::app::{App, AppData};
use crate::ecs::resource::billboard::BillboardData;
use crate::vulkanr::command::RRCommandPool;
use crate::vulkanr::core::RRDevice;
use crate::vulkanr::render::{create_gbuffer_render_pass, RRRender};
use crate::vulkanr::renderer::deferred::create_gbuffer_framebuffer;
use crate::vulkanr::swapchain::RRSwapchain;

use anyhow::Result;
use std::rc::Rc;
use vulkanalia::prelude::v1_0::*;

impl App {
    pub(crate) unsafe fn init_ray_tracing_with_resources(
        instance: &Instance,
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrswapchain: &RRSwapchain,
        rrcommand_pool: &RRCommandPool,
        rrrender: &mut RRRender,
    ) -> Result<()> {
        log::info!("Initializing Ray Tracing resources...");

        data.raytracing
            .init_gbuffer(instance, rrdevice, rrswapchain, rrcommand_pool)?;

        create_gbuffer_render_pass(instance, rrdevice, rrrender)?;

        if let Some(ref gbuffer) = data.raytracing.gbuffer {
            create_gbuffer_framebuffer(instance, rrdevice, rrrender, gbuffer)?;
        }
        log::info!("Created G-Buffer render pass and framebuffer");

        if let Some(hdr_buffer) = &mut data.viewport.hdr_buffer {
            hdr_buffer.attach_depth(rrdevice, rrrender.gbuffer_depth_image_view)?;
        }

        log::info!("Ray Tracing initialization complete");
        Ok(())
    }

    pub(crate) unsafe fn build_acceleration_structures_with_resources(
        instance: &Instance,
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrcommand_pool: &Rc<RRCommandPool>,
    ) -> Result<()> {
        data.raytracing.command_pool = rrcommand_pool.command_pool;
        let water_transforms = crate::ecs::systems::collect_water_instances(&data.ecs_world);
        let mesh_transforms =
            crate::ecs::systems::collect_mesh_transforms(&data.ecs_world, &data.ecs_assets);
        data.raytracing.build_acceleration_structures(
            instance,
            rrdevice,
            rrcommand_pool,
            &data.graphics_resources.meshes,
            &mesh_transforms,
            &water_transforms,
        )
    }

    pub(crate) unsafe fn create_ray_tracing_pipelines_with_resources(
        instance: &Instance,
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrswapchain: &RRSwapchain,
        rrrender: &RRRender,
    ) -> Result<()> {
        let (offscreen_render_pass, offscreen_extent) =
            if let Some(ref offscreen) = data.viewport.offscreen {
                (Some(offscreen.render_pass), Some(offscreen.extent()))
            } else {
                (None, None)
            };

        let hdr_render_pass = data.viewport.hdr_buffer.as_ref().map(|hdr| hdr.render_pass);

        {
            let mut billboard = data.ecs_world.resource_mut::<BillboardData>();
            data.raytracing.create_pipelines(
                instance,
                rrdevice,
                rrswapchain,
                rrrender,
                &data.graphics_resources,
                &mut billboard.render_state.descriptor_set,
                offscreen_render_pass,
                offscreen_extent,
                hdr_render_pass,
            )?;
        }

        Self::create_tonemap_pipeline_with_resources(rrdevice, data, rrrender)?;
        Self::create_bloom_pipelines_with_resources(rrdevice, data, rrrender)?;
        Self::create_dof_pipeline_with_resources(rrdevice, data, rrrender)?;
        Self::create_auto_exposure_pipelines_with_resources(rrdevice, data)?;
        Self::create_onion_skin_pipeline_with_resources(instance, rrdevice, data, rrrender)?;
        crate::hooks::effect::EffectHooks::run_setup(instance, rrdevice, data, rrrender)?;

        Ok(())
    }

    pub(crate) unsafe fn create_onion_skin_pipeline_with_resources(
        instance: &Instance,
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrrender: &RRRender,
    ) -> Result<()> {
        let hdr_buffer = match data.viewport.hdr_buffer {
            Some(ref hdr) => hdr,
            None => {
                log!("HDR buffer not available, skipping onion skin pipeline");
                return Ok(());
            }
        };

        let offscreen = match data.viewport.offscreen {
            Some(ref o) => o,
            None => {
                log!("Offscreen not available, skipping onion skin pipeline");
                return Ok(());
            }
        };

        let resolve_image_view = offscreen.resolve_color_image_view;
        let offscreen_format = offscreen.format;
        let width = hdr_buffer.width;
        let height = hdr_buffer.height;

        data.raytracing.create_onion_skin_pipeline(
            instance,
            rrdevice,
            rrrender,
            &data.graphics_resources,
            resolve_image_view,
            offscreen_format,
            width,
            height,
        )?;

        log!("Onion skin pipeline created successfully");
        Ok(())
    }
}
