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
        let water_transforms = crate::app::model_loader::collect_water_instances(&data.ecs_world);
        let mesh_transforms =
            crate::app::model_loader::collect_mesh_transforms(&data.ecs_world, &data.ecs_assets);
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
        Self::create_flame_pipeline_with_resources(instance, rrdevice, data, rrrender)?;
        Self::create_water_pipeline_with_resources(instance, rrdevice, data, rrrender)?;
        Self::create_wind_pipeline_with_resources(instance, rrdevice, data, rrrender)?;

        Ok(())
    }

    pub(crate) unsafe fn create_water_pipeline_with_resources(
        instance: &Instance,
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrrender: &RRRender,
    ) -> Result<()> {
        let hdr_view = match data.viewport.hdr_buffer {
            Some(ref hdr) => hdr.color_image_view,
            None => {
                log!("HDR buffer not available, skipping water pipeline");
                return Ok(());
            }
        };

        let depth_view = rrrender.gbuffer_depth_image_view;
        let w = data.viewport.width;
        let h = data.viewport.height;

        let water_buffer = thyllore_vulkan_core::resource::WaterBuffer::new(
            instance,
            rrdevice,
            data.raytracing.command_pool,
            w,
            h,
            hdr_view,
            depth_view,
        )?;

        data.viewport.water_buffer = Some(water_buffer);

        let (Some(water_buffer), Some(hdr_buffer)) = (
            data.viewport.water_buffer.as_ref(),
            data.viewport.hdr_buffer.as_ref(),
        ) else {
            log!("Water buffer not available, skipping water pipeline");
            return Ok(());
        };

        data.raytracing.create_water_pipeline(
            instance,
            rrdevice,
            rrrender,
            &data.graphics_resources,
            water_buffer,
            hdr_buffer,
        )?;

        log!("Water pipeline created successfully");
        Ok(())
    }

    pub(crate) unsafe fn create_wind_pipeline_with_resources(
        instance: &Instance,
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrrender: &RRRender,
    ) -> Result<()> {
        let Some(wind_buffer) = data.viewport.wind_buffer.as_ref() else {
            log!("Wind buffer not available, skipping wind pipeline");
            return Ok(());
        };
        data.raytracing.create_wind_pipeline(
            instance,
            rrdevice,
            rrrender,
            &data.graphics_resources,
            wind_buffer,
            rrrender.gbuffer_depth_image_view,
        )?;
        log!("Wind pipeline created successfully");
        Ok(())
    }

    pub(crate) unsafe fn create_flame_pipeline_with_resources(
        instance: &Instance,
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrrender: &RRRender,
    ) -> Result<()> {
        let flame_buffer = match data.viewport.flame_buffer {
            Some(ref flame) => flame,
            None => {
                log!("Flame buffer not available, skipping flame pipeline");
                return Ok(());
            }
        };

        let position_image_view = match data.raytracing.gbuffer {
            Some(ref gbuffer) => gbuffer.position_image_view,
            None => {
                log!("GBuffer not available, skipping flame pipeline");
                return Ok(());
            }
        };

        let position_sampler = match data.raytracing.gbuffer_sampler {
            Some(sampler) => sampler,
            None => {
                log!("GBuffer sampler not available, skipping flame pipeline");
                return Ok(());
            }
        };

        data.raytracing.create_flame_pipeline(
            instance,
            rrdevice,
            rrrender,
            &data.graphics_resources,
            flame_buffer,
            position_image_view,
            position_sampler,
            rrrender.gbuffer_depth_image_view,
        )?;

        log!("Flame pipeline created successfully");
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

    pub(crate) unsafe fn create_tonemap_pipeline_with_resources(
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrrender: &RRRender,
    ) -> Result<()> {
        let (hdr_image_view, hdr_sampler) = match data.viewport.hdr_buffer {
            Some(ref hdr) => (hdr.color_image_view, hdr.sampler),
            None => {
                log!("HDR buffer not available, skipping tonemap pipeline");
                return Ok(());
            }
        };

        let (offscreen_render_pass, offscreen_extent) = match data.viewport.offscreen {
            Some(ref offscreen) => (offscreen.render_pass, offscreen.extent()),
            None => {
                log!("Offscreen not available, skipping tonemap pipeline");
                return Ok(());
            }
        };

        let gbuffer = match data.raytracing.gbuffer {
            Some(ref gb) => gb,
            None => {
                log!("GBuffer not available, skipping tonemap pipeline");
                return Ok(());
            }
        };
        let position_image_view = gbuffer.position_image_view;

        let gbuffer_sampler = match data.raytracing.gbuffer_sampler {
            Some(s) => s,
            None => {
                log!("GBuffer sampler not available, skipping tonemap pipeline");
                return Ok(());
            }
        };

        let scene_buffer = match data.raytracing.scene_uniform_buffer {
            Some(b) => b,
            None => {
                log!("Scene buffer not available, skipping tonemap pipeline");
                return Ok(());
            }
        };

        let scene_buffer_size =
            std::mem::size_of::<thyllore_vulkan_core::data::SceneUniformData>() as vk::DeviceSize;

        data.raytracing.create_tonemap_pipeline(
            rrdevice,
            rrrender,
            hdr_image_view,
            hdr_sampler,
            position_image_view,
            gbuffer_sampler,
            scene_buffer,
            scene_buffer_size,
            offscreen_render_pass,
            offscreen_extent,
        )?;

        log!("Tonemap pipeline created successfully");
        Ok(())
    }

    pub(crate) unsafe fn create_bloom_pipelines_with_resources(
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrrender: &RRRender,
    ) -> Result<()> {
        let hdr_image_view = match data.viewport.hdr_buffer {
            Some(ref hdr) => hdr.color_image_view,
            None => {
                log!("HDR buffer not available, skipping bloom pipelines");
                return Ok(());
            }
        };

        let bloom_chain = match data.viewport.bloom_chain {
            Some(ref chain) => chain,
            None => {
                log!("Bloom chain not available, skipping bloom pipelines");
                return Ok(());
            }
        };

        data.raytracing
            .create_bloom_pipelines(rrdevice, rrrender, hdr_image_view, bloom_chain)?;

        if let (Some(ref bloom_chain), Some(ref tonemap_desc)) = (
            &data.viewport.bloom_chain,
            &data.raytracing.tonemap_descriptor,
        ) {
            if let Some(ref first_mip) = bloom_chain.mip_levels.first() {
                tonemap_desc.update_bloom_sampler(
                    rrdevice,
                    first_mip.image_view,
                    bloom_chain.sampler,
                )?;
                log!("Updated tonemap descriptor with bloom texture");
            }
        }

        log!("Bloom pipelines created successfully");
        Ok(())
    }

    pub(crate) unsafe fn create_dof_pipeline_with_resources(
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrrender: &RRRender,
    ) -> Result<()> {
        let hdr_image_view = match data.viewport.hdr_buffer {
            Some(ref hdr) => hdr.color_image_view,
            None => {
                log!("HDR buffer not available, skipping DOF pipeline");
                return Ok(());
            }
        };

        let hdr_sampler = match data.viewport.hdr_buffer {
            Some(ref hdr) => hdr.sampler,
            None => return Ok(()),
        };

        let dof_buffer = match data.viewport.dof_buffer {
            Some(ref buf) => buf,
            None => {
                log!("DOF buffer not available, skipping DOF pipeline");
                return Ok(());
            }
        };

        let depth_image_view = rrrender.gbuffer_depth_image_view;
        if depth_image_view == vk::ImageView::null() {
            log!("GBuffer depth image view not available, skipping DOF pipeline");
            return Ok(());
        }

        let depth_sampler = match data.raytracing.gbuffer_sampler {
            Some(s) => s,
            None => {
                log!("GBuffer sampler not available, skipping DOF pipeline");
                return Ok(());
            }
        };

        let dof_render_pass = dof_buffer.render_pass;

        data.raytracing.create_dof_pipeline(
            rrdevice,
            rrrender,
            hdr_image_view,
            hdr_sampler,
            depth_image_view,
            depth_sampler,
            dof_render_pass,
        )?;

        if let (Some(ref dof_buffer), Some(ref tonemap_desc)) = (
            &data.viewport.dof_buffer,
            &data.raytracing.tonemap_descriptor,
        ) {
            tonemap_desc.update_hdr_sampler(
                rrdevice,
                dof_buffer.output_image_view,
                dof_buffer.sampler,
            )?;
            log!("Updated tonemap descriptor binding 0 to DOF output");
        }

        log!("DOF pipeline created successfully");
        Ok(())
    }

    pub(crate) unsafe fn create_auto_exposure_pipelines_with_resources(
        rrdevice: &RRDevice,
        data: &mut AppData,
    ) -> Result<()> {
        let ae_buffers = match data.viewport.auto_exposure_buffers {
            Some(ref buf) => buf,
            None => {
                log!(
                    "AutoExposure buffers not available, \
                    skipping pipeline"
                );
                return Ok(());
            }
        };

        let (hdr_image_view, hdr_sampler) = Self::resolve_auto_exposure_input(data);

        if hdr_image_view == vk::ImageView::null() {
            log!("No HDR input available for AutoExposure");
            return Ok(());
        }

        let histogram_buffer = ae_buffers.histogram_buffer;
        let histogram_buffer_size = (256 * 4) as u64;
        let luminance_buffer = ae_buffers.luminance_buffer;
        let luminance_buffer_size = (2 * 4) as u64;

        data.raytracing.create_auto_exposure_pipelines(
            rrdevice,
            hdr_image_view,
            hdr_sampler,
            histogram_buffer,
            histogram_buffer_size,
            luminance_buffer,
            luminance_buffer_size,
        )?;

        log!("AutoExposure pipelines created successfully");
        Ok(())
    }

    fn resolve_auto_exposure_input(data: &AppData) -> (vk::ImageView, vk::Sampler) {
        if let Some(ref dof_buffer) = data.viewport.dof_buffer {
            return (dof_buffer.output_image_view, dof_buffer.sampler);
        }

        if let Some(ref hdr_buffer) = data.viewport.hdr_buffer {
            return (hdr_buffer.color_image_view, hdr_buffer.sampler);
        }

        (vk::ImageView::null(), vk::Sampler::null())
    }
}
