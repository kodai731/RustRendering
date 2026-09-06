use crate::app::{App, AppData};
use crate::vulkanr::core::RRDevice;
use crate::vulkanr::render::RRRender;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

impl App {
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
            crate::app::init::MAX_FRAMES_IN_FLIGHT,
        )?;

        log!("Tonemap pipeline created successfully");
        Ok(())
    }

    pub(crate) unsafe fn create_bloom_pipelines_with_resources(
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrrender: &RRRender,
    ) -> Result<()> {
        let bloom_chain = match data.viewport.bloom_chain {
            Some(ref chain) => chain,
            None => {
                log!("Bloom chain not available, skipping bloom pipelines");
                return Ok(());
            }
        };

        data.raytracing.create_bloom_pipelines(
            rrdevice,
            rrrender,
            bloom_chain,
            crate::app::init::MAX_FRAMES_IN_FLIGHT,
        )?;

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
            crate::app::init::MAX_FRAMES_IN_FLIGHT,
        )?;

        log!("AutoExposure pipelines created successfully");
        Ok(())
    }

    fn resolve_auto_exposure_input(data: &AppData) -> (vk::ImageView, vk::Sampler) {
        if let Some(ref hdr_buffer) = data.viewport.hdr_buffer {
            return (hdr_buffer.color_image_view, hdr_buffer.sampler);
        }

        (vk::ImageView::null(), vk::Sampler::null())
    }
}
