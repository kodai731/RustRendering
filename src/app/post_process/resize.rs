use anyhow::Result;

use crate::app::App;

impl App {
    pub unsafe fn resize_post_process_bindings(&mut self) -> Result<()> {
        self.update_postprocessing_descriptors_on_resize()?;
        self.data.post_process.forget_bindings();
        Ok(())
    }

    unsafe fn update_postprocessing_descriptors_on_resize(&mut self) -> Result<()> {
        if let (Some(ref hdr_buffer), Some(ref tonemap_descriptor)) = (
            &self.data.viewport.hdr_buffer,
            &self.data.raytracing.tonemap_descriptor,
        ) {
            tonemap_descriptor.update_hdr_sampler(
                &self.rrdevice,
                hdr_buffer.color_image_view,
                hdr_buffer.sampler,
            )?;
            tonemap_descriptor.update_bloom_sampler(
                &self.rrdevice,
                hdr_buffer.color_image_view,
                hdr_buffer.sampler,
            )?;
        }

        {
            let render_targets = self.resource::<crate::vulkanr::context::RenderTargets>();
            let depth_image_view = render_targets.render.gbuffer_depth_image_view;

            if let (Some(ref hdr_buffer), Some(ref dof_descriptor)) = (
                &self.data.viewport.hdr_buffer,
                &self.data.raytracing.dof_descriptor,
            ) {
                let depth_sampler = self
                    .data
                    .raytracing
                    .gbuffer_sampler
                    .unwrap_or(hdr_buffer.sampler);

                dof_descriptor.update_image_views(
                    &self.rrdevice,
                    hdr_buffer.color_image_view,
                    hdr_buffer.sampler,
                    depth_image_view,
                    depth_sampler,
                )?;
            }
        }

        if let (Some(ref gbuffer), Some(gbuffer_sampler), Some(ref tonemap_descriptor)) = (
            &self.data.raytracing.gbuffer,
            self.data.raytracing.gbuffer_sampler,
            &self.data.raytracing.tonemap_descriptor,
        ) {
            tonemap_descriptor.update_position_sampler(
                &self.rrdevice,
                gbuffer.position_image_view,
                gbuffer_sampler,
            )?;
        }

        self.update_auto_exposure_descriptors_on_resize()?;

        Ok(())
    }

    unsafe fn update_auto_exposure_descriptors_on_resize(&self) -> Result<()> {
        let Some(ref hdr_buffer) = self.data.viewport.hdr_buffer else {
            return Ok(());
        };
        let (hdr_image_view, hdr_sampler) = (hdr_buffer.color_image_view, hdr_buffer.sampler);

        let ae_buffers = match self.data.viewport.auto_exposure_buffers {
            Some(ref buf) => buf,
            None => return Ok(()),
        };

        if let Some(ref hist_desc) = self.data.raytracing.auto_exposure_histogram_descriptor {
            hist_desc.update_bindings(
                &self.rrdevice,
                hdr_image_view,
                hdr_sampler,
                ae_buffers.histogram_buffer,
                (256 * 4) as u64,
            )?;
        }

        if let Some(ref avg_desc) = self.data.raytracing.auto_exposure_average_descriptor {
            avg_desc.update_bindings(
                &self.rrdevice,
                ae_buffers.histogram_buffer,
                (256 * 4) as u64,
                ae_buffers.luminance_buffer,
                (2 * 4) as u64,
            )?;
        }
        Ok(())
    }
}
