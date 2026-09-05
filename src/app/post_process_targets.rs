use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::vulkanr::resource::TransientHandle;

#[derive(Debug, Default)]
pub struct PostProcessFrameTargets {
    pub dof_output: Option<TransientHandle>,
    pub dof_framebuffer: vk::Framebuffer,
    bound_input_generation: Vec<Option<u64>>,
}

impl PostProcessFrameTargets {
    pub fn forget_bindings(&mut self) {
        self.bound_input_generation.clear();
    }

    fn is_bound(&self, frame_slot: usize, generation: u64) -> bool {
        self.bound_input_generation
            .get(frame_slot)
            .copied()
            .flatten()
            == Some(generation)
    }

    fn mark_bound(&mut self, frame_slot: usize, generation: u64) {
        if self.bound_input_generation.len() <= frame_slot {
            self.bound_input_generation.resize(frame_slot + 1, None);
        }
        self.bound_input_generation[frame_slot] = Some(generation);
    }
}

const HDR_DIRECT_GENERATION: u64 = 0;

impl App {
    pub unsafe fn prepare_post_process_targets(&mut self, frame_slot: usize) -> Result<()> {
        let (tonemap_input_view, tonemap_input_sampler, generation) = self.acquire_dof_output()?;

        if self
            .data
            .post_process_targets
            .is_bound(frame_slot, generation)
        {
            return Ok(());
        }

        if let Some(ref tonemap_descriptor) = self.data.raytracing.tonemap_descriptor {
            tonemap_descriptor.update_hdr_sampler_at(
                &self.rrdevice,
                frame_slot,
                tonemap_input_view,
                tonemap_input_sampler,
            )?;
        }

        if let Some(ref histogram_descriptor) =
            self.data.raytracing.auto_exposure_histogram_descriptor
        {
            histogram_descriptor.update_hdr_image_at(
                &self.rrdevice,
                frame_slot,
                tonemap_input_view,
                tonemap_input_sampler,
            )?;
        }

        self.data
            .post_process_targets
            .mark_bound(frame_slot, generation);
        Ok(())
    }

    unsafe fn acquire_dof_output(&mut self) -> Result<(vk::ImageView, vk::Sampler, u64)> {
        let hdr_binding = self
            .data
            .viewport
            .hdr_buffer
            .as_ref()
            .map(|hdr| (hdr.color_image_view, hdr.sampler))
            .ok_or_else(|| anyhow::anyhow!("HDR buffer not initialized"))?;

        let dof_enabled = self.data.raytracing.dof_pipeline.is_some();
        let Some(dof_buffer) = self
            .data
            .viewport
            .dof_buffer
            .as_ref()
            .filter(|_| dof_enabled)
        else {
            self.data.post_process_targets.dof_output = None;
            self.data.post_process_targets.dof_framebuffer = vk::Framebuffer::null();
            return Ok((hdr_binding.0, hdr_binding.1, HDR_DIRECT_GENERATION));
        };

        let desc = dof_buffer.output_desc();
        let render_pass = dof_buffer.render_pass;
        let sampler = dof_buffer.sampler;
        let transient = &mut self.data.viewport.transient;

        let handle = transient.acquire(&self.instance, &self.rrdevice, desc)?;
        let output = transient.get(handle)?;
        let framebuffer = transient.framebuffer(
            &self.rrdevice.device,
            render_pass,
            &[output.view],
            desc.width,
            desc.height,
        )?;

        self.data.post_process_targets.dof_output = Some(handle);
        self.data.post_process_targets.dof_framebuffer = framebuffer;
        Ok((output.view, sampler, output.generation))
    }
}
