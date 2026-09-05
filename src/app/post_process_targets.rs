use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::vulkanr::resource::{BloomMipTarget, TransientHandle};

#[derive(Debug, Default)]
pub struct PostProcessFrameTargets {
    pub dof_output: Option<TransientHandle>,
    pub dof_framebuffer: vk::Framebuffer,
    pub bloom_handles: Vec<TransientHandle>,
    pub bloom_mips: Vec<BloomMipTarget>,
    bound_generations: Vec<Option<Vec<u64>>>,
}

impl PostProcessFrameTargets {
    pub fn forget_bindings(&mut self) {
        self.bound_generations.clear();
    }

    fn is_bound(&self, frame_slot: usize, generations: &[u64]) -> bool {
        self.bound_generations
            .get(frame_slot)
            .and_then(|bound| bound.as_deref())
            == Some(generations)
    }

    fn mark_bound(&mut self, frame_slot: usize, generations: Vec<u64>) {
        if self.bound_generations.len() <= frame_slot {
            self.bound_generations.resize(frame_slot + 1, None);
        }
        self.bound_generations[frame_slot] = Some(generations);
    }
}

const HDR_DIRECT_GENERATION: u64 = 0;

struct InputBinding {
    view: vk::ImageView,
    sampler: vk::Sampler,
    generation: u64,
}

impl App {
    pub unsafe fn prepare_post_process_targets(&mut self, frame_slot: usize) -> Result<()> {
        let tonemap_input = self.acquire_dof_output()?;
        let (bloom_input, bloom_mip_views) = self.acquire_bloom_mips()?;

        let mut generations = vec![tonemap_input.generation, bloom_input.generation];
        generations.extend(self.data.bloom_mip_generations());
        if self
            .data
            .post_process_targets
            .is_bound(frame_slot, &generations)
        {
            return Ok(());
        }

        if let Some(ref tonemap_descriptor) = self.data.raytracing.tonemap_descriptor {
            tonemap_descriptor.update_hdr_sampler_at(
                &self.rrdevice,
                frame_slot,
                tonemap_input.view,
                tonemap_input.sampler,
            )?;
            tonemap_descriptor.update_bloom_sampler_at(
                &self.rrdevice,
                frame_slot,
                bloom_input.view,
                bloom_input.sampler,
            )?;
        }

        if let Some(ref histogram_descriptor) =
            self.data.raytracing.auto_exposure_histogram_descriptor
        {
            histogram_descriptor.update_hdr_image_at(
                &self.rrdevice,
                frame_slot,
                tonemap_input.view,
                tonemap_input.sampler,
            )?;
        }

        if let (Some(bloom_descriptors), Some(bloom_chain), Some(hdr_buffer)) = (
            self.data.raytracing.bloom_descriptors.as_ref(),
            self.data.viewport.bloom_chain.as_ref(),
            self.data.viewport.hdr_buffer.as_ref(),
        ) {
            if !bloom_mip_views.is_empty() {
                bloom_descriptors.update_image_views_at(
                    &self.rrdevice,
                    frame_slot,
                    hdr_buffer.color_image_view,
                    &bloom_mip_views,
                    bloom_chain.sampler,
                )?;
            }
        }

        self.data
            .post_process_targets
            .mark_bound(frame_slot, generations);
        Ok(())
    }

    fn hdr_binding(&self) -> Result<InputBinding> {
        self.data
            .viewport
            .hdr_buffer
            .as_ref()
            .map(|hdr| InputBinding {
                view: hdr.color_image_view,
                sampler: hdr.sampler,
                generation: HDR_DIRECT_GENERATION,
            })
            .ok_or_else(|| anyhow::anyhow!("HDR buffer not initialized"))
    }

    unsafe fn acquire_dof_output(&mut self) -> Result<InputBinding> {
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
            return self.hdr_binding();
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
        Ok(InputBinding {
            view: output.view,
            sampler,
            generation: output.generation,
        })
    }

    unsafe fn acquire_bloom_mips(&mut self) -> Result<(InputBinding, Vec<vk::ImageView>)> {
        let bloom_enabled = self
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::BloomSettings>()
            .is_some_and(|settings| settings.enabled)
            && self.data.raytracing.bloom_downsample_pipeline.is_some();
        let Some(bloom_chain) = self
            .data
            .viewport
            .bloom_chain
            .as_ref()
            .filter(|_| bloom_enabled)
        else {
            self.data.post_process_targets.bloom_handles.clear();
            self.data.post_process_targets.bloom_mips.clear();
            return Ok((self.hdr_binding()?, Vec::new()));
        };

        let sampler = bloom_chain.sampler;
        let render_pass = bloom_chain.downsample_render_pass;
        let descs: Vec<_> = (0..bloom_chain.mip_count())
            .filter_map(|index| bloom_chain.mip_desc(index))
            .collect();
        let transient = &mut self.data.viewport.transient;

        let mut handles = Vec::with_capacity(descs.len());
        let mut mips = Vec::with_capacity(descs.len());
        for desc in descs {
            let handle = transient.acquire(&self.instance, &self.rrdevice, desc)?;
            let image = transient.get(handle)?;
            let framebuffer = transient.framebuffer(
                &self.rrdevice.device,
                render_pass,
                &[image.view],
                desc.width,
                desc.height,
            )?;
            handles.push(handle);
            mips.push(BloomMipTarget {
                image: image.image,
                view: image.view,
                framebuffer,
                extent: vk::Extent2D {
                    width: desc.width,
                    height: desc.height,
                },
            });
        }

        let first = mips
            .first()
            .ok_or_else(|| anyhow::anyhow!("bloom chain has no mip levels"))?;
        let binding = InputBinding {
            view: first.view,
            sampler,
            generation: transient.get(handles[0])?.generation,
        };
        let views = mips.iter().map(|mip| mip.view).collect();

        self.data.post_process_targets.bloom_handles = handles;
        self.data.post_process_targets.bloom_mips = mips;
        Ok((binding, views))
    }
}

impl crate::app::data::AppData {
    fn bloom_mip_generations(&self) -> Vec<u64> {
        self.post_process_targets
            .bloom_handles
            .iter()
            .filter_map(|handle| self.viewport.transient.get(*handle).ok())
            .map(|image| image.generation)
            .collect()
    }
}
