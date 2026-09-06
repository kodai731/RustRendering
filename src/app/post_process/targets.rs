use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::hooks::pass::TransientSlot;
use crate::vulkanr::resource::BloomMipTarget;

pub const DOF_OUTPUT: TransientSlot = TransientSlot("dof.output");
pub const MAX_BLOOM_MIPS: usize = 8;
pub const BLOOM_MIPS: [TransientSlot; MAX_BLOOM_MIPS] = [
    TransientSlot("bloom.mip0"),
    TransientSlot("bloom.mip1"),
    TransientSlot("bloom.mip2"),
    TransientSlot("bloom.mip3"),
    TransientSlot("bloom.mip4"),
    TransientSlot("bloom.mip5"),
    TransientSlot("bloom.mip6"),
    TransientSlot("bloom.mip7"),
];

/// Remembers which image generations a frame slot's descriptor set was last written with,
/// so a transient that landed on the same pooled image needs no rewrite.
#[derive(Debug, Default)]
struct BoundGenerations {
    per_slot: Vec<Option<Vec<u64>>>,
}

impl BoundGenerations {
    fn is_bound(&self, frame_slot: usize, generations: &[u64]) -> bool {
        self.per_slot
            .get(frame_slot)
            .and_then(|bound| bound.as_deref())
            == Some(generations)
    }

    fn mark_bound(&mut self, frame_slot: usize, generations: Vec<u64>) {
        if self.per_slot.len() <= frame_slot {
            self.per_slot.resize(frame_slot + 1, None);
        }
        self.per_slot[frame_slot] = Some(generations);
    }

    fn forget(&mut self) {
        self.per_slot.clear();
    }
}

#[derive(Debug, Default)]
pub struct PostProcessFrameTargets {
    pub dof_framebuffer: vk::Framebuffer,
    pub bloom_mips: Vec<BloomMipTarget>,
    tonemap_bound: BoundGenerations,
    histogram_bound: BoundGenerations,
    bloom_bound: BoundGenerations,
}

impl PostProcessFrameTargets {
    pub fn forget_bindings(&mut self) {
        self.tonemap_bound.forget();
        self.histogram_bound.forget();
        self.bloom_bound.forget();
    }
}

const HDR_DIRECT_GENERATION: u64 = 0;

struct InputBinding {
    view: vk::ImageView,
    sampler: vk::Sampler,
    generation: u64,
}

pub fn is_dof_enabled(app: &App) -> bool {
    app.data.raytracing.dof_pipeline.is_some() && app.data.viewport.dof_buffer.is_some()
}

pub fn bloom_mip_count(app: &App) -> usize {
    let enabled = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::BloomSettings>()
        .is_some_and(|settings| settings.enabled)
        && app.data.raytracing.bloom_downsample_pipeline.is_some();
    if !enabled {
        return 0;
    }
    app.data
        .viewport
        .bloom_chain
        .as_ref()
        .map(|chain| chain.mip_count().min(MAX_BLOOM_MIPS))
        .unwrap_or(0)
}

impl App {
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

    fn transient_binding(&self, slot: TransientSlot, sampler: vk::Sampler) -> Result<InputBinding> {
        let handle = self.data.frame_transients.handle(slot)?;
        let image = self.data.viewport.transient.get(handle)?;
        Ok(InputBinding {
            view: image.view,
            sampler,
            generation: image.generation,
        })
    }

    fn tonemap_input_binding(&self) -> Result<InputBinding> {
        match self.data.viewport.dof_buffer.as_ref() {
            Some(dof_buffer) if is_dof_enabled(self) => {
                self.transient_binding(DOF_OUTPUT, dof_buffer.sampler)
            }
            _ => self.hdr_binding(),
        }
    }

    fn bloom_input_binding(&self) -> Result<InputBinding> {
        match self.data.viewport.bloom_chain.as_ref() {
            Some(chain) if bloom_mip_count(self) > 0 => {
                self.transient_binding(BLOOM_MIPS[0], chain.sampler)
            }
            _ => self.hdr_binding(),
        }
    }

    pub(crate) unsafe fn prepare_dof_target(&mut self) -> Result<()> {
        let Some(dof_buffer) = self
            .data
            .viewport
            .dof_buffer
            .as_ref()
            .filter(|_| is_dof_enabled(self))
        else {
            self.data.post_process.dof_framebuffer = vk::Framebuffer::null();
            return Ok(());
        };
        let desc = dof_buffer.output_desc();
        let render_pass = dof_buffer.render_pass;

        let handle = self.data.frame_transients.handle(DOF_OUTPUT)?;
        let transient = &mut self.data.viewport.transient;
        let output = transient.get(handle)?;
        self.data.post_process.dof_framebuffer = transient.framebuffer(
            &self.rrdevice.device,
            render_pass,
            &[output.view],
            desc.width,
            desc.height,
        )?;
        Ok(())
    }

    pub(crate) unsafe fn prepare_bloom_targets(&mut self, frame_slot: usize) -> Result<()> {
        let mip_count = bloom_mip_count(self);
        let Some(bloom_chain) = self
            .data
            .viewport
            .bloom_chain
            .as_ref()
            .filter(|_| mip_count > 0)
        else {
            self.data.post_process.bloom_mips.clear();
            return Ok(());
        };
        let sampler = bloom_chain.sampler;
        let render_pass = bloom_chain.downsample_render_pass;

        let mut mips = Vec::with_capacity(mip_count);
        let mut generations = Vec::with_capacity(mip_count);
        for slot in &BLOOM_MIPS[..mip_count] {
            let handle = self.data.frame_transients.handle(*slot)?;
            let transient = &mut self.data.viewport.transient;
            let image = transient.get(handle)?;
            let extent = vk::Extent2D {
                width: image.desc.width,
                height: image.desc.height,
            };
            let framebuffer = transient.framebuffer(
                &self.rrdevice.device,
                render_pass,
                &[image.view],
                extent.width,
                extent.height,
            )?;
            mips.push(BloomMipTarget {
                image: image.image,
                view: image.view,
                framebuffer,
                extent,
            });
            generations.push(image.generation);
        }
        let views: Vec<vk::ImageView> = mips.iter().map(|mip| mip.view).collect();
        self.data.post_process.bloom_mips = mips;

        if self
            .data
            .post_process
            .bloom_bound
            .is_bound(frame_slot, &generations)
        {
            return Ok(());
        }
        if let (Some(bloom_descriptors), Some(hdr_buffer)) = (
            self.data.raytracing.bloom_descriptors.as_ref(),
            self.data.viewport.hdr_buffer.as_ref(),
        ) {
            bloom_descriptors.update_image_views_at(
                &self.rrdevice,
                frame_slot,
                hdr_buffer.color_image_view,
                &views,
                sampler,
            )?;
        }
        self.data
            .post_process
            .bloom_bound
            .mark_bound(frame_slot, generations);
        Ok(())
    }

    pub(crate) unsafe fn prepare_auto_exposure_input(&mut self, frame_slot: usize) -> Result<()> {
        let input = self.tonemap_input_binding()?;
        let generations = vec![input.generation];
        if self
            .data
            .post_process
            .histogram_bound
            .is_bound(frame_slot, &generations)
        {
            return Ok(());
        }

        if let Some(histogram_descriptor) = self
            .data
            .raytracing
            .auto_exposure_histogram_descriptor
            .as_ref()
        {
            histogram_descriptor.update_hdr_image_at(
                &self.rrdevice,
                frame_slot,
                input.view,
                input.sampler,
            )?;
        }
        self.data
            .post_process
            .histogram_bound
            .mark_bound(frame_slot, generations);
        Ok(())
    }

    pub(crate) unsafe fn prepare_tonemap_inputs(&mut self, frame_slot: usize) -> Result<()> {
        let tonemap_input = self.tonemap_input_binding()?;
        let bloom_input = self.bloom_input_binding()?;
        let generations = vec![tonemap_input.generation, bloom_input.generation];
        if self
            .data
            .post_process
            .tonemap_bound
            .is_bound(frame_slot, &generations)
        {
            return Ok(());
        }

        if let Some(tonemap_descriptor) = self.data.raytracing.tonemap_descriptor.as_ref() {
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
        self.data
            .post_process
            .tonemap_bound
            .mark_bound(frame_slot, generations);
        Ok(())
    }
}
