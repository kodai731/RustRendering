use crate::core::device::*;
use crate::descriptor::pass_manifest::{SetRole, BLOOM_DOWNSAMPLE, BLOOM_UPSAMPLE};
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::bloom_downsample;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
struct BloomFrameSets {
    downsample_sets: Vec<vk::DescriptorSet>,
    upsample_sets: Vec<vk::DescriptorSet>,
}

#[derive(Clone, Debug, Default)]
pub struct RRBloomDescriptorSets {
    pub layout: ReflectedSetLayout,
    frames: Vec<BloomFrameSets>,
}

impl RRBloomDescriptorSets {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::new(vec![&BLOOM_DOWNSAMPLE, &BLOOM_UPSAMPLE], SetRole::Local)
    }

    pub unsafe fn new(
        rrdevice: &RRDevice,
        mip_count: usize,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;

        let mut frames = Vec::with_capacity(frames_in_flight.max(1));
        for _ in 0..frames_in_flight.max(1) {
            frames.push(BloomFrameSets {
                downsample_sets: layout.allocate_sets(rrdevice, mip_count)?,
                upsample_sets: layout.allocate_sets(rrdevice, mip_count.saturating_sub(1))?,
            });
        }

        Ok(Self { layout, frames })
    }

    fn frame(&self, frame_slot: usize) -> Result<&BloomFrameSets> {
        self.frames.get(frame_slot).ok_or_else(|| {
            anyhow!(
                "bloom descriptor slot {frame_slot} exceeds {} frames",
                self.frames.len()
            )
        })
    }

    pub fn downsample_set(&self, frame_slot: usize, mip_index: usize) -> Result<vk::DescriptorSet> {
        self.frame(frame_slot)?
            .downsample_sets
            .get(mip_index)
            .copied()
            .ok_or_else(|| anyhow!("bloom downsample set {mip_index} is missing"))
    }

    pub fn upsample_set(&self, frame_slot: usize, pass_index: usize) -> Result<vk::DescriptorSet> {
        self.frame(frame_slot)?
            .upsample_sets
            .get(pass_index)
            .copied()
            .ok_or_else(|| anyhow!("bloom upsample set {pass_index} is missing"))
    }

    unsafe fn write_input(
        &self,
        rrdevice: &RRDevice,
        descriptor_set: vk::DescriptorSet,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<()> {
        self.layout
            .writer(descriptor_set)
            .image(
                bloom_downsample::INPUT_SAMPLER,
                image_view,
                sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn update_image_views_at(
        &self,
        rrdevice: &RRDevice,
        frame_slot: usize,
        hdr_image_view: vk::ImageView,
        mip_image_views: &[vk::ImageView],
        sampler: vk::Sampler,
    ) -> Result<()> {
        let frame = self.frame(frame_slot)?;
        if mip_image_views.len() != frame.downsample_sets.len() {
            return Err(anyhow!(
                "bloom mip view count {} does not match {} descriptor sets",
                mip_image_views.len(),
                frame.downsample_sets.len()
            ));
        }

        for (mip_index, set) in frame.downsample_sets.iter().enumerate() {
            let input_view = match mip_index {
                0 => hdr_image_view,
                _ => mip_image_views[mip_index - 1],
            };
            self.write_input(rrdevice, *set, input_view, sampler)?;
        }

        for (pass_index, set) in frame.upsample_sets.iter().enumerate() {
            let source_view_index = mip_image_views.len() - 1 - pass_index;
            self.write_input(rrdevice, *set, mip_image_views[source_view_index], sampler)?;
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
        self.frames.clear();
    }
}
