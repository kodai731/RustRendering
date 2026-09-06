use crate::core::device::*;
use crate::descriptor::pass_manifest::TONEMAP;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::tonemap;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRToneMapDescriptorSet {
    pub layout: ReflectedSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl RRToneMapDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&TONEMAP)
    }

    pub unsafe fn new(rrdevice: &RRDevice, set_count: usize) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let descriptor_sets = layout.allocate_sets(rrdevice, set_count.max(1))?;

        Ok(Self {
            layout,
            descriptor_sets,
        })
    }

    pub fn descriptor_set(&self, frame_slot: usize) -> Result<vk::DescriptorSet> {
        self.descriptor_sets
            .get(frame_slot)
            .copied()
            .ok_or_else(|| {
                anyhow!(
                    "tonemap descriptor slot {frame_slot} exceeds {} sets",
                    self.descriptor_sets.len()
                )
            })
    }

    pub unsafe fn write_all(
        &self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
        position_image_view: vk::ImageView,
        position_sampler: vk::Sampler,
        scene_buffer: vk::Buffer,
        scene_buffer_size: vk::DeviceSize,
    ) -> Result<()> {
        self.update_hdr_sampler(rrdevice, hdr_image_view, hdr_sampler)?;
        self.update_bloom_sampler(rrdevice, hdr_image_view, hdr_sampler)?;
        self.update_position_sampler(rrdevice, position_image_view, position_sampler)?;
        self.update_scene_buffer(rrdevice, scene_buffer, scene_buffer_size)
    }

    pub unsafe fn update_hdr_sampler(
        &self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
    ) -> Result<()> {
        for frame_slot in 0..self.descriptor_sets.len() {
            self.update_hdr_sampler_at(rrdevice, frame_slot, hdr_image_view, hdr_sampler)?;
        }
        Ok(())
    }

    pub unsafe fn update_hdr_sampler_at(
        &self,
        rrdevice: &RRDevice,
        frame_slot: usize,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set(frame_slot)?)
            .image(
                tonemap::HDR_SAMPLER,
                hdr_image_view,
                hdr_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn update_bloom_sampler(
        &self,
        rrdevice: &RRDevice,
        bloom_image_view: vk::ImageView,
        bloom_sampler: vk::Sampler,
    ) -> Result<()> {
        for frame_slot in 0..self.descriptor_sets.len() {
            self.update_bloom_sampler_at(rrdevice, frame_slot, bloom_image_view, bloom_sampler)?;
        }
        Ok(())
    }

    pub unsafe fn update_bloom_sampler_at(
        &self,
        rrdevice: &RRDevice,
        frame_slot: usize,
        bloom_image_view: vk::ImageView,
        bloom_sampler: vk::Sampler,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set(frame_slot)?)
            .image(
                tonemap::BLOOM_SAMPLER,
                bloom_image_view,
                bloom_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn update_position_sampler(
        &self,
        rrdevice: &RRDevice,
        position_image_view: vk::ImageView,
        position_sampler: vk::Sampler,
    ) -> Result<()> {
        for descriptor_set in &self.descriptor_sets {
            self.layout
                .writer(*descriptor_set)
                .image(
                    tonemap::POSITION_SAMPLER,
                    position_image_view,
                    position_sampler,
                    vk::ImageLayout::GENERAL,
                )?
                .apply(rrdevice);
        }
        Ok(())
    }

    pub unsafe fn update_scene_buffer(
        &self,
        rrdevice: &RRDevice,
        scene_buffer: vk::Buffer,
        scene_buffer_size: vk::DeviceSize,
    ) -> Result<()> {
        for descriptor_set in &self.descriptor_sets {
            self.layout
                .writer(*descriptor_set)
                .buffer(tonemap::SCENE_DATA, scene_buffer, 0, scene_buffer_size)?
                .apply(rrdevice);
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}
