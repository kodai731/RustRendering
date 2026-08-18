use crate::core::device::*;
use crate::descriptor::pass_shaders::TONEMAP_SHADERS;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::vulkan::*;

const HDR_SAMPLER_BINDING: u32 = 0;
const BLOOM_SAMPLER_BINDING: u32 = 1;
const POSITION_SAMPLER_BINDING: u32 = 2;
const SCENE_UBO_BINDING: u32 = 3;

#[derive(Clone, Debug, Default)]
pub struct RRToneMapDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRToneMapDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::new(TONEMAP_SHADERS.to_vec(), 0)
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let descriptor_set = layout.allocate_set(rrdevice)?;

        Ok(Self {
            layout,
            descriptor_set,
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
        self.layout
            .writer(self.descriptor_set)
            .image(
                HDR_SAMPLER_BINDING,
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
        self.layout
            .writer(self.descriptor_set)
            .image(
                BLOOM_SAMPLER_BINDING,
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
        self.layout
            .writer(self.descriptor_set)
            .image(
                POSITION_SAMPLER_BINDING,
                position_image_view,
                position_sampler,
                vk::ImageLayout::GENERAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn update_scene_buffer(
        &self,
        rrdevice: &RRDevice,
        scene_buffer: vk::Buffer,
        scene_buffer_size: vk::DeviceSize,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set)
            .buffer(SCENE_UBO_BINDING, scene_buffer, 0, scene_buffer_size)?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}
