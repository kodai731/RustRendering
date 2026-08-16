use crate::core::device::*;
use crate::core::swapchain::*;
use crate::data::*;
use crate::descriptor::pass_shaders::BILLBOARD_SHADERS;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::vulkan::*;

const UBO_BINDING: u32 = 0;
const TEXTURE_SAMPLER_BINDING: u32 = 1;
const POSITION_SAMPLER_BINDING: u32 = 2;
const MAX_BILLBOARDS_PER_SWAPCHAIN_IMAGE: usize = 5;

#[derive(Clone, Debug, Default)]
pub struct RRBillboardDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub rrdata: Vec<RRData>,
}

impl RRBillboardDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::new(BILLBOARD_SHADERS.to_vec(), 0)
    }

    pub unsafe fn new(rrdevice: &RRDevice, rrswapchain: &RRSwapchain) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let max_sets = rrswapchain.swapchain_images.len() * MAX_BILLBOARDS_PER_SWAPCHAIN_IMAGE;
        let descriptor_pool = layout.create_pool(
            rrdevice,
            max_sets as u32,
            vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
        )?;

        Ok(Self {
            layout,
            descriptor_pool,
            descriptor_sets: Vec::new(),
            rrdata: Vec::new(),
        })
    }

    pub unsafe fn allocate_descriptor_sets(
        &mut self,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
    ) -> Result<()> {
        let count = self.rrdata.len() * rrswapchain.swapchain_images.len();
        self.descriptor_sets = self
            .layout
            .allocate_sets(rrdevice, self.descriptor_pool, count)?;
        Ok(())
    }

    pub unsafe fn update_descriptor_sets(
        &mut self,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        billboard_texture: &crate::resource::image::RRImage,
    ) -> Result<()> {
        let swapchain_images_len = rrswapchain.swapchain_images.len();

        for (billboard_index, rrdata) in self.rrdata.iter().enumerate() {
            for image_index in 0..swapchain_images_len {
                let descriptor_set =
                    self.descriptor_sets[billboard_index * swapchain_images_len + image_index];
                self.layout
                    .writer(descriptor_set)
                    .buffer(
                        UBO_BINDING,
                        rrdata.rruniform_buffers[image_index].buffer,
                        0,
                        std::mem::size_of::<UniformBufferObject>() as u64,
                    )?
                    .image(
                        TEXTURE_SAMPLER_BINDING,
                        billboard_texture.image_view,
                        billboard_texture.sampler,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    )?
                    .apply(rrdevice);
            }
        }

        Ok(())
    }

    pub unsafe fn update_position_sampler(
        &self,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        position_image_view: vk::ImageView,
        position_sampler: vk::Sampler,
    ) -> Result<()> {
        let set_count = self.rrdata.len() * rrswapchain.swapchain_images.len();

        for descriptor_set in self.descriptor_sets.iter().take(set_count) {
            self.layout
                .writer(*descriptor_set)
                .image(
                    POSITION_SAMPLER_BINDING,
                    position_image_view,
                    position_sampler,
                    vk::ImageLayout::GENERAL,
                )?
                .apply(rrdevice);
        }

        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if !self.descriptor_sets.is_empty() {
            device
                .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets)
                .ok();
            self.descriptor_sets.clear();
        }

        if self.descriptor_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.descriptor_pool = vk::DescriptorPool::null();
        }
        self.layout.destroy(device);
    }
}
