use crate::core::device::*;
use crate::core::swapchain::*;
use crate::data::*;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRBillboardDescriptorSet {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub rrdata: Vec<RRData>,
}

impl RRBillboardDescriptorSet {
    pub unsafe fn new(rrdevice: &RRDevice, rrswapchain: &RRSwapchain) -> Result<Self> {
        let layout = Self::create_layout(rrdevice)?;
        let pool = Self::create_pool(rrdevice, rrswapchain)?;

        Ok(Self {
            descriptor_set_layout: layout,
            descriptor_pool: pool,
            descriptor_sets: Vec::new(),
            rrdata: Vec::new(),
        })
    }

    pub unsafe fn create_layout(rrdevice: &RRDevice) -> Result<vk::DescriptorSetLayout> {
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .build();

        let texture_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let position_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let bindings = [ubo_binding, texture_binding, position_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

        let layout = rrdevice.device.create_descriptor_set_layout(&info, None)?;
        Ok(layout)
    }

    pub unsafe fn create_pool(
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
    ) -> Result<vk::DescriptorPool> {
        let max_sets = (rrswapchain.swapchain_images.len() * 5) as u32;

        let ubo_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(max_sets)
            .build();

        let sampler_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(max_sets * 2)
            .build();

        let pool_sizes = [ubo_size, sampler_size];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let pool = rrdevice.device.create_descriptor_pool(&info, None)?;
        Ok(pool)
    }

    pub unsafe fn allocate_descriptor_sets(
        &mut self,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
    ) -> Result<()> {
        let swapchain_images_len = rrswapchain.swapchain_images.len();
        let layouts: Vec<_> = (0..self.rrdata.len() * swapchain_images_len)
            .map(|_| self.descriptor_set_layout)
            .collect();

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        self.descriptor_sets = rrdevice.device.allocate_descriptor_sets(&alloc_info)?;
        Ok(())
    }

    pub unsafe fn update_descriptor_sets(
        &mut self,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        billboard_texture: &crate::resource::image::RRImage,
    ) -> Result<()> {
        let swapchain_images_len = rrswapchain.swapchain_images.len();

        for i in 0..self.rrdata.len() {
            for j in 0..swapchain_images_len {
                let descriptor_set_index = i * swapchain_images_len + j;
                let rrdata = &self.rrdata[i];

                let buffer_info = vk::DescriptorBufferInfo::builder()
                    .buffer(rrdata.rruniform_buffers[j].buffer)
                    .offset(0)
                    .range(std::mem::size_of::<UniformBufferObject>() as u64)
                    .build();

                let ubo_write = vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[descriptor_set_index])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_info))
                    .build();

                let texture_info = vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(billboard_texture.image_view)
                    .sampler(billboard_texture.sampler)
                    .build();

                let texture_write = vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[descriptor_set_index])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&texture_info))
                    .build();

                rrdevice.device.update_descriptor_sets(
                    &[ubo_write, texture_write],
                    &[] as &[vk::CopyDescriptorSet],
                );
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
        let swapchain_images_len = rrswapchain.swapchain_images.len();

        println!("update_position_sampler: rrdata.len={}, swapchain_images_len={}, descriptor_sets.len={}, position_image_view={:?}, position_sampler={:?}",
            self.rrdata.len(), swapchain_images_len, self.descriptor_sets.len(), position_image_view, position_sampler);

        for i in 0..self.rrdata.len() {
            for j in 0..swapchain_images_len {
                let descriptor_set_index = i * swapchain_images_len + j;

                let position_info = vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(position_image_view)
                    .sampler(position_sampler)
                    .build();

                let position_write = vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[descriptor_set_index])
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&position_info))
                    .build();

                rrdevice
                    .device
                    .update_descriptor_sets(&[position_write], &[] as &[vk::CopyDescriptorSet]);
            }
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

        if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.descriptor_set_layout = vk::DescriptorSetLayout::null();
        }
    }
}
