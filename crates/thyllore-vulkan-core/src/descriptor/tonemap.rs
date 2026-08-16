use crate::core::device::*;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRToneMapDescriptorSet {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRToneMapDescriptorSet {
    pub fn layout_bindings() -> Vec<vk::DescriptorSetLayoutBinding> {
        let hdr_sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let bloom_sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let position_sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let scene_ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        vec![
            hdr_sampler_binding,
            bloom_sampler_binding,
            position_sampler_binding,
            scene_ubo_binding,
        ]
    }

    pub unsafe fn create_layout(rrdevice: &RRDevice) -> Result<vk::DescriptorSetLayout> {
        let bindings = Self::layout_bindings();
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        Ok(rrdevice.device.create_descriptor_set_layout(&info, None)?)
    }

    pub unsafe fn create_pool(rrdevice: &RRDevice) -> Result<vk::DescriptorPool> {
        let sampler_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(3);

        let ubo_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1);

        let pool_sizes = [sampler_size, ubo_size];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let pool = rrdevice.device.create_descriptor_pool(&info, None)?;
        Ok(pool)
    }

    pub unsafe fn allocate_and_update(
        &mut self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
        position_image_view: vk::ImageView,
        position_sampler: vk::Sampler,
        scene_buffer: vk::Buffer,
        scene_buffer_size: vk::DeviceSize,
    ) -> Result<()> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = rrdevice.device.allocate_descriptor_sets(&alloc_info)?;
        self.descriptor_set = descriptor_sets[0];

        self.update_hdr_sampler(rrdevice, hdr_image_view, hdr_sampler)?;
        self.update_bloom_sampler(rrdevice, hdr_image_view, hdr_sampler)?;
        self.update_position_sampler(rrdevice, position_image_view, position_sampler)?;
        self.update_scene_buffer(rrdevice, scene_buffer, scene_buffer_size)?;

        Ok(())
    }

    pub unsafe fn update_hdr_sampler(
        &self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
    ) -> Result<()> {
        let hdr_image_info = vk::DescriptorImageInfo::builder()
            .image_view(hdr_image_view)
            .sampler(hdr_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let hdr_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&hdr_image_info))
            .build();

        rrdevice
            .device
            .update_descriptor_sets(&[hdr_write], &[] as &[vk::CopyDescriptorSet]);

        Ok(())
    }

    pub unsafe fn update_bloom_sampler(
        &self,
        rrdevice: &RRDevice,
        bloom_image_view: vk::ImageView,
        bloom_sampler: vk::Sampler,
    ) -> Result<()> {
        let bloom_image_info = vk::DescriptorImageInfo::builder()
            .image_view(bloom_image_view)
            .sampler(bloom_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let bloom_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&bloom_image_info))
            .build();

        rrdevice
            .device
            .update_descriptor_sets(&[bloom_write], &[] as &[vk::CopyDescriptorSet]);

        Ok(())
    }

    pub unsafe fn update_position_sampler(
        &self,
        rrdevice: &RRDevice,
        position_image_view: vk::ImageView,
        position_sampler: vk::Sampler,
    ) -> Result<()> {
        let position_image_info = vk::DescriptorImageInfo::builder()
            .image_view(position_image_view)
            .sampler(position_sampler)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let position_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&position_image_info))
            .build();

        rrdevice
            .device
            .update_descriptor_sets(&[position_write], &[] as &[vk::CopyDescriptorSet]);

        Ok(())
    }

    pub unsafe fn update_scene_buffer(
        &self,
        rrdevice: &RRDevice,
        scene_buffer: vk::Buffer,
        scene_buffer_size: vk::DeviceSize,
    ) -> Result<()> {
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(scene_buffer)
            .offset(0)
            .range(scene_buffer_size)
            .build();

        let scene_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info))
            .build();

        rrdevice
            .device
            .update_descriptor_sets(&[scene_write], &[] as &[vk::CopyDescriptorSet]);

        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
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
