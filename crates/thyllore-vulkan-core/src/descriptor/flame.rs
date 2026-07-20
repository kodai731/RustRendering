use crate::core::device::*;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRFlameDescriptorSet {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRFlameDescriptorSet {
    pub unsafe fn create_layout(rrdevice: &RRDevice) -> Result<vk::DescriptorSetLayout> {
        let flame_ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::GEOMETRY | vk::ShaderStageFlags::FRAGMENT)
            .build();

        let position_sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let accum_sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let interval_sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let history_sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let bindings = [
            flame_ubo_binding,
            position_sampler_binding,
            accum_sampler_binding,
            interval_sampler_binding,
            history_sampler_binding,
        ];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        Ok(rrdevice.device.create_descriptor_set_layout(&info, None)?)
    }

    pub unsafe fn create_pool(rrdevice: &RRDevice) -> Result<vk::DescriptorPool> {
        let sampler_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(4);

        let ubo_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1);

        let pool_sizes = [sampler_size, ubo_size];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        Ok(rrdevice.device.create_descriptor_pool(&info, None)?)
    }

    pub unsafe fn allocate_and_update(
        &mut self,
        rrdevice: &RRDevice,
        flame_ubo_buffer: vk::Buffer,
        flame_ubo_size: vk::DeviceSize,
        position_image_view: vk::ImageView,
        position_sampler: vk::Sampler,
        accum_image_view: vk::ImageView,
        interval_image_view: vk::ImageView,
        history_image_view: vk::ImageView,
        flame_sampler: vk::Sampler,
    ) -> Result<()> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = rrdevice.device.allocate_descriptor_sets(&alloc_info)?;
        self.descriptor_set = descriptor_sets[0];

        self.update_flame_ubo(rrdevice, flame_ubo_buffer, flame_ubo_size);
        self.update_image_views(
            rrdevice,
            position_image_view,
            position_sampler,
            accum_image_view,
            interval_image_view,
            history_image_view,
            flame_sampler,
        );

        Ok(())
    }

    pub unsafe fn update_flame_ubo(
        &self,
        rrdevice: &RRDevice,
        flame_ubo_buffer: vk::Buffer,
        flame_ubo_size: vk::DeviceSize,
    ) {
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(flame_ubo_buffer)
            .offset(0)
            .range(flame_ubo_size)
            .build();

        let write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info))
            .build();

        rrdevice
            .device
            .update_descriptor_sets(&[write], &[] as &[vk::CopyDescriptorSet]);
    }

    pub unsafe fn update_image_views(
        &self,
        rrdevice: &RRDevice,
        position_image_view: vk::ImageView,
        position_sampler: vk::Sampler,
        accum_image_view: vk::ImageView,
        interval_image_view: vk::ImageView,
        history_image_view: vk::ImageView,
        flame_sampler: vk::Sampler,
    ) {
        let position_info = vk::DescriptorImageInfo::builder()
            .image_view(position_image_view)
            .sampler(position_sampler)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let accum_info = vk::DescriptorImageInfo::builder()
            .image_view(accum_image_view)
            .sampler(flame_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let interval_info = vk::DescriptorImageInfo::builder()
            .image_view(interval_image_view)
            .sampler(flame_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let history_info = vk::DescriptorImageInfo::builder()
            .image_view(history_image_view)
            .sampler(flame_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&position_info))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&accum_info))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&interval_info))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(4)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&history_info))
                .build(),
        ];

        rrdevice
            .device
            .update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
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
