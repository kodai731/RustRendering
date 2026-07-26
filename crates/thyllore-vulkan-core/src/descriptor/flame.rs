use crate::core::device::*;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRFlameDescriptorSet {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: [vk::DescriptorSet; 2],
    pub scene_depth_sampler: vk::Sampler,
}

impl RRFlameDescriptorSet {
    pub unsafe fn create_layout(rrdevice: &RRDevice) -> Result<vk::DescriptorSetLayout> {
        let flame_ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
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

        let sdf_sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let scene_depth_sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(6)
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
            sdf_sampler_binding,
            scene_depth_sampler_binding,
        ];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        Ok(rrdevice.device.create_descriptor_set_layout(&info, None)?)
    }

    pub unsafe fn create_pool(rrdevice: &RRDevice) -> Result<vk::DescriptorPool> {
        let sampler_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(12);
        let ubo_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(2);

        let pool_sizes = [sampler_size, ubo_size];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(2);

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
        history_image_views: [vk::ImageView; 2],
        flame_sampler: vk::Sampler,
        sdf_image_view: vk::ImageView,
        sdf_sampler: vk::Sampler,
        scene_depth_view: vk::ImageView,
    ) -> Result<()> {
        let layouts = [self.descriptor_set_layout; 2];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = rrdevice.device.allocate_descriptor_sets(&alloc_info)?;
        self.descriptor_sets = [descriptor_sets[0], descriptor_sets[1]];

        // Create scene depth sampler (NEAREST/CLAMP_TO_EDGE — depth formats must not use LINEAR)
        let scene_depth_sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .build();
        self.scene_depth_sampler = rrdevice.device.create_sampler(&scene_depth_sampler_info, None)?;

        // Write UBO to both sets
        for i in 0..2 {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(flame_ubo_buffer)
                .offset(0)
                .range(flame_ubo_size)
                .build();

            let write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(std::slice::from_ref(&buffer_info))
                .build();

            rrdevice
                .device
                .update_descriptor_sets(&[write], &[] as &[vk::CopyDescriptorSet]);
        }

        // Write bindings 1-3 to both sets, binding 4 with ping-pong (set i gets history_image_views[1-i])
        for i in 0..2 {
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
                .image_view(history_image_views[1 - i])
                .sampler(flame_sampler)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build();

            let sdf_info = vk::DescriptorImageInfo::builder()
                .image_view(sdf_image_view)
                .sampler(sdf_sampler)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build();

            let scene_depth_info = vk::DescriptorImageInfo::builder()
                .image_view(scene_depth_view)
                .sampler(self.scene_depth_sampler)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
                .build();

            let writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&position_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&accum_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(3)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&interval_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(4)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&history_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(5)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&sdf_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(6)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&scene_depth_info))
                    .build(),
            ];

            rrdevice
                .device
                .update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }

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

        // Write to both descriptor sets
        for i in 0..2 {
            let write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info))
                .build();

            rrdevice
                .device
                .update_descriptor_sets(&[write], &[] as &[vk::CopyDescriptorSet]);
        }
    }

    pub unsafe fn update_image_views(
        &self,
        rrdevice: &RRDevice,
        position_image_view: vk::ImageView,
        position_sampler: vk::Sampler,
        accum_image_view: vk::ImageView,
        interval_image_view: vk::ImageView,
        history_image_views: [vk::ImageView; 2],
        flame_sampler: vk::Sampler,
        sdf_image_view: vk::ImageView,
        sdf_sampler: vk::Sampler,
        scene_depth_view: vk::ImageView,
    ) {
        // Update both descriptor sets with ping-pong history views
        for i in 0..2 {
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
                .image_view(history_image_views[1 - i])
                .sampler(flame_sampler)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build();

            let sdf_info = vk::DescriptorImageInfo::builder()
                .image_view(sdf_image_view)
                .sampler(sdf_sampler)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build();

            let scene_depth_info = vk::DescriptorImageInfo::builder()
                .image_view(scene_depth_view)
                .sampler(self.scene_depth_sampler)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
                .build();

            let writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&position_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&accum_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(3)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&interval_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(4)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&history_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(5)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&sdf_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(6)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&scene_depth_info))
                    .build(),
            ];

            rrdevice
                .device
                .update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }
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

        device.destroy_sampler(self.scene_depth_sampler, None);
    }
}
