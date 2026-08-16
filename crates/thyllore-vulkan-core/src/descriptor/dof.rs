use crate::core::device::*;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRDofDescriptorSet {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRDofDescriptorSet {
    pub fn layout_bindings() -> Vec<vk::DescriptorSetLayoutBinding> {
        let hdr_sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let depth_sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        vec![hdr_sampler_binding, depth_sampler_binding]
    }

    pub unsafe fn create_layout(rrdevice: &RRDevice) -> Result<vk::DescriptorSetLayout> {
        let bindings = Self::layout_bindings();
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        Ok(rrdevice.device.create_descriptor_set_layout(&info, None)?)
    }

    pub unsafe fn create_pool(rrdevice: &RRDevice) -> Result<vk::DescriptorPool> {
        let sampler_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(2);

        let pool_sizes = [sampler_size];
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
        depth_image_view: vk::ImageView,
        depth_sampler: vk::Sampler,
    ) -> Result<()> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = rrdevice.device.allocate_descriptor_sets(&alloc_info)?;
        self.descriptor_set = descriptor_sets[0];

        self.update_image_views(
            rrdevice,
            hdr_image_view,
            hdr_sampler,
            depth_image_view,
            depth_sampler,
        );

        Ok(())
    }

    pub unsafe fn update_image_views(
        &self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
        depth_image_view: vk::ImageView,
        depth_sampler: vk::Sampler,
    ) {
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

        let depth_image_info = vk::DescriptorImageInfo::builder()
            .image_view(depth_image_view)
            .sampler(depth_sampler)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
            .build();

        let depth_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&depth_image_info))
            .build();

        rrdevice
            .device
            .update_descriptor_sets(&[hdr_write, depth_write], &[] as &[vk::CopyDescriptorSet]);
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
