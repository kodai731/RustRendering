use crate::core::device::*;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRBloomDescriptorSets {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub downsample_sets: Vec<vk::DescriptorSet>,
    pub upsample_sets: Vec<vk::DescriptorSet>,
}

impl RRBloomDescriptorSets {
    pub fn layout_bindings() -> Vec<vk::DescriptorSetLayoutBinding> {
        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        vec![sampler_binding]
    }

    pub unsafe fn create_layout(rrdevice: &RRDevice) -> Result<vk::DescriptorSetLayout> {
        let bindings = Self::layout_bindings();
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        Ok(rrdevice.device.create_descriptor_set_layout(&info, None)?)
    }

    pub unsafe fn create_pool(rrdevice: &RRDevice, set_count: u32) -> Result<vk::DescriptorPool> {
        let sampler_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(set_count);

        let pool_sizes = [sampler_size];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(set_count);

        let pool = rrdevice.device.create_descriptor_pool(&info, None)?;
        Ok(pool)
    }

    unsafe fn allocate_set(&self, rrdevice: &RRDevice) -> Result<vk::DescriptorSet> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let sets = rrdevice.device.allocate_descriptor_sets(&alloc_info)?;
        Ok(sets[0])
    }

    unsafe fn update_set(
        rrdevice: &RRDevice,
        descriptor_set: vk::DescriptorSet,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        let image_info = vk::DescriptorImageInfo::builder()
            .image_view(image_view)
            .sampler(sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info))
            .build();

        rrdevice
            .device
            .update_descriptor_sets(&[write], &[] as &[vk::CopyDescriptorSet]);
    }

    pub unsafe fn allocate_and_update(
        &mut self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        mip_image_views: &[vk::ImageView],
        sampler: vk::Sampler,
    ) -> Result<()> {
        self.allocate_downsample_sets(rrdevice, hdr_image_view, mip_image_views, sampler)?;
        self.allocate_upsample_sets(rrdevice, mip_image_views, sampler)?;
        Ok(())
    }

    unsafe fn allocate_downsample_sets(
        &mut self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        mip_image_views: &[vk::ImageView],
        sampler: vk::Sampler,
    ) -> Result<()> {
        for i in 0..mip_image_views.len() {
            let set = self.allocate_set(rrdevice)?;
            let input_view = if i == 0 {
                hdr_image_view
            } else {
                mip_image_views[i - 1]
            };
            Self::update_set(rrdevice, set, input_view, sampler);
            self.downsample_sets.push(set);
        }
        Ok(())
    }

    unsafe fn allocate_upsample_sets(
        &mut self,
        rrdevice: &RRDevice,
        mip_image_views: &[vk::ImageView],
        sampler: vk::Sampler,
    ) -> Result<()> {
        for i in (0..mip_image_views.len() - 1).rev() {
            let set = self.allocate_set(rrdevice)?;
            Self::update_set(rrdevice, set, mip_image_views[i + 1], sampler);
            self.upsample_sets.push(set);
        }
        Ok(())
    }

    pub unsafe fn update_image_views(
        &self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        mip_image_views: &[vk::ImageView],
        sampler: vk::Sampler,
    ) {
        for (i, set) in self.downsample_sets.iter().enumerate() {
            let input_view = if i == 0 {
                hdr_image_view
            } else {
                mip_image_views[i - 1]
            };
            Self::update_set(rrdevice, *set, input_view, sampler);
        }

        for (pass_idx, set) in self.upsample_sets.iter().enumerate() {
            let source_view_idx = mip_image_views.len() - 1 - pass_idx;
            Self::update_set(rrdevice, *set, mip_image_views[source_view_idx], sampler);
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

        self.downsample_sets.clear();
        self.upsample_sets.clear();
    }
}
