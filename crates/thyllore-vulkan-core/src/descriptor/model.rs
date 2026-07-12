use crate::core::device::*;
use crate::core::swapchain::*;
use crate::data::*;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRDescriptorSet {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub rrdata: Vec<RRData>,
}

impl RRDescriptorSet {
    pub unsafe fn new(rrdevice: &RRDevice, rrswapchain: &RRSwapchain) -> Self {
        let mut rrdescriptor_sets = RRDescriptorSet::default();
        let _ = create_descriptor_set_layout(rrdevice, &mut rrdescriptor_sets);
        let _ = create_descriptor_pool(rrdevice, rrswapchain, &mut rrdescriptor_sets);
        println!("rrdescriptor_sets: {:?}", rrdescriptor_sets);
        rrdescriptor_sets
    }

    pub unsafe fn create_descriptor_set(
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        rrdescriptor_set: &mut RRDescriptorSet,
    ) -> Result<()> {
        if let Err(e) = create_descriptor_sets(rrdevice, rrswapchain, rrdescriptor_set) {
            println!("error creating descriptor set: {:?}", e);
        }
        Ok(())
    }

    pub unsafe fn delete_data(&mut self, rrdevice: &RRDevice) {
        if !self.descriptor_sets.is_empty() {
            rrdevice
                .device
                .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets)
                .ok();
            self.descriptor_sets.clear();
        }

        for i in 0..self.rrdata.len() {
            self.rrdata[i].delete(rrdevice);
        }
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

unsafe fn create_descriptor_set_layout(
    rrdevice: &RRDevice,
    rrdescriptor_set: &mut RRDescriptorSet,
) -> Result<()> {
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let bindings = &[ubo_binding, sampler_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
    rrdescriptor_set.descriptor_set_layout =
        rrdevice.device.create_descriptor_set_layout(&info, None)?;

    Ok(())
}

unsafe fn create_descriptor_pool(
    rrdevice: &RRDevice,
    rrswapchain: &RRSwapchain,
    rrdescriptor_set: &mut RRDescriptorSet,
) -> Result<()> {
    let max_meshes = 30;
    let descriptor_count = (rrswapchain.swapchain_images.len() * max_meshes) as u32;

    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(descriptor_count);

    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(descriptor_count);

    let pool_sizes = &[ubo_size, sampler_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(descriptor_count)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
    rrdescriptor_set.descriptor_pool = rrdevice.device.create_descriptor_pool(&info, None)?;

    Ok(())
}

unsafe fn create_descriptor_sets(
    rrdevice: &RRDevice,
    rrswapchain: &RRSwapchain,
    rrdescriptor_set: &mut RRDescriptorSet,
) -> Result<()> {
    if !rrdescriptor_set.descriptor_sets.is_empty() {
        rrdevice
            .device
            .free_descriptor_sets(
                rrdescriptor_set.descriptor_pool,
                &rrdescriptor_set.descriptor_sets,
            )
            .ok();
        rrdescriptor_set.descriptor_sets.clear();
    }

    let num_sets = rrswapchain.swapchain_images.len() * rrdescriptor_set.rrdata.len().max(1);
    let layouts = vec![rrdescriptor_set.descriptor_set_layout; num_sets];
    println!("{}, {}", "layouts length", layouts.len());
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(rrdescriptor_set.descriptor_pool)
        .set_layouts(&layouts);
    rrdescriptor_set.descriptor_sets = rrdevice.device.allocate_descriptor_sets(&info)?;

    for j in 0..rrdescriptor_set.rrdata.len() {
        let rrdata = &rrdescriptor_set.rrdata[j];

        for i in 0..rrswapchain.swapchain_images.len() {
            let descriptor_set_index = j * rrswapchain.swapchain_images.len() + i;

            let info = vk::DescriptorBufferInfo::builder()
                .buffer(rrdata.rruniform_buffers[i].buffer)
                .offset(0)
                .range(size_of::<UniformBufferObject>() as u64);

            let buffer_info = &[info];

            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(rrdescriptor_set.descriptor_sets[descriptor_set_index])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info);

            if rrdata.image_view != vk::ImageView::null() && rrdata.sampler != vk::Sampler::null() {
                let info = vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(rrdata.image_view)
                    .sampler(rrdata.sampler);
                let image_info = &[info];

                let sampler_write = vk::WriteDescriptorSet::builder()
                    .dst_set(rrdescriptor_set.descriptor_sets[descriptor_set_index])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(image_info);

                rrdevice.device.update_descriptor_sets(
                    &[ubo_write, sampler_write],
                    &[] as &[vk::CopyDescriptorSet],
                );
            } else {
                rrdevice
                    .device
                    .update_descriptor_sets(&[ubo_write], &[] as &[vk::CopyDescriptorSet]);
            }
        }
    }

    Ok(())
}
