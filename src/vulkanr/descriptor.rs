use super::data::*;
use super::device::*;
use super::swapchain::*;
use super::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRDescriptorSet {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub rrdata: RRData,
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
}
unsafe fn create_descriptor_set_layout(
    rrdevice: &RRDevice,
    rrdescriptor_set: &mut RRDescriptorSet,
) -> Result<()> {
    // The descriptor layout specifies the types of resources that are going to be accessed by the pipeline,
    // just like a render pass specifies the types of attachments that will be accessed
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
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count((rrswapchain.swapchain_images.len() * 4) as u32); // This pool size structure is referenced by the main vk::DescriptorPoolCreateInfo
                                                                            //along with the maximum number of descriptor sets that may be allocated:

    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count((rrswapchain.swapchain_images.len() * 4) as u32);

    let pool_sizes = &[ubo_size, sampler_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets((rrswapchain.swapchain_images.len() * 4) as u32);
    rrdescriptor_set.descriptor_pool = rrdevice.device.create_descriptor_pool(&info, None)?;

    Ok(())
}

unsafe fn create_descriptor_sets(
    rrdevice: &RRDevice,
    rrswapchain: &RRSwapchain,
    rrdescriptor_set: &mut RRDescriptorSet,
) -> Result<()> {
    /*
    A descriptor is a way for shaders to freely access resources like buffers and images
    Usage of descriptors consists of three parts:

    Specify a descriptor layout during pipeline creation
    Allocate a descriptor set from a descriptor pool
    Bind the descriptor set during rendering
     */
    let layouts = vec![rrdescriptor_set.descriptor_set_layout; rrswapchain.swapchain_images.len()]; //  create one descriptor set for each swapchain image, all with the same layout.
    println!("{}, {}", "layouts length", layouts.len());
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(rrdescriptor_set.descriptor_pool)
        .set_layouts(&layouts);
    rrdescriptor_set.descriptor_sets = rrdevice.device.allocate_descriptor_sets(&info)?;

    for i in 0..rrswapchain.swapchain_images.len() {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(rrdescriptor_set.rrdata.rruniform_buffers[i].buffer)
            .offset(0)
            .range(size_of::<UniformBufferObject>() as u64);
        // The configuration of descriptors is updated using the update_descriptor_sets function,
        // which takes an array of vk::WriteDescriptorSet structs as parameter.
        let buffer_info = &[info];

        let info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(rrdescriptor_set.rrdata.image_view)
            .sampler(rrdescriptor_set.rrdata.sampler);
        let image_info = &[info];

        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(rrdescriptor_set.descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0) // Remember that descriptors can be arrays, so we also need to specify the first index in the array that we want to update
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_info);
        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(rrdescriptor_set.descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(image_info);

        rrdevice
            .device
            .update_descriptor_sets(&[ubo_write, sampler_write], &[] as &[vk::CopyDescriptorSet]);
    }

    Ok(())
}
