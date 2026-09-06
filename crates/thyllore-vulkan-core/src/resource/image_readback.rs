use crate::command::*;
use crate::core::device::*;
use crate::resource::buffer::create_buffer;
use crate::vulkan::*;

pub unsafe fn copy_image_to_host_buffer(
    instance: &Instance,
    rrdevice: &RRDevice,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    image: vk::Image,
    width: u32,
    height: u32,
    image_size: vk::DeviceSize,
    layout: vk::ImageLayout,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let (buffer, buffer_memory) = create_buffer(
        instance,
        rrdevice,
        image_size,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let command_buffer = begin_single_time_commands(rrdevice, command_pool)?;
    record_image_to_buffer_copy(
        &rrdevice.device,
        command_buffer,
        image,
        buffer,
        width,
        height,
        layout,
    );
    end_single_time_commands(rrdevice, queue, command_pool, command_buffer)?;

    Ok((buffer, buffer_memory))
}

unsafe fn record_image_to_buffer_copy(
    device: &crate::core::device::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    buffer: vk::Buffer,
    width: u32,
    height: u32,
    layout: vk::ImageLayout,
) {
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    let barrier_to_transfer = vk::ImageMemoryBarrier::builder()
        .old_layout(layout)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier_to_transfer.build()],
    );

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });
    device.cmd_copy_image_to_buffer(
        command_buffer,
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        buffer,
        &[region.build()],
    );

    let barrier_back = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ);
    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier_back.build()],
    );
}
