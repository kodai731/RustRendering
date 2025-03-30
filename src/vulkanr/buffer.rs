use super::device::*;
use super::vulkan::*;
use crate::vulkanr::command::*;
use crate::vulkanr::data::UniformBufferObject;
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

#[derive(Clone, Debug, Default)]
pub struct RRUniformBuffer {
    pub buffer: vk::Buffer,
    pub buffer_memory: vk::DeviceMemory,
    pub uniform_buffer_object: UniformBufferObject,
}

impl RRUniformBuffer {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        uniform_buffer_object: UniformBufferObject,
    ) -> Self {
        let mut rruniform_buffer = Self::default();
        let Ok((uniform_buffer, uniform_buffer_memory)) = create_buffer(
            instance,
            rrdevice,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) else {
            panic!("Unable to create uniform buffer");
        };

        rruniform_buffer.buffer = uniform_buffer;
        rruniform_buffer.buffer_memory = uniform_buffer_memory;
        rruniform_buffer.uniform_buffer_object = uniform_buffer_object;
        rruniform_buffer
    }
}

// unsafe fn create_uniform_buffers(
//     instance: &Instance,
//     rrdevice: &RRDevice,
//     rrswapchain: &RRSwapchain,
// ) -> Result<()> {
//     data.uniform_buffers.clear();
//     data.uniform_buffer_memories.clear();
//
//     for _ in 0..data.swapchain_images.len() {
//         let (uniform_buffer, uniform_buffer_memory) = Self::create_buffer(
//             instance,
//             device,
//             data,
//             size_of::<UniformBufferObject>() as u64,
//             vk::BufferUsageFlags::UNIFORM_BUFFER,
//             vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
//         )?;
//         data.uniform_buffers.push(uniform_buffer);
//         data.uniform_buffer_memories.push(uniform_buffer_memory);
//     }
//
//     Ok(())
// }

#[derive(Clone, Debug, Default)]
pub struct RRIndexBuffer {
    pub buffer: vk::Buffer,
    pub buffer_memory: vk::DeviceMemory,
    pub indices: u32,
}

impl RRIndexBuffer {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        rr_command_pool: &RRCommandPool,
        size: u64,
        data: *const c_void,
        length: usize,
    ) -> Self {
        let mut rrindex_buffer = RRIndexBuffer::default();
        let Ok((staging_buffer, staging_buffer_memory)) = create_buffer(
            instance,
            rrdevice,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        ) else {
            panic!("Unable to create buffer");
        };
        let Ok(map_memory) =
            rrdevice
                .device
                .map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())
        else {
            panic!("Failed to map staging buffer");
        };

        memcpy(data, map_memory.cast(), length);
        rrdevice.device.unmap_memory(staging_buffer_memory);

        let Ok((index_buffer, index_buffer_memory)) = create_buffer(
            instance,
            rrdevice,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, //  we're not able to use map_memory, instead can be copied
        ) else {
            panic!("failed to create buffer")
        };

        copy_buffer(
            rrdevice,
            rr_command_pool,
            staging_buffer,
            index_buffer,
            size,
        )
        .expect("failed to create buffer");

        rrdevice.device.destroy_buffer(staging_buffer, None);
        rrdevice.device.free_memory(staging_buffer_memory, None);

        rrindex_buffer.buffer = index_buffer;
        rrindex_buffer.buffer_memory = index_buffer_memory;
        rrindex_buffer.indices = length as u32;

        rrindex_buffer
    }
}

#[derive(Clone, Debug, Default)]
pub struct RRVertexBuffer {
    pub buffer: vk::Buffer,
    pub buffer_memory: vk::DeviceMemory,
    pub vertices: u32,
}

impl RRVertexBuffer {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        rr_command_pool: &RRCommandPool,
        size: u64,
        data: *const c_void,
        length: usize,
    ) -> Self {
        let mut rrvertex_buffer = RRVertexBuffer::default();
        let Ok((staging_buffer, staging_buffer_memory)) = create_buffer(
            instance,
            rrdevice,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        ) else {
            panic!("failed to create buffer");
        };
        println!("created staging buffer");
        let Ok(map_memory) =
            rrdevice
                .device
                .map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())
        else {
            panic!("failed to map buffer")
        };

        memcpy(data, map_memory.cast(), length);
        rrdevice.device.unmap_memory(staging_buffer_memory);
        println!("mapped staging buffer");

        let Ok((vertex_buffer, vertex_buffer_memory)) = create_buffer(
            instance,
            rrdevice,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, //  we're not able to use map_memory, instead can be copied
        ) else {
            panic!("failed to create buffer");
        };
        println!("created vertex buffer");

        if let Err(e) = copy_buffer(
            rrdevice,
            rr_command_pool,
            staging_buffer,
            vertex_buffer,
            size,
        ) {
            panic!("failed to copy buffer");
        }

        rrdevice.device.destroy_buffer(staging_buffer, None);
        rrdevice.device.free_memory(staging_buffer_memory, None);

        rrvertex_buffer.buffer = vertex_buffer;
        rrvertex_buffer.buffer_memory = vertex_buffer_memory;
        rrvertex_buffer.vertices = length as u32;

        rrvertex_buffer
    }
}

#[derive(Clone, Debug, Default)]
pub struct RRBuffer {
    pub buffer: vk::Buffer,
    pub buffer_memory: vk::DeviceMemory,
    pub indices: u32,
}

impl RRBuffer {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Self {
        let mut rrbuffer = RRBuffer::default();
        let Ok((buffer, buffer_memory)) =
            create_buffer(instance, rrdevice, size, usage, properties)
        else {
            panic!("failed to create buffer")
        };
        rrbuffer.buffer = buffer;
        rrbuffer.buffer_memory = buffer_memory;
        rrbuffer
    }
}

pub unsafe fn create_buffer(
    instance: &Instance,
    rrdevice: &RRDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = rrdevice.device.create_buffer(&buffer_info, None)?;
    let requirements = rrdevice.device.get_buffer_memory_requirements(buffer);
    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            rrdevice.physical_device,
            properties,
            requirements,
        )?);
    let buffer_memory = rrdevice.device.allocate_memory(&memory_info, None)?;
    rrdevice
        .device
        .bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

pub unsafe fn copy_buffer(
    rrdevice: &RRDevice,
    rrcommand_pool: &RRCommandPool,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(rrdevice, rrcommand_pool.command_pool)?;
    let regions = vk::BufferCopy::builder().size(size);
    rrdevice
        .device
        .cmd_copy_buffer(command_buffer, source, destination, &[regions]);
    end_single_time_commands(
        rrdevice,
        rrdevice.graphics_queue,
        rrcommand_pool.command_pool,
        command_buffer,
    )?;

    Ok(())
}

pub unsafe fn copy_buffer_to_image(
    rrdevice: &RRDevice,
    rrcommand_buffer: &RRCommandPool,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(rrdevice, rrcommand_buffer.command_pool)?;
    let subresources = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(subresources)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    //  has already been transitioned to the layout that is optimal for copying pixels
    rrdevice.device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    end_single_time_commands(
        rrdevice,
        rrdevice.graphics_queue,
        rrcommand_buffer.command_pool,
        command_buffer,
    )?;

    Ok(())
}
