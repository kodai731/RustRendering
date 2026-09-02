use crate::command::*;
use crate::core::device::*;
use crate::data::UniformBufferObject;
use crate::vulkan::*;
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
        name: &str,
    ) -> Result<Self> {
        let mut rruniform_buffer = Self::default();
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            rrdevice,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        log!(
            "RRUniformBuffer::new [{}] buffer={:?}, memory={:?}",
            name,
            uniform_buffer,
            uniform_buffer_memory
        );

        rruniform_buffer.buffer = uniform_buffer;
        rruniform_buffer.buffer_memory = uniform_buffer_memory;
        rruniform_buffer.uniform_buffer_object = uniform_buffer_object;
        Ok(rruniform_buffer)
    }

    pub unsafe fn update(
        &mut self,
        rrdevice: &RRDevice,
        ubo: &UniformBufferObject,
    ) -> anyhow::Result<()> {
        let memory = rrdevice.device.map_memory(
            self.buffer_memory,
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(ubo, memory.cast(), 1);
        rrdevice.device.unmap_memory(self.buffer_memory);

        self.uniform_buffer_object = ubo.clone();

        Ok(())
    }

    pub unsafe fn destroy(&mut self, rrdevice: &RRDevice) {
        if self.buffer != vk::Buffer::null() {
            rrdevice.device.destroy_buffer(self.buffer, None);
            self.buffer = vk::Buffer::null();
        }
        if self.buffer_memory != vk::DeviceMemory::null() {
            rrdevice.device.free_memory(self.buffer_memory, None);
            self.buffer_memory = vk::DeviceMemory::null();
        }
    }
}

impl Drop for RRUniformBuffer {
    fn drop(&mut self) {
        if self.buffer != vk::Buffer::null() {
            log_warn!("RRUniformBuffer dropped without calling destroy()");
        }
    }
}

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
    ) -> Result<Self> {
        let mut rrindex_buffer = RRIndexBuffer::default();
        let (staging_buffer, staging_buffer_memory) = create_buffer(
            instance,
            rrdevice,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        let map_memory = rrdevice.device.map_memory(
            staging_buffer_memory,
            0,
            size,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(data, map_memory.cast(), size as usize);
        rrdevice.device.unmap_memory(staging_buffer_memory);

        let (index_buffer, index_buffer_memory) = create_buffer(
            instance,
            rrdevice,
            size,
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        copy_buffer(
            rrdevice,
            rr_command_pool,
            staging_buffer,
            index_buffer,
            size,
        )?;

        rrdevice.device.destroy_buffer(staging_buffer, None);
        rrdevice.device.free_memory(staging_buffer_memory, None);

        rrindex_buffer.buffer = index_buffer;
        rrindex_buffer.buffer_memory = index_buffer_memory;
        rrindex_buffer.indices = length as u32;

        Ok(rrindex_buffer)
    }

    pub unsafe fn destroy(&mut self, rrdevice: &RRDevice) {
        if self.buffer != vk::Buffer::null() {
            rrdevice.device.destroy_buffer(self.buffer, None);
            self.buffer = vk::Buffer::null();
        }
        if self.buffer_memory != vk::DeviceMemory::null() {
            rrdevice.device.free_memory(self.buffer_memory, None);
            self.buffer_memory = vk::DeviceMemory::null();
        }
        self.indices = 0;
    }
}

impl Drop for RRIndexBuffer {
    fn drop(&mut self) {
        if self.buffer != vk::Buffer::null() {
            log_warn!("RRIndexBuffer dropped without calling destroy()");
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RRVertexBuffer {
    pub buffer: vk::Buffer,
    pub buffer_memory: vk::DeviceMemory,
    pub vertices: u32,
    pub staging_buffer: vk::Buffer,
    pub staging_memory: vk::DeviceMemory,
    pub staging_capacity: u64,
}

impl RRVertexBuffer {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        rr_command_pool: &RRCommandPool,
        size: u64,
        data: *const c_void,
        length: usize,
    ) -> Result<Self> {
        let mut rrvertex_buffer = RRVertexBuffer::default();
        let (staging_buffer, staging_buffer_memory) = create_buffer(
            instance,
            rrdevice,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        let map_memory = rrdevice.device.map_memory(
            staging_buffer_memory,
            0,
            size,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(data, map_memory.cast(), size as usize);
        rrdevice.device.unmap_memory(staging_buffer_memory);

        let (vertex_buffer, vertex_buffer_memory) = create_buffer(
            instance,
            rrdevice,
            size,
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        copy_buffer(
            rrdevice,
            rr_command_pool,
            staging_buffer,
            vertex_buffer,
            size,
        )?;

        rrvertex_buffer.buffer = vertex_buffer;
        rrvertex_buffer.buffer_memory = vertex_buffer_memory;
        rrvertex_buffer.vertices = length as u32;
        rrvertex_buffer.staging_buffer = staging_buffer;
        rrvertex_buffer.staging_memory = staging_buffer_memory;
        rrvertex_buffer.staging_capacity = size;

        Ok(rrvertex_buffer)
    }

    pub unsafe fn destroy(&mut self, rrdevice: &RRDevice) {
        if self.staging_buffer != vk::Buffer::null() {
            rrdevice.device.destroy_buffer(self.staging_buffer, None);
            self.staging_buffer = vk::Buffer::null();
        }
        if self.staging_memory != vk::DeviceMemory::null() {
            rrdevice.device.free_memory(self.staging_memory, None);
            self.staging_memory = vk::DeviceMemory::null();
        }
        self.staging_capacity = 0;
        if self.buffer != vk::Buffer::null() {
            rrdevice.device.destroy_buffer(self.buffer, None);
            self.buffer = vk::Buffer::null();
        }
        if self.buffer_memory != vk::DeviceMemory::null() {
            rrdevice.device.free_memory(self.buffer_memory, None);
            self.buffer_memory = vk::DeviceMemory::null();
        }
        self.vertices = 0;
    }
}

impl Drop for RRVertexBuffer {
    fn drop(&mut self) {
        if self.buffer != vk::Buffer::null() {
            log_warn!("RRVertexBuffer dropped without calling destroy()");
        }
        if self.staging_buffer != vk::Buffer::null() {
            log_warn!(
                "RRVertexBuffer dropped without calling destroy() (staging buffer not freed)"
            );
        }
    }
}

impl RRVertexBuffer {
    pub unsafe fn update(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        rr_command_pool: &RRCommandPool,
        size: u64,
        data: *const c_void,
        length: usize,
    ) -> Result<()> {
        // Allocate or grow staging buffer if needed
        let (staging_buffer, staging_memory) =
            if self.staging_buffer == vk::Buffer::null() || size > self.staging_capacity {
                // Free old staging buffer if it exists
                if self.staging_buffer != vk::Buffer::null() {
                    rrdevice.device.destroy_buffer(self.staging_buffer, None);
                }
                if self.staging_memory != vk::DeviceMemory::null() {
                    rrdevice.device.free_memory(self.staging_memory, None);
                }
                let new_capacity = size * 2;
                let (buf, mem) = create_buffer(
                    instance,
                    rrdevice,
                    new_capacity,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                )?;
                self.staging_buffer = buf;
                self.staging_memory = mem;
                self.staging_capacity = new_capacity;
                (buf, mem)
            } else {
                (self.staging_buffer, self.staging_memory)
            };

        let map_memory =
            rrdevice
                .device
                .map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())?;

        memcpy(data, map_memory.cast(), size as usize);
        rrdevice.device.unmap_memory(staging_memory);

        copy_buffer(rrdevice, rr_command_pool, staging_buffer, self.buffer, size)?;

        self.vertices = length as u32;

        Ok(())
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
    ) -> Result<Self> {
        let mut rrbuffer = RRBuffer::default();
        let (buffer, buffer_memory) = create_buffer(instance, rrdevice, size, usage, properties)?;
        rrbuffer.buffer = buffer;
        rrbuffer.buffer_memory = buffer_memory;
        Ok(rrbuffer)
    }

    pub unsafe fn destroy(&mut self, rrdevice: &RRDevice) {
        if self.buffer != vk::Buffer::null() {
            rrdevice.device.destroy_buffer(self.buffer, None);
            self.buffer = vk::Buffer::null();
        }
        if self.buffer_memory != vk::DeviceMemory::null() {
            rrdevice.device.free_memory(self.buffer_memory, None);
            self.buffer_memory = vk::DeviceMemory::null();
        }
        self.indices = 0;
    }
}

impl Drop for RRBuffer {
    fn drop(&mut self) {
        if self.buffer != vk::Buffer::null() {
            log_warn!("RRBuffer dropped without calling destroy()");
        }
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

    let mut memory_allocate_flags_info =
        vk::MemoryAllocateFlagsInfo::builder().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

    let mut memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            rrdevice.physical_device,
            properties,
            requirements,
        )?);

    if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
        memory_info = memory_info.push_next(&mut memory_allocate_flags_info);
    }

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
