use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::resource::buffer::create_buffer;
use crate::vulkan::*;

/// Per-instance hit shading record matching the GLSL `HitShadingRecord` layout.
/// repr(C) ensures the Rust and GLSL std430 layouts match:
///   vertex_address: u64  (offset 0,   8 bytes)
///   index_address: u64   (offset 8,   8 bytes)
///   model: [[f32;4];4]   (offset 16,  64 bytes)
///   normal_matrix: [[f32;4];4] (offset 80, 64 bytes)
///   base_color: [f32;4]  (offset 144, 16 bytes)
/// Total: 160 bytes per record.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HitShadingRecord {
    pub vertex_address: u64,
    pub index_address: u64,
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
    pub base_color: [f32; 4],
}

impl HitShadingRecord {
    pub fn default_record() -> Self {
        Self {
            vertex_address: 0,
            index_address: 0,
            model: [[0.0; 4]; 4],
            normal_matrix: [[0.0; 4]; 4],
            base_color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}
/// HitShadingTable: a storage buffer holding an array of `HitShadingRecord`.
#[derive(Clone, Debug)]
pub struct HitShadingTable {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub capacity: usize,
}

impl HitShadingTable {
    /// Create a new HitShadingTable with capacity for `capacity` instances.
    pub unsafe fn new(instance: &Instance, rrdevice: &RRDevice, capacity: usize) -> Result<Self> {
        let size = (std::mem::size_of::<HitShadingRecord>() * capacity) as vk::DeviceSize;

        let (buffer, memory) = create_buffer(
            instance,
            rrdevice,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        Ok(Self {
            buffer,
            memory,
            capacity,
        })
    }

    /// Upload records to the buffer via memory mapping.
    pub unsafe fn upload(&self, rrdevice: &RRDevice, records: &[HitShadingRecord]) -> Result<()> {
        let record_size = std::mem::size_of::<HitShadingRecord>();
        let data_len = records.len() * record_size;
        let data = std::slice::from_raw_parts(records.as_ptr() as *const u8, data_len);

        let mapped = rrdevice.device.map_memory(
            self.memory,
            0,
            data.len() as vk::DeviceSize,
            vk::MemoryMapFlags::empty(),
        )?;

        std::ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, data.len());

        rrdevice.device.unmap_memory(self.memory);

        Ok(())
    }

    /// Get the Vulkan buffer handle.
    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// Destroy the underlying buffer and memory.
    pub unsafe fn destroy(mut self, rrdevice: &RRDevice) {
        if self.buffer != vk::Buffer::null() {
            rrdevice.device.destroy_buffer(self.buffer, None);
            self.buffer = vk::Buffer::null();
        }
        if self.memory != vk::DeviceMemory::null() {
            rrdevice.device.free_memory(self.memory, None);
            self.memory = vk::DeviceMemory::null();
        }
    }
}
