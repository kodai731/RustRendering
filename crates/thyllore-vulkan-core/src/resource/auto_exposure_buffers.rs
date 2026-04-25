use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::core::RRDevice;
use crate::resource::buffer::create_buffer;
use crate::vulkan::*;

const HISTOGRAM_BIN_COUNT: u32 = 256;
const HISTOGRAM_BUFFER_SIZE: u64 = (HISTOGRAM_BIN_COUNT * std::mem::size_of::<u32>() as u32) as u64;
const LUMINANCE_BUFFER_SIZE: u64 = (2 * std::mem::size_of::<f32>() as u32) as u64;

#[derive(Clone, Debug, Default)]
pub struct AutoExposureBuffers {
    pub histogram_buffer: vk::Buffer,
    pub histogram_buffer_memory: vk::DeviceMemory,
    pub luminance_buffer: vk::Buffer,
    pub luminance_buffer_memory: vk::DeviceMemory,
    pub width: u32,
    pub height: u32,
}

impl AutoExposureBuffers {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let (histogram_buffer, histogram_buffer_memory) = create_buffer(
            instance,
            rrdevice,
            HISTOGRAM_BUFFER_SIZE,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let (luminance_buffer, luminance_buffer_memory) = create_buffer(
            instance,
            rrdevice,
            LUMINANCE_BUFFER_SIZE,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        Self::zero_luminance_buffer(&rrdevice.device, luminance_buffer_memory)?;

        log!("Created AutoExposure buffers: {}x{}", width, height);

        Ok(Self {
            histogram_buffer,
            histogram_buffer_memory,
            luminance_buffer,
            luminance_buffer_memory,
            width,
            height,
        })
    }

    unsafe fn zero_luminance_buffer(
        device: &vulkanalia::Device,
        memory: vk::DeviceMemory,
    ) -> Result<()> {
        let data = device.map_memory(
            memory,
            0,
            LUMINANCE_BUFFER_SIZE,
            vk::MemoryMapFlags::empty(),
        )?;

        std::ptr::write_bytes(data as *mut u8, 0, LUMINANCE_BUFFER_SIZE as usize);

        device.unmap_memory(memory);
        Ok(())
    }

    pub unsafe fn read_adapted_exposure(&self, device: &vulkanalia::Device) -> f32 {
        let data = match device.map_memory(
            self.luminance_buffer_memory,
            0,
            LUMINANCE_BUFFER_SIZE,
            vk::MemoryMapFlags::empty(),
        ) {
            Ok(ptr) => ptr,
            Err(_) => return 0.0,
        };

        let values = data as *const f32;
        let adapted = *values.add(1);

        device.unmap_memory(self.luminance_buffer_memory);
        adapted
    }

    pub unsafe fn resize(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        new_width: u32,
        new_height: u32,
    ) -> Result<()> {
        if new_width == self.width && new_height == self.height {
            return Ok(());
        }

        self.destroy(&rrdevice.device);
        let new_buf = Self::new(instance, rrdevice, new_width, new_height)?;
        *self = new_buf;

        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.histogram_buffer != vk::Buffer::null() {
            device.destroy_buffer(self.histogram_buffer, None);
            self.histogram_buffer = vk::Buffer::null();
        }
        if self.histogram_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(self.histogram_buffer_memory, None);
            self.histogram_buffer_memory = vk::DeviceMemory::null();
        }

        if self.luminance_buffer != vk::Buffer::null() {
            device.destroy_buffer(self.luminance_buffer, None);
            self.luminance_buffer = vk::Buffer::null();
        }
        if self.luminance_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(self.luminance_buffer_memory, None);
            self.luminance_buffer_memory = vk::DeviceMemory::null();
        }

        log!("Destroyed AutoExposure buffers");
    }
}

impl Drop for AutoExposureBuffers {
    fn drop(&mut self) {
        if self.histogram_buffer != vk::Buffer::null() {
            log_warn!("AutoExposureBuffers dropped without calling destroy()");
        }
    }
}
