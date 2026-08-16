use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::core::RRDevice;
use crate::resource::buffer::create_buffer;
use crate::resource::image::{create_image, create_image_view, transition_image_layout};

/// Object id at offset 0, world position at `READBACK_POSITION_OFFSET`. Picking reads both from
/// the same pixel so a click can compare what the flame covers against what the surface covers.
pub const READBACK_POSITION_OFFSET: vk::DeviceSize = 16;
pub const READBACK_STAGING_SIZE: vk::DeviceSize = READBACK_POSITION_OFFSET + 16;

#[derive(Clone, Debug, Default)]
pub struct RRGBuffer {
    pub position_image: vk::Image,
    pub position_image_memory: vk::DeviceMemory,
    pub position_image_view: vk::ImageView,

    pub normal_image: vk::Image,
    pub normal_image_memory: vk::DeviceMemory,
    pub normal_image_view: vk::ImageView,

    pub albedo_image: vk::Image,
    pub albedo_image_memory: vk::DeviceMemory,
    pub albedo_image_view: vk::ImageView,

    pub object_id_image: vk::Image,
    pub object_id_image_memory: vk::DeviceMemory,
    pub object_id_image_view: vk::ImageView,

    pub shadow_mask_image: vk::Image,
    pub shadow_mask_image_memory: vk::DeviceMemory,
    pub shadow_mask_image_view: vk::ImageView,

    pub readback_staging_buffer: vk::Buffer,
    pub readback_staging_memory: vk::DeviceMemory,

    pub width: u32,
    pub height: u32,
}

impl RRGBuffer {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let (position_image, position_image_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            vk::SampleCountFlags::_1,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let position_image_view = create_image_view(
            rrdevice,
            position_image,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        let (normal_image, normal_image_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            vk::SampleCountFlags::_1,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let normal_image_view = create_image_view(
            rrdevice,
            normal_image,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        let (albedo_image, albedo_image_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            vk::SampleCountFlags::_1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let albedo_image_view = create_image_view(
            rrdevice,
            albedo_image,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        let (object_id_image, object_id_image_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            vk::SampleCountFlags::_1,
            vk::Format::R32_UINT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let object_id_image_view = create_image_view(
            rrdevice,
            object_id_image,
            vk::Format::R32_UINT,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        let (shadow_mask_image, shadow_mask_image_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            vk::SampleCountFlags::_1,
            vk::Format::R32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let shadow_mask_image_view = create_image_view(
            rrdevice,
            shadow_mask_image,
            vk::Format::R32_SFLOAT,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        let (readback_staging_buffer, readback_staging_memory) = create_buffer(
            instance,
            rrdevice,
            READBACK_STAGING_SIZE,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        log::info!(
            "Created G-Buffer: {}x{} (position, normal, albedo, object_id, shadow mask)",
            width,
            height
        );

        Ok(Self {
            position_image,
            position_image_memory,
            position_image_view,
            normal_image,
            normal_image_memory,
            normal_image_view,
            albedo_image,
            albedo_image_memory,
            albedo_image_view,
            object_id_image,
            object_id_image_memory,
            object_id_image_view,
            shadow_mask_image,
            shadow_mask_image_memory,
            shadow_mask_image_view,
            readback_staging_buffer,
            readback_staging_memory,
            width,
            height,
        })
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
        if new_width == 0 || new_height == 0 {
            return Ok(());
        }

        self.destroy(rrdevice);
        *self = RRGBuffer::new(instance, rrdevice, new_width, new_height)?;
        Ok(())
    }

    pub unsafe fn destroy(&mut self, rrdevice: &RRDevice) {
        destroy_image_set(
            &rrdevice.device,
            &mut self.position_image_view,
            &mut self.position_image,
            &mut self.position_image_memory,
        );
        destroy_image_set(
            &rrdevice.device,
            &mut self.normal_image_view,
            &mut self.normal_image,
            &mut self.normal_image_memory,
        );
        destroy_image_set(
            &rrdevice.device,
            &mut self.albedo_image_view,
            &mut self.albedo_image,
            &mut self.albedo_image_memory,
        );
        destroy_image_set(
            &rrdevice.device,
            &mut self.object_id_image_view,
            &mut self.object_id_image,
            &mut self.object_id_image_memory,
        );
        destroy_image_set(
            &rrdevice.device,
            &mut self.shadow_mask_image_view,
            &mut self.shadow_mask_image,
            &mut self.shadow_mask_image_memory,
        );

        if self.readback_staging_buffer != vk::Buffer::null() {
            rrdevice
                .device
                .destroy_buffer(self.readback_staging_buffer, None);
            self.readback_staging_buffer = vk::Buffer::null();
        }
        if self.readback_staging_memory != vk::DeviceMemory::null() {
            rrdevice
                .device
                .free_memory(self.readback_staging_memory, None);
            self.readback_staging_memory = vk::DeviceMemory::null();
        }

        log!("Destroyed G-Buffer");
    }

    pub unsafe fn transition_layouts(
        &self,
        rrdevice: &RRDevice,
        command_pool: vk::CommandPool,
    ) -> Result<()> {
        transition_image_layout(
            rrdevice,
            rrdevice.graphics_queue,
            command_pool,
            self.position_image,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            1,
        )?;

        transition_image_layout(
            rrdevice,
            rrdevice.graphics_queue,
            command_pool,
            self.normal_image,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            1,
        )?;

        transition_image_layout(
            rrdevice,
            rrdevice.graphics_queue,
            command_pool,
            self.albedo_image,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            1,
        )?;

        transition_image_layout(
            rrdevice,
            rrdevice.graphics_queue,
            command_pool,
            self.object_id_image,
            vk::Format::R32_UINT,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            1,
        )?;

        transition_image_layout(
            rrdevice,
            rrdevice.graphics_queue,
            command_pool,
            self.shadow_mask_image,
            vk::Format::R32_SFLOAT,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            1,
        )?;

        Ok(())
    }
}

unsafe fn destroy_image_set(
    device: &vulkanalia::Device,
    view: &mut vk::ImageView,
    image: &mut vk::Image,
    memory: &mut vk::DeviceMemory,
) {
    if *view != vk::ImageView::null() {
        device.destroy_image_view(*view, None);
        *view = vk::ImageView::null();
    }
    if *image != vk::Image::null() {
        device.destroy_image(*image, None);
        *image = vk::Image::null();
    }
    if *memory != vk::DeviceMemory::null() {
        device.free_memory(*memory, None);
        *memory = vk::DeviceMemory::null();
    }
}

impl Drop for RRGBuffer {
    fn drop(&mut self) {
        if self.position_image != vk::Image::null() {
            log_warn!("RRGBuffer dropped without calling destroy()");
        }
    }
}
