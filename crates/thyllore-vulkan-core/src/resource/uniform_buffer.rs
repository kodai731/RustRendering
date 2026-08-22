use std::marker::PhantomData;

use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::resource::buffer::create_buffer;
use crate::vulkan::Instance;
use thyllore_spirv_reflect::GpuBlock;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Placement {
    #[default]
    HostMapped,
    DeviceUpdated,
}

#[derive(Clone, Debug, Default)]
pub struct UniformBuffer<T: GpuBlock> {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    slot_stride: vk::DeviceSize,
    slot_count: usize,
    placement: Placement,
    _block: PhantomData<T>,
}

impl<T: GpuBlock> UniformBuffer<T> {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        slot_count: usize,
        placement: Placement,
    ) -> Result<Self> {
        if slot_count == 0 {
            return Err(anyhow!(
                "{} uniform buffer needs at least one slot",
                T::NAME
            ));
        }
        let alignment = rrdevice.min_uniform_buffer_offset_alignment.max(1);
        let slot_stride = (T::SIZE as vk::DeviceSize).div_ceil(alignment) * alignment;

        let usage = match placement {
            Placement::HostMapped => vk::BufferUsageFlags::UNIFORM_BUFFER,
            Placement::DeviceUpdated => {
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
        };
        let (buffer, memory) = create_buffer(
            instance,
            rrdevice,
            slot_stride * slot_count as vk::DeviceSize,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        Ok(Self {
            buffer,
            memory,
            slot_stride,
            slot_count,
            placement,
            _block: PhantomData,
        })
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn block_size(&self) -> vk::DeviceSize {
        T::SIZE as vk::DeviceSize
    }

    pub fn slot_offset(&self, slot: usize) -> Result<vk::DeviceSize> {
        if slot >= self.slot_count {
            return Err(anyhow!(
                "{} slot {slot} is out of range (slot count {})",
                T::NAME,
                self.slot_count
            ));
        }
        Ok(self.slot_stride * slot as vk::DeviceSize)
    }

    pub unsafe fn write_slot(&self, rrdevice: &RRDevice, slot: usize, value: &T) -> Result<()> {
        let offset = self.slot_offset(slot)?;
        let bytes = value.as_bytes();
        let mapped = rrdevice.device.map_memory(
            self.memory,
            offset,
            bytes.len() as vk::DeviceSize,
            vk::MemoryMapFlags::empty(),
        )?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
        rrdevice.device.unmap_memory(self.memory);
        Ok(())
    }

    /// Transfers inside the command buffer so frames in flight never race on one slot.
    pub unsafe fn record_update(
        &self,
        device: &vulkanalia::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        value: &T,
        reader_stage: vk::PipelineStageFlags,
    ) -> Result<()> {
        if self.placement != Placement::DeviceUpdated {
            return Err(anyhow!(
                "{} uniform buffer is {:?}; record_update needs Placement::DeviceUpdated",
                T::NAME,
                self.placement
            ));
        }
        let offset = self.slot_offset(slot)?;

        self.record_barrier(
            device,
            cmd,
            offset,
            (reader_stage, vk::AccessFlags::UNIFORM_READ),
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            ),
        );
        device.cmd_update_buffer(cmd, self.buffer, offset, value.as_bytes());
        self.record_barrier(
            device,
            cmd,
            offset,
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            ),
            (reader_stage, vk::AccessFlags::UNIFORM_READ),
        );
        Ok(())
    }

    unsafe fn record_barrier(
        &self,
        device: &vulkanalia::Device,
        cmd: vk::CommandBuffer,
        offset: vk::DeviceSize,
        (src_stage, src_access): (vk::PipelineStageFlags, vk::AccessFlags),
        (dst_stage, dst_access): (vk::PipelineStageFlags, vk::AccessFlags),
    ) {
        let barrier = vk::BufferMemoryBarrier::builder()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.buffer)
            .offset(offset)
            .size(self.block_size())
            .build();
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[barrier],
            &[] as &[vk::ImageMemoryBarrier],
        );
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.buffer != vk::Buffer::null() {
            device.destroy_buffer(self.buffer, None);
            self.buffer = vk::Buffer::null();
        }
        if self.memory != vk::DeviceMemory::null() {
            device.free_memory(self.memory, None);
            self.memory = vk::DeviceMemory::null();
        }
    }
}
