use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::descriptor::pass_shaders::{frame_set_shaders, FRAME_SET};
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::resource::buffer::create_buffer;
use crate::vulkan::Instance;
use thyllore_render_core::FrameUBO;

const FRAME_UBO_BINDING: u32 = 0;

#[derive(Clone, Debug, Default)]
pub struct FrameDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub pool: vk::DescriptorPool,
    pub sets: Vec<vk::DescriptorSet>,
    pub buffers: Vec<vk::Buffer>,
    pub buffer_memories: Vec<vk::DeviceMemory>,
}

impl FrameDescriptorSet {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        swapchain_image_count: usize,
    ) -> anyhow::Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let pool = layout.create_pool(
            rrdevice,
            swapchain_image_count as u32,
            vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
        )?;
        let sets = layout.allocate_sets(rrdevice, pool, swapchain_image_count)?;

        let mut buffers = Vec::with_capacity(swapchain_image_count);
        let mut buffer_memories = Vec::with_capacity(swapchain_image_count);

        for _ in 0..swapchain_image_count {
            let (buffer, memory) = create_buffer(
                instance,
                rrdevice,
                size_of::<FrameUBO>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            buffers.push(buffer);
            buffer_memories.push(memory);
        }

        let mut frame_set = Self {
            layout,
            pool,
            sets,
            buffers,
            buffer_memories,
        };
        frame_set.write_descriptor_sets(rrdevice)?;

        Ok(frame_set)
    }

    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::new(frame_set_shaders(), FRAME_SET)
    }

    unsafe fn write_descriptor_sets(&mut self, rrdevice: &RRDevice) -> anyhow::Result<()> {
        for (set, buffer) in self.sets.iter().zip(&self.buffers) {
            self.layout
                .writer(*set)
                .buffer(FRAME_UBO_BINDING, *buffer, 0, size_of::<FrameUBO>() as u64)?
                .apply(rrdevice);
        }
        Ok(())
    }

    pub unsafe fn update(
        &self,
        rrdevice: &RRDevice,
        image_index: usize,
        ubo: &FrameUBO,
    ) -> anyhow::Result<()> {
        let memory = rrdevice.device.map_memory(
            self.buffer_memories[image_index],
            0,
            size_of::<FrameUBO>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(ubo, memory.cast(), 1);
        rrdevice
            .device
            .unmap_memory(self.buffer_memories[image_index]);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        for &buffer in &self.buffers {
            device.destroy_buffer(buffer, None);
        }
        for &memory in &self.buffer_memories {
            device.free_memory(memory, None);
        }

        if !self.sets.is_empty() {
            device.free_descriptor_sets(self.pool, &self.sets).ok();
        }
        if self.pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.pool, None);
        }
        self.layout.destroy(device);
    }
}
