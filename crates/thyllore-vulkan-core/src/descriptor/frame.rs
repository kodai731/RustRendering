use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::resource::buffer::create_buffer;
use crate::vulkan::Instance;
use thyllore_render_core::FrameUBO;

#[derive(Clone, Debug, Default)]
pub struct FrameDescriptorSet {
    pub layout: vk::DescriptorSetLayout,
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
        let layout = Self::create_layout(rrdevice)?;
        let pool = Self::create_pool(rrdevice, swapchain_image_count)?;
        let sets = Self::allocate_sets(rrdevice, layout, pool, swapchain_image_count)?;

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
        frame_set.write_descriptor_sets(rrdevice);

        Ok(frame_set)
    }

    pub fn layout_bindings() -> Vec<vk::DescriptorSetLayoutBinding> {
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(
                vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::GEOMETRY
                    | vk::ShaderStageFlags::FRAGMENT,
            )
            .build();

        vec![ubo_binding]
    }

    unsafe fn create_layout(rrdevice: &RRDevice) -> anyhow::Result<vk::DescriptorSetLayout> {
        let bindings = Self::layout_bindings();
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        Ok(rrdevice.device.create_descriptor_set_layout(&info, None)?)
    }

    unsafe fn create_pool(rrdevice: &RRDevice, count: usize) -> anyhow::Result<vk::DescriptorPool> {
        let pool_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(count as u32);

        let pool_sizes = &[pool_size];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(count as u32)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        Ok(rrdevice.device.create_descriptor_pool(&info, None)?)
    }

    unsafe fn allocate_sets(
        rrdevice: &RRDevice,
        layout: vk::DescriptorSetLayout,
        pool: vk::DescriptorPool,
        count: usize,
    ) -> anyhow::Result<Vec<vk::DescriptorSet>> {
        let layouts = vec![layout; count];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        Ok(rrdevice.device.allocate_descriptor_sets(&info)?)
    }

    unsafe fn write_descriptor_sets(&mut self, rrdevice: &RRDevice) {
        for (i, &set) in self.sets.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.buffers[i])
                .offset(0)
                .range(size_of::<FrameUBO>() as u64);

            let buffer_infos = &[buffer_info];
            let write = vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_infos);

            rrdevice
                .device
                .update_descriptor_sets(&[write], &[] as &[vk::CopyDescriptorSet]);
        }
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
        if self.layout != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
