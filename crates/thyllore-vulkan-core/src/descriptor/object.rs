use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::resource::buffer::create_buffer;
use crate::vulkan::Instance;
use thyllore_render_core::ObjectUBO;

pub type ObjectId = u32;

#[derive(Clone, Debug, Default)]
pub struct ObjectDescriptorSet {
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub sets: Vec<vk::DescriptorSet>,
    pub buffers: Vec<vk::Buffer>,
    pub buffer_memories: Vec<vk::DeviceMemory>,
    pub max_objects: usize,
    next_slot: usize,
    reserved_slot_count: usize,
}

impl ObjectDescriptorSet {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        swapchain_image_count: usize,
        max_objects: usize,
    ) -> anyhow::Result<Self> {
        let layout = Self::create_layout(rrdevice)?;
        let total_sets = swapchain_image_count * max_objects;
        let pool = Self::create_pool(rrdevice, total_sets)?;
        let sets = Self::allocate_sets(rrdevice, layout, pool, total_sets)?;

        let mut buffers = Vec::with_capacity(total_sets);
        let mut buffer_memories = Vec::with_capacity(total_sets);

        for _ in 0..total_sets {
            let (buffer, memory) = create_buffer(
                instance,
                rrdevice,
                size_of::<ObjectUBO>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            buffers.push(buffer);
            buffer_memories.push(memory);
        }

        let mut object_set = Self {
            layout,
            pool,
            sets,
            buffers,
            buffer_memories,
            max_objects,
            next_slot: 0,
            reserved_slot_count: 0,
        };
        object_set.write_descriptor_sets(rrdevice, swapchain_image_count);

        Ok(object_set)
    }

    unsafe fn create_layout(rrdevice: &RRDevice) -> anyhow::Result<vk::DescriptorSetLayout> {
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let bindings = &[ubo_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

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

    #[allow(unused_variables)]
    unsafe fn write_descriptor_sets(&mut self, rrdevice: &RRDevice, swapchain_image_count: usize) {
        for (i, &set) in self.sets.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.buffers[i])
                .offset(0)
                .range(size_of::<ObjectUBO>() as u64);

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

    pub fn get_set_index(&self, image_index: usize, object_index: usize) -> usize {
        image_index * self.max_objects + object_index
    }

    pub fn allocate_slot(&mut self) -> usize {
        let slot = self.next_slot;
        if slot >= self.max_objects {
            log!(
                "[ObjectDescriptorSet] WARNING: slot {} exceeds max_objects {}. GPU buffer overflow!",
                slot, self.max_objects
            );
        }
        self.next_slot += 1;
        slot
    }

    pub fn get_next_slot(&self) -> usize {
        self.next_slot
    }

    pub fn seal_reserved_slots(&mut self) {
        self.reserved_slot_count = self.next_slot;
    }

    pub fn reset_to_reserved(&mut self) {
        self.next_slot = self.reserved_slot_count;
    }

    pub unsafe fn update(
        &self,
        rrdevice: &RRDevice,
        image_index: usize,
        object_index: usize,
        ubo: &ObjectUBO,
    ) -> anyhow::Result<()> {
        if object_index >= self.max_objects {
            anyhow::bail!(
                "object_index {} exceeds max_objects {}",
                object_index,
                self.max_objects
            );
        }
        let idx = self.get_set_index(image_index, object_index);
        let memory = rrdevice.device.map_memory(
            self.buffer_memories[idx],
            0,
            size_of::<ObjectUBO>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(ubo, memory.cast(), 1);
        rrdevice.device.unmap_memory(self.buffer_memories[idx]);
        Ok(())
    }

    pub unsafe fn ensure_capacity(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        swapchain_image_count: usize,
        required_objects: usize,
    ) -> anyhow::Result<()> {
        if required_objects <= self.max_objects {
            return Ok(());
        }

        for &buffer in &self.buffers {
            rrdevice.device.destroy_buffer(buffer, None);
        }
        for &memory in &self.buffer_memories {
            rrdevice.device.free_memory(memory, None);
        }
        if self.pool != vk::DescriptorPool::null() {
            rrdevice.device.destroy_descriptor_pool(self.pool, None);
        }

        let total_sets = swapchain_image_count * required_objects;
        self.pool = Self::create_pool(rrdevice, total_sets)?;
        self.sets = Self::allocate_sets(rrdevice, self.layout, self.pool, total_sets)?;

        self.buffers = Vec::with_capacity(total_sets);
        self.buffer_memories = Vec::with_capacity(total_sets);

        for _ in 0..total_sets {
            let (buffer, memory) = create_buffer(
                instance,
                rrdevice,
                size_of::<ObjectUBO>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            self.buffers.push(buffer);
            self.buffer_memories.push(memory);
        }

        self.max_objects = required_objects;
        self.write_descriptor_sets(rrdevice, swapchain_image_count);

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
