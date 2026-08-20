use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::descriptor::pass_manifest::SetRole;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::model;
use crate::resource::buffer::create_buffer;
use crate::vulkan::Instance;
use thyllore_render_core::ObjectUBO;

pub type ObjectId = u32;

#[derive(Clone, Debug, Default)]
pub struct ObjectDescriptorSet {
    pub layout: ReflectedSetLayout,
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
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let mut object_set = Self {
            layout,
            sets: Vec::new(),
            buffers: Vec::new(),
            buffer_memories: Vec::new(),
            max_objects,
            next_slot: 0,
            reserved_slot_count: 0,
        };
        object_set.grow_to(instance, rrdevice, swapchain_image_count * max_objects)?;

        Ok(object_set)
    }

    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::shared(SetRole::Object)
    }

    unsafe fn grow_to(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        total_sets: usize,
    ) -> anyhow::Result<()> {
        let first_new = self.sets.len();
        let additional = total_sets.saturating_sub(first_new);
        if additional == 0 {
            return Ok(());
        }

        self.sets
            .extend(self.layout.allocate_sets(rrdevice, additional)?);
        for _ in 0..additional {
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

        for (set, buffer) in self.sets[first_new..]
            .iter()
            .zip(&self.buffers[first_new..])
        {
            self.layout
                .writer(*set)
                .buffer(model::OBJECT, *buffer, 0, size_of::<ObjectUBO>() as u64)?
                .apply(rrdevice);
        }
        Ok(())
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

        self.grow_to(instance, rrdevice, swapchain_image_count * required_objects)?;
        self.max_objects = required_objects;
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        for &buffer in &self.buffers {
            device.destroy_buffer(buffer, None);
        }
        for &memory in &self.buffer_memories {
            device.free_memory(memory, None);
        }
        self.layout.destroy(device);
    }
}
