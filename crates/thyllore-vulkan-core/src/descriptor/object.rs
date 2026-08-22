use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::descriptor::pass_manifest::SetRole;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::model;
use crate::resource::uniform_buffer::{Placement, UniformBuffer};
use crate::vulkan::Instance;
use thyllore_render_core::ObjectUBO;

pub type ObjectId = u32;

#[derive(Clone, Debug, Default)]
pub struct ObjectDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub sets: Vec<vk::DescriptorSet>,
    pub buffers: Vec<UniformBuffer<ObjectUBO>>,
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
            self.buffers.push(UniformBuffer::new(
                instance,
                rrdevice,
                1,
                Placement::HostMapped,
            )?);
        }

        for (set, buffer) in self.sets[first_new..]
            .iter()
            .zip(&self.buffers[first_new..])
        {
            self.layout
                .writer(*set)
                .uniform(model::OBJECT, buffer, 0)?
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
        self.buffers[idx].write_slot(rrdevice, 0, ubo)
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
        for buffer in &mut self.buffers {
            buffer.destroy(device);
        }
        self.layout.destroy(device);
    }
}
