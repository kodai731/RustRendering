use std::collections::HashMap;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::descriptor::pass_shaders::{standard_graphics_shaders, MATERIAL_SET};
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::resource::buffer::create_buffer;
use crate::resource::image::RRImage;
use crate::vulkan::Instance;
use thyllore_render_core::MaterialUBO;

pub type MaterialId = u32;

const MATERIAL_TEXTURE_BINDING: u32 = 0;
const MATERIAL_UBO_BINDING: u32 = 1;

#[derive(Clone, Debug)]
pub struct Material {
    pub id: MaterialId,
    pub name: String,
    pub descriptor_set: vk::DescriptorSet,
    pub textures: Vec<RRImage>,
    pub uniform_buffer: vk::Buffer,
    pub uniform_buffer_memory: vk::DeviceMemory,
    pub properties: MaterialUBO,
}

#[derive(Clone, Debug, Default)]
pub struct MaterialManager {
    pub layout: ReflectedSetLayout,
    pub pool: vk::DescriptorPool,
    pub materials: HashMap<MaterialId, Material>,
    next_id: MaterialId,
    capacity: u32,
}

impl MaterialManager {
    pub unsafe fn new(rrdevice: &RRDevice, max_materials: u32) -> anyhow::Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let pool = layout.create_pool(
            rrdevice,
            max_materials,
            vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
        )?;

        Ok(Self {
            layout,
            pool,
            materials: HashMap::new(),
            next_id: 0,
            capacity: max_materials,
        })
    }

    pub unsafe fn ensure_capacity(
        &mut self,
        rrdevice: &RRDevice,
        required: u32,
    ) -> anyhow::Result<()> {
        if required <= self.capacity {
            return Ok(());
        }

        if self.pool != vk::DescriptorPool::null() {
            rrdevice.device.destroy_descriptor_pool(self.pool, None);
        }

        self.pool = self.layout.create_pool(
            rrdevice,
            required,
            vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
        )?;
        self.capacity = required;
        Ok(())
    }

    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::new(standard_graphics_shaders(), MATERIAL_SET)
    }

    pub unsafe fn create_material(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        name: &str,
        texture: RRImage,
        properties: MaterialUBO,
    ) -> anyhow::Result<MaterialId> {
        self.create_material_with_texture(
            instance,
            rrdevice,
            name,
            texture.image_view,
            texture.sampler,
            properties,
        )
    }

    pub unsafe fn create_material_with_texture(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        name: &str,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        properties: MaterialUBO,
    ) -> anyhow::Result<MaterialId> {
        let descriptor_set = self.layout.allocate_sets(rrdevice, self.pool, 1)?[0];

        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            rrdevice,
            size_of::<MaterialUBO>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let memory = rrdevice.device.map_memory(
            uniform_buffer_memory,
            0,
            size_of::<MaterialUBO>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(&properties, memory.cast(), 1);
        rrdevice.device.unmap_memory(uniform_buffer_memory);

        self.layout
            .writer(descriptor_set)
            .image(
                MATERIAL_TEXTURE_BINDING,
                image_view,
                sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .buffer(
                MATERIAL_UBO_BINDING,
                uniform_buffer,
                0,
                size_of::<MaterialUBO>() as u64,
            )?
            .apply(rrdevice);

        let id = self.next_id;
        self.next_id += 1;

        let material = Material {
            id,
            name: name.to_string(),
            descriptor_set,
            textures: vec![],
            uniform_buffer,
            uniform_buffer_memory,
            properties,
        };

        self.materials.insert(id, material);
        Ok(id)
    }

    pub fn get(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(&id)
    }

    pub unsafe fn clear_materials(&mut self, device: &vulkanalia::Device) {
        for material in self.materials.values() {
            device.destroy_buffer(material.uniform_buffer, None);
            device.free_memory(material.uniform_buffer_memory, None);
        }

        if self.pool != vk::DescriptorPool::null() {
            device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
                .ok();
        }

        self.materials.clear();
        self.next_id = 0;
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        for material in self.materials.values() {
            device.destroy_buffer(material.uniform_buffer, None);
            device.free_memory(material.uniform_buffer_memory, None);
        }

        if self.pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.pool, None);
        }
        self.layout.destroy(device);
    }
}
