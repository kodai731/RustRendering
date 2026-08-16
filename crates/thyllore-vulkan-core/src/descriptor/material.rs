use std::collections::HashMap;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::resource::buffer::create_buffer;
use crate::resource::image::RRImage;
use crate::vulkan::Instance;
use thyllore_render_core::MaterialUBO;

pub type MaterialId = u32;

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
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub materials: HashMap<MaterialId, Material>,
    next_id: MaterialId,
    capacity: u32,
}

impl MaterialManager {
    pub unsafe fn new(rrdevice: &RRDevice, max_materials: u32) -> anyhow::Result<Self> {
        let layout = Self::create_layout(rrdevice)?;
        let pool = Self::create_pool(rrdevice, max_materials)?;

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

        self.pool = Self::create_pool(rrdevice, required)?;
        self.capacity = required;
        Ok(())
    }

    pub fn layout_bindings() -> Vec<vk::DescriptorSetLayoutBinding> {
        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        vec![sampler_binding, ubo_binding]
    }

    unsafe fn create_layout(rrdevice: &RRDevice) -> anyhow::Result<vk::DescriptorSetLayout> {
        let bindings = Self::layout_bindings();
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        Ok(rrdevice.device.create_descriptor_set_layout(&info, None)?)
    }

    unsafe fn create_pool(
        rrdevice: &RRDevice,
        max_materials: u32,
    ) -> anyhow::Result<vk::DescriptorPool> {
        let sampler_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(max_materials);

        let ubo_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(max_materials);

        let pool_sizes = &[sampler_size, ubo_size];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(max_materials)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        Ok(rrdevice.device.create_descriptor_pool(&info, None)?)
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
        let layouts = &[self.layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.pool)
            .set_layouts(layouts);

        let descriptor_sets = rrdevice.device.allocate_descriptor_sets(&alloc_info)?;
        let descriptor_set = descriptor_sets[0];

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

        let image_info = vk::DescriptorImageInfo::builder()
            .sampler(sampler)
            .image_view(image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(uniform_buffer)
            .offset(0)
            .range(size_of::<MaterialUBO>() as u64);

        let image_infos = &[image_info];
        let buffer_infos = &[buffer_info];

        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(image_infos);

        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_infos);

        rrdevice
            .device
            .update_descriptor_sets(&[sampler_write, ubo_write], &[] as &[vk::CopyDescriptorSet]);

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
        if self.layout != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
