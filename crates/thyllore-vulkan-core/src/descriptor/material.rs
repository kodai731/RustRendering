use std::collections::HashMap;

use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::descriptor::pass_manifest::SetRole;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::model;
use crate::resource::image::RRImage;
use crate::resource::uniform_buffer::{Placement, UniformBuffer};
use crate::vulkan::Instance;
use thyllore_render_core::MaterialUBO;

pub type MaterialId = u32;

#[derive(Clone, Debug)]
pub struct Material {
    pub id: MaterialId,
    pub name: String,
    pub descriptor_set: vk::DescriptorSet,
    pub textures: Vec<RRImage>,
    pub uniform_buffer: UniformBuffer<MaterialUBO>,
    pub properties: MaterialUBO,
}

#[derive(Clone, Debug, Default)]
pub struct MaterialManager {
    pub layout: ReflectedSetLayout,
    pub materials: HashMap<MaterialId, Material>,
    next_id: MaterialId,
    recycled_sets: Vec<vk::DescriptorSet>,
}

impl MaterialManager {
    pub unsafe fn new(rrdevice: &RRDevice) -> anyhow::Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;

        Ok(Self {
            layout,
            materials: HashMap::new(),
            next_id: 0,
            recycled_sets: Vec::new(),
        })
    }

    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::shared(SetRole::Material)
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
        let descriptor_set = match self.recycled_sets.pop() {
            Some(set) => set,
            None => self.layout.allocate_set(rrdevice)?,
        };

        let uniform_buffer = UniformBuffer::new(instance, rrdevice, 1, Placement::HostMapped)?;
        uniform_buffer.write_slot(rrdevice, 0, &properties)?;

        self.layout
            .writer(descriptor_set)
            .image(
                model::TEX_SAMPLER,
                image_view,
                sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .uniform(model::MATERIAL, &uniform_buffer, 0)?
            .apply(rrdevice);

        let id = self.next_id;
        self.next_id += 1;

        let material = Material {
            id,
            name: name.to_string(),
            descriptor_set,
            textures: vec![],
            uniform_buffer,
            properties,
        };

        self.materials.insert(id, material);
        Ok(id)
    }

    pub fn get(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(&id)
    }

    pub unsafe fn clear_materials(&mut self, device: &vulkanalia::Device) {
        for material in self.materials.values_mut() {
            material.uniform_buffer.destroy(device);
            self.recycled_sets.push(material.descriptor_set);
        }

        self.materials.clear();
        self.next_id = 0;
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        for material in self.materials.values_mut() {
            material.uniform_buffer.destroy(device);
        }
        self.recycled_sets.clear();
        self.layout.destroy(device);
    }
}
