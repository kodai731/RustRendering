use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::descriptor::reflection::{
    kind_accepts, reflect_shader_bytes, DescriptorSetTable, ShaderReflection,
};

pub fn reflect_shader_files<P: AsRef<Path>>(paths: &[P]) -> Result<DescriptorSetTable> {
    let mut reflections = Vec::with_capacity(paths.len());
    for path in paths {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("read shader {} for reflection", path.display()))?;
        let reflection: ShaderReflection = reflect_shader_bytes(&bytes)
            .with_context(|| format!("reflect shader {}", path.display()))?;
        reflections.push(reflection);
    }
    Ok(DescriptorSetTable::from_reflections(&reflections)?)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorTypeOverride {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
}

#[derive(Clone, Debug, Default)]
pub struct ReflectedSetLayout {
    pub handle: vk::DescriptorSetLayout,
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl ReflectedSetLayout {
    pub fn resolve_bindings(
        table: &DescriptorSetTable,
        set: u32,
        overrides: &[DescriptorTypeOverride],
    ) -> Result<Vec<vk::DescriptorSetLayoutBinding>> {
        let mut bindings = table.layout_bindings(set);
        if bindings.is_empty() {
            return Err(anyhow!("shaders declare no descriptor set {set}"));
        }

        for override_entry in overrides {
            let reflected = table.binding(set, override_entry.binding).ok_or_else(|| {
                anyhow!(
                    "override targets set {set} binding {} which no shader declares",
                    override_entry.binding
                )
            })?;
            if !kind_accepts(reflected.kind, override_entry.descriptor_type) {
                return Err(anyhow!(
                    "override {:?} is incompatible with shader {:?} at set {set} binding {}",
                    override_entry.descriptor_type,
                    reflected.kind,
                    override_entry.binding
                ));
            }
            let target = bindings
                .iter_mut()
                .find(|binding| binding.binding == override_entry.binding)
                .ok_or_else(|| anyhow!("override binding {} missing", override_entry.binding))?;
            target.descriptor_type = override_entry.descriptor_type;
        }

        Ok(bindings)
    }

    pub unsafe fn create(
        rrdevice: &RRDevice,
        table: &DescriptorSetTable,
        set: u32,
        overrides: &[DescriptorTypeOverride],
    ) -> Result<Self> {
        let bindings = Self::resolve_bindings(table, set, overrides)?;
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let handle = rrdevice.device.create_descriptor_set_layout(&info, None)?;
        Ok(Self { handle, bindings })
    }

    pub fn bindings(&self) -> &[vk::DescriptorSetLayoutBinding] {
        &self.bindings
    }

    pub fn descriptor_type(&self, binding: u32) -> Result<vk::DescriptorType> {
        self.bindings
            .iter()
            .find(|entry| entry.binding == binding)
            .map(|entry| entry.descriptor_type)
            .ok_or_else(|| anyhow!("descriptor set layout has no binding {binding}"))
    }

    pub fn pool_sizes(&self, max_sets: u32) -> Vec<vk::DescriptorPoolSize> {
        let mut counts: BTreeMap<i32, u32> = BTreeMap::new();
        for binding in &self.bindings {
            *counts.entry(binding.descriptor_type.as_raw()).or_default() +=
                binding.descriptor_count * max_sets;
        }
        counts
            .into_iter()
            .map(|(raw_type, count)| {
                vk::DescriptorPoolSize::builder()
                    .type_(vk::DescriptorType::from_raw(raw_type))
                    .descriptor_count(count)
                    .build()
            })
            .collect()
    }

    pub unsafe fn create_pool(
        &self,
        rrdevice: &RRDevice,
        max_sets: u32,
    ) -> Result<vk::DescriptorPool> {
        let pool_sizes = self.pool_sizes(max_sets);
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets);
        Ok(rrdevice.device.create_descriptor_pool(&info, None)?)
    }

    pub unsafe fn allocate_sets(
        &self,
        rrdevice: &RRDevice,
        pool: vk::DescriptorPool,
        count: usize,
    ) -> Result<Vec<vk::DescriptorSet>> {
        let layouts = vec![self.handle; count];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        Ok(rrdevice.device.allocate_descriptor_sets(&info)?)
    }

    pub fn writer(&self, dst_set: vk::DescriptorSet) -> DescriptorSetWriter<'_> {
        DescriptorSetWriter {
            layout: self,
            dst_set,
            buffer_infos: Vec::new(),
            image_infos: Vec::new(),
            entries: Vec::new(),
        }
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.handle != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.handle, None);
            self.handle = vk::DescriptorSetLayout::null();
        }
    }
}

enum WriteSource {
    Buffer(usize),
    Image(usize),
}

struct WriteEntry {
    binding: u32,
    descriptor_type: vk::DescriptorType,
    source: WriteSource,
}

pub struct DescriptorSetWriter<'a> {
    layout: &'a ReflectedSetLayout,
    dst_set: vk::DescriptorSet,
    buffer_infos: Vec<vk::DescriptorBufferInfo>,
    image_infos: Vec<vk::DescriptorImageInfo>,
    entries: Vec<WriteEntry>,
}

impl DescriptorSetWriter<'_> {
    pub fn buffer(
        mut self,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<Self> {
        let descriptor_type = self.layout.descriptor_type(binding)?;
        self.buffer_infos.push(
            vk::DescriptorBufferInfo::builder()
                .buffer(buffer)
                .offset(offset)
                .range(range)
                .build(),
        );
        self.entries.push(WriteEntry {
            binding,
            descriptor_type,
            source: WriteSource::Buffer(self.buffer_infos.len() - 1),
        });
        Ok(self)
    }

    pub fn image(
        mut self,
        binding: u32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        image_layout: vk::ImageLayout,
    ) -> Result<Self> {
        let descriptor_type = self.layout.descriptor_type(binding)?;
        self.image_infos.push(
            vk::DescriptorImageInfo::builder()
                .image_view(image_view)
                .sampler(sampler)
                .image_layout(image_layout)
                .build(),
        );
        self.entries.push(WriteEntry {
            binding,
            descriptor_type,
            source: WriteSource::Image(self.image_infos.len() - 1),
        });
        Ok(self)
    }

    pub unsafe fn apply(self, rrdevice: &RRDevice) {
        let writes: Vec<vk::WriteDescriptorSet> = self
            .entries
            .iter()
            .map(|entry| {
                let write = vk::WriteDescriptorSet::builder()
                    .dst_set(self.dst_set)
                    .dst_binding(entry.binding)
                    .dst_array_element(0)
                    .descriptor_type(entry.descriptor_type);
                match entry.source {
                    WriteSource::Buffer(index) => write
                        .buffer_info(std::slice::from_ref(&self.buffer_infos[index]))
                        .build(),
                    WriteSource::Image(index) => write
                        .image_info(std::slice::from_ref(&self.image_infos[index]))
                        .build(),
                }
            })
            .collect();

        rrdevice
            .device
            .update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }
}
