use std::path::Path;

use anyhow::{anyhow, Context, Result};
use vulkanalia::prelude::v1_0::*;

use crate::core::descriptor_allocator::PoolSignature;
use crate::core::device::RRDevice;
use crate::descriptor::reflection::{
    kind_accepts, reflect_shader_bytes, DescriptorSetTable, ShaderReflection,
};

fn load_reflections<P: AsRef<Path>>(paths: &[P]) -> Result<Vec<ShaderReflection>> {
    let mut reflections = Vec::with_capacity(paths.len());
    for path in paths {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("read shader {} for reflection", path.display()))?;
        let reflection = reflect_shader_bytes(&bytes)
            .with_context(|| format!("reflect shader {}", path.display()))?;
        reflections.push(reflection);
    }
    Ok(reflections)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorTypeOverride {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedLayoutSpec {
    pub shaders: Vec<&'static str>,
    pub set: u32,
    pub overrides: Vec<DescriptorTypeOverride>,
}

impl ReflectedLayoutSpec {
    pub fn new(shaders: Vec<&'static str>, set: u32) -> Self {
        Self {
            shaders,
            set,
            overrides: Vec::new(),
        }
    }

    pub fn with_override(mut self, binding: u32, descriptor_type: vk::DescriptorType) -> Self {
        self.overrides.push(DescriptorTypeOverride {
            binding,
            descriptor_type,
        });
        self
    }

    pub fn reflect_table(&self) -> Result<DescriptorSetTable> {
        let mut reflections = load_reflections(&self.shaders)?;
        for reflection in &mut reflections {
            reflection
                .bindings
                .retain(|binding| binding.set == self.set);
        }
        Ok(DescriptorSetTable::from_reflections(&reflections)?)
    }

    pub fn resolve_bindings(&self) -> Result<Vec<vk::DescriptorSetLayoutBinding>> {
        ReflectedSetLayout::resolve_bindings(&self.reflect_table()?, self.set, &self.overrides)
    }
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

    pub unsafe fn create(rrdevice: &RRDevice, spec: &ReflectedLayoutSpec) -> Result<Self> {
        let bindings = spec.resolve_bindings()?;
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

    pub unsafe fn allocate_sets(
        &self,
        rrdevice: &RRDevice,
        count: usize,
    ) -> Result<Vec<vk::DescriptorSet>> {
        let signature = PoolSignature::from_bindings(&self.bindings);
        rrdevice.allocate_descriptor_sets(self.handle, &signature, count)
    }

    pub unsafe fn allocate_set(&self, rrdevice: &RRDevice) -> Result<vk::DescriptorSet> {
        self.allocate_sets(rrdevice, 1)?
            .pop()
            .ok_or_else(|| anyhow!("descriptor allocator returned no set"))
    }

    pub fn writer(&self, dst_set: vk::DescriptorSet) -> DescriptorSetWriter<'_> {
        DescriptorSetWriter {
            layout: self,
            dst_set,
            buffer_infos: Vec::new(),
            image_infos: Vec::new(),
            acceleration_structures: Vec::new(),
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
    AccelerationStructure(usize),
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
    acceleration_structures: Vec<vk::AccelerationStructureKHR>,
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

    pub fn acceleration_structure(
        mut self,
        binding: u32,
        acceleration_structure: vk::AccelerationStructureKHR,
    ) -> Result<Self> {
        let descriptor_type = self.layout.descriptor_type(binding)?;
        self.acceleration_structures.push(acceleration_structure);
        self.entries.push(WriteEntry {
            binding,
            descriptor_type,
            source: WriteSource::AccelerationStructure(self.acceleration_structures.len() - 1),
        });
        Ok(self)
    }

    pub unsafe fn apply(self, rrdevice: &RRDevice) {
        let acceleration_infos: Vec<vk::WriteDescriptorSetAccelerationStructureKHR> = self
            .acceleration_structures
            .iter()
            .map(|handle| {
                vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                    .acceleration_structures(std::slice::from_ref(handle))
                    .build()
            })
            .collect();

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
                    WriteSource::AccelerationStructure(index) => {
                        let mut write = write.build();
                        write.next = (&acceleration_infos[index]
                            as *const vk::WriteDescriptorSetAccelerationStructureKHR)
                            .cast();
                        write.descriptor_count = 1;
                        write
                    }
                }
            })
            .collect();

        rrdevice
            .device
            .update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }
}
