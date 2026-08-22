use anyhow::{anyhow, Context, Result};
use vulkanalia::prelude::v1_0::*;

use crate::core::descriptor_allocator::PoolSignature;
use crate::core::device::RRDevice;
use crate::descriptor::pass_manifest::{passes_with_role, PassShaders, SetRole, ShaderFile};
use crate::descriptor::reflection::{
    kind_accepts, reflect_shader_bytes, DescriptorSetTable, LayoutMismatch, ShaderReflection,
};
use crate::resource::uniform_buffer::UniformBuffer;
use thyllore_spirv_reflect::{GpuBlock, ShaderBinding};

fn load_reflections(files: &[ShaderFile]) -> Result<Vec<ShaderReflection>> {
    let mut reflections = Vec::with_capacity(files.len());
    for file in files {
        let bytes = std::fs::read(file.path)
            .with_context(|| format!("read shader {} for reflection", file.path))?;
        let reflection = reflect_shader_bytes(&bytes)
            .with_context(|| format!("reflect shader {}", file.path))?;
        if reflection.stages != [file.stage] {
            return Err(anyhow!(
                "shader {} is declared as {:?} in passes.toml but its SPIR-V entry points are {:?}",
                file.path,
                file.stage,
                reflection.stages
            ));
        }
        reflections.push(reflection);
    }
    Ok(reflections)
}

pub fn verify_pass_layouts(pass: &PassShaders, layouts: &[ReflectedSetLayout]) -> Result<()> {
    let reflections = load_reflections(pass.stages)?;
    let table = DescriptorSetTable::from_reflections(&reflections)?;
    for set in table.set_indices() {
        let layout = layouts.get(set as usize).ok_or_else(|| {
            anyhow!(
                "pass `{}` uses descriptor set {set} but only {} layouts were given",
                pass.name(),
                layouts.len()
            )
        })?;
        let mismatches: Vec<String> = table
            .verify_layout(set, layout.bindings())
            .into_iter()
            .filter(|mismatch| !matches!(mismatch, LayoutMismatch::UnusedInShaders { .. }))
            .map(|mismatch| format!("{mismatch:?}"))
            .collect();
        if !mismatches.is_empty() {
            return Err(anyhow!(
                "pass `{}` set {set}: shader bindings are not covered by the given layout: {}",
                pass.name(),
                mismatches.join(", ")
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorTypeOverride {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedLayoutSpec {
    pub passes: Vec<&'static PassShaders>,
    pub role: SetRole,
    pub overrides: Vec<DescriptorTypeOverride>,
}

impl ReflectedLayoutSpec {
    pub fn new(passes: Vec<&'static PassShaders>, role: SetRole) -> Self {
        Self {
            passes,
            role,
            overrides: Vec::new(),
        }
    }

    pub fn shared(role: SetRole) -> Self {
        Self::new(passes_with_role(role), role)
    }

    pub fn local(pass: &'static PassShaders) -> Self {
        Self::new(vec![pass], SetRole::Local)
    }

    pub fn for_role(pass: &'static PassShaders, role: SetRole) -> Self {
        match role {
            SetRole::Local => Self::local(pass),
            SetRole::Frame | SetRole::Material | SetRole::Object => Self::shared(role),
        }
    }

    pub fn with_override(
        mut self,
        binding: ShaderBinding,
        descriptor_type: vk::DescriptorType,
    ) -> Self {
        self.overrides.push(DescriptorTypeOverride {
            binding: binding.binding,
            descriptor_type,
        });
        self
    }

    pub fn set_index(&self) -> Result<u32> {
        let mut resolved = None;
        for pass in &self.passes {
            let set = pass.set_index(self.role).ok_or_else(|| {
                anyhow!(
                    "pass `{}` binds no {:?} descriptor set (see shaders/passes.toml)",
                    pass.name(),
                    self.role
                )
            })?;
            match resolved {
                None => resolved = Some(set),
                Some(previous) if previous != set => {
                    return Err(anyhow!(
                        "{:?} set index differs between passes sharing one layout ({previous} vs {set} in `{}`)",
                        self.role,
                        pass.name()
                    ));
                }
                Some(_) => {}
            }
        }
        resolved.ok_or_else(|| anyhow!("layout spec for {:?} lists no pass", self.role))
    }

    pub fn shader_files(&self) -> Vec<ShaderFile> {
        let mut files: Vec<ShaderFile> = Vec::new();
        for file in self.passes.iter().flat_map(|pass| pass.stages.iter()) {
            if !files.iter().any(|known| known.path == file.path) {
                files.push(*file);
            }
        }
        files
    }

    pub fn reflect_table(&self) -> Result<DescriptorSetTable> {
        let set = self.set_index()?;
        let mut reflections = load_reflections(&self.shader_files())?;
        for reflection in &mut reflections {
            reflection.bindings.retain(|binding| binding.set == set);
        }
        Ok(DescriptorSetTable::from_reflections(&reflections)?)
    }

    pub fn resolve_bindings(&self) -> Result<Vec<vk::DescriptorSetLayoutBinding>> {
        ReflectedSetLayout::resolve_bindings(
            &self.reflect_table()?,
            self.set_index()?,
            &self.overrides,
        )
    }
}

fn reflected_block_sizes(table: &DescriptorSetTable, set: u32) -> Vec<(u32, u32)> {
    table
        .bindings(set)
        .map(|bindings| {
            bindings
                .iter()
                .filter_map(|(index, merged)| {
                    merged.block.as_ref().map(|block| (*index, block.size))
                })
                .collect()
        })
        .unwrap_or_default()
}

#[derive(Clone, Debug, Default)]
pub struct ReflectedSetLayout {
    pub handle: vk::DescriptorSetLayout,
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
    block_sizes: Vec<(u32, u32)>,
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
        let set = spec.set_index()?;
        let table = spec.reflect_table()?;
        let bindings = Self::resolve_bindings(&table, set, &spec.overrides)?;
        let block_sizes = reflected_block_sizes(&table, set);

        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let handle = rrdevice.device.create_descriptor_set_layout(&info, None)?;
        Ok(Self {
            handle,
            bindings,
            block_sizes,
        })
    }

    pub fn bindings(&self) -> &[vk::DescriptorSetLayoutBinding] {
        &self.bindings
    }

    pub fn block_size(&self, binding: u32) -> Result<u32> {
        self.block_sizes
            .iter()
            .find(|(index, _)| *index == binding)
            .map(|(_, size)| *size)
            .ok_or_else(|| anyhow!("descriptor set layout binding {binding} is not a buffer block"))
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
    fn resolve_descriptor_type(&self, binding: ShaderBinding) -> Result<vk::DescriptorType> {
        let descriptor_type = self.layout.descriptor_type(binding.binding)?;
        if !kind_accepts(binding.kind, descriptor_type) {
            return Err(anyhow!(
                "binding {} is {:?} in the shader but the layout holds {:?}",
                binding.binding,
                binding.kind,
                descriptor_type
            ));
        }
        Ok(descriptor_type)
    }

    pub fn buffer(
        mut self,
        binding: ShaderBinding,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<Self> {
        let descriptor_type = self.resolve_descriptor_type(binding)?;
        self.buffer_infos.push(
            vk::DescriptorBufferInfo::builder()
                .buffer(buffer)
                .offset(offset)
                .range(range)
                .build(),
        );
        self.entries.push(WriteEntry {
            binding: binding.binding,
            descriptor_type,
            source: WriteSource::Buffer(self.buffer_infos.len() - 1),
        });
        Ok(self)
    }

    pub fn uniform<T: GpuBlock>(
        self,
        binding: ShaderBinding,
        uniform: &UniformBuffer<T>,
        slot: usize,
    ) -> Result<Self> {
        self.check_block_covers::<T>(binding)?;
        let offset = uniform.slot_offset(slot)?;
        self.buffer(binding, uniform.handle(), offset, uniform.block_size())
    }

    pub fn uniform_dynamic<T: GpuBlock>(
        self,
        binding: ShaderBinding,
        uniform: &UniformBuffer<T>,
    ) -> Result<Self> {
        self.check_block_covers::<T>(binding)?;
        self.buffer(binding, uniform.handle(), 0, uniform.block_size())
    }

    fn check_block_covers<T: GpuBlock>(&self, binding: ShaderBinding) -> Result<()> {
        let shader_size = self.layout.block_size(binding.binding)? as usize;
        if T::SIZE < shader_size {
            return Err(anyhow!(
                "binding {}: shader block is {shader_size} bytes but Rust `{}` is {} bytes",
                binding.binding,
                T::NAME,
                T::SIZE
            ));
        }
        Ok(())
    }

    pub fn image(
        mut self,
        binding: ShaderBinding,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        image_layout: vk::ImageLayout,
    ) -> Result<Self> {
        let descriptor_type = self.resolve_descriptor_type(binding)?;
        self.image_infos.push(
            vk::DescriptorImageInfo::builder()
                .image_view(image_view)
                .sampler(sampler)
                .image_layout(image_layout)
                .build(),
        );
        self.entries.push(WriteEntry {
            binding: binding.binding,
            descriptor_type,
            source: WriteSource::Image(self.image_infos.len() - 1),
        });
        Ok(self)
    }

    pub fn acceleration_structure(
        mut self,
        binding: ShaderBinding,
        acceleration_structure: vk::AccelerationStructureKHR,
    ) -> Result<Self> {
        let descriptor_type = self.resolve_descriptor_type(binding)?;
        self.acceleration_structures.push(acceleration_structure);
        self.entries.push(WriteEntry {
            binding: binding.binding,
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
