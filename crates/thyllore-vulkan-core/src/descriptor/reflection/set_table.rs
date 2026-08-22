use std::collections::BTreeMap;

use vulkanalia::prelude::v1_0::*;

use thyllore_spirv_reflect::{
    DescriptorCount, DescriptorKind, ReflectError, ReflectedBinding, ReflectedBlock,
    ShaderReflection, ShaderStage,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergedBinding {
    pub name: String,
    pub kind: DescriptorKind,
    pub count: DescriptorCount,
    pub block: Option<ReflectedBlock>,
    pub stages: vk::ShaderStageFlags,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DescriptorSetTable {
    sets: BTreeMap<u32, BTreeMap<u32, MergedBinding>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutMismatch {
    MissingInLayout {
        binding: u32,
        shader_kind: DescriptorKind,
    },
    TypeDiffers {
        binding: u32,
        shader_kind: DescriptorKind,
        layout_type: vk::DescriptorType,
    },
    CountDiffers {
        binding: u32,
        shader_count: DescriptorCount,
        layout_count: u32,
    },
    StageMissing {
        binding: u32,
        shader_stages: vk::ShaderStageFlags,
        layout_stages: vk::ShaderStageFlags,
    },
    UnusedInShaders {
        binding: u32,
    },
}

impl DescriptorSetTable {
    pub fn from_reflections(reflections: &[ShaderReflection]) -> Result<Self, ReflectError> {
        let mut table = Self::default();
        for reflection in reflections {
            let stages = reflection
                .stages
                .iter()
                .fold(vk::ShaderStageFlags::empty(), |flags, stage| {
                    flags | shader_stage_flags(*stage)
                });
            for binding in &reflection.bindings {
                table.merge_binding(binding, stages)?;
            }
        }
        Ok(table)
    }

    fn merge_binding(
        &mut self,
        binding: &ReflectedBinding,
        stages: vk::ShaderStageFlags,
    ) -> Result<(), ReflectError> {
        let slot = self
            .sets
            .entry(binding.set)
            .or_default()
            .entry(binding.binding);

        match slot {
            std::collections::btree_map::Entry::Vacant(vacant) => {
                vacant.insert(MergedBinding {
                    name: binding.name.clone(),
                    kind: binding.kind,
                    count: binding.count,
                    block: binding.block.clone(),
                    stages,
                });
            }
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                let existing = occupied.get_mut();
                let is_same_resource = existing.kind == binding.kind
                    && existing.count == binding.count
                    && existing.block == binding.block;
                if !is_same_resource {
                    return Err(ReflectError::ConflictingBinding {
                        set: binding.set,
                        binding: binding.binding,
                    });
                }
                existing.stages |= stages;
            }
        }
        Ok(())
    }

    pub fn set_indices(&self) -> Vec<u32> {
        self.sets.keys().copied().collect()
    }

    pub fn bindings(&self, set: u32) -> Option<&BTreeMap<u32, MergedBinding>> {
        self.sets.get(&set)
    }

    pub fn binding(&self, set: u32, binding: u32) -> Option<&MergedBinding> {
        self.sets
            .get(&set)
            .and_then(|bindings| bindings.get(&binding))
    }

    pub fn layout_bindings(&self, set: u32) -> Vec<vk::DescriptorSetLayoutBinding> {
        self.sets
            .get(&set)
            .map(|bindings| {
                bindings
                    .iter()
                    .map(|(index, merged)| {
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(*index)
                            .descriptor_type(default_descriptor_type(merged.kind))
                            .descriptor_count(match merged.count {
                                DescriptorCount::Fixed(count) => count,
                                DescriptorCount::Unbounded => 0,
                            })
                            .stage_flags(merged.stages)
                            .build()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn verify_layout(
        &self,
        set: u32,
        layout: &[vk::DescriptorSetLayoutBinding],
    ) -> Vec<LayoutMismatch> {
        let empty = BTreeMap::new();
        let reflected = self.sets.get(&set).unwrap_or(&empty);
        let mut mismatches = Vec::new();

        for (index, merged) in reflected {
            let Some(handwritten) = layout.iter().find(|entry| entry.binding == *index) else {
                mismatches.push(LayoutMismatch::MissingInLayout {
                    binding: *index,
                    shader_kind: merged.kind,
                });
                continue;
            };
            mismatches.extend(compare_binding(*index, merged, handwritten));
        }

        for handwritten in layout {
            if !reflected.contains_key(&handwritten.binding) {
                mismatches.push(LayoutMismatch::UnusedInShaders {
                    binding: handwritten.binding,
                });
            }
        }

        mismatches
    }
}

fn compare_binding(
    index: u32,
    merged: &MergedBinding,
    handwritten: &vk::DescriptorSetLayoutBinding,
) -> Vec<LayoutMismatch> {
    let mut mismatches = Vec::new();

    if !kind_accepts(merged.kind, handwritten.descriptor_type) {
        mismatches.push(LayoutMismatch::TypeDiffers {
            binding: index,
            shader_kind: merged.kind,
            layout_type: handwritten.descriptor_type,
        });
    }

    let count_matches = match merged.count {
        DescriptorCount::Fixed(count) => count == handwritten.descriptor_count,
        DescriptorCount::Unbounded => handwritten.descriptor_count > 0,
    };
    if !count_matches {
        mismatches.push(LayoutMismatch::CountDiffers {
            binding: index,
            shader_count: merged.count,
            layout_count: handwritten.descriptor_count,
        });
    }

    if !handwritten.stage_flags.contains(merged.stages) {
        mismatches.push(LayoutMismatch::StageMissing {
            binding: index,
            shader_stages: merged.stages,
            layout_stages: handwritten.stage_flags,
        });
    }

    mismatches
}

pub fn kind_accepts(kind: DescriptorKind, descriptor_type: vk::DescriptorType) -> bool {
    match kind {
        DescriptorKind::UniformBuffer => matches!(
            descriptor_type,
            vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
        ),
        DescriptorKind::StorageBuffer => matches!(
            descriptor_type,
            vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
        ),
        DescriptorKind::Sampler
        | DescriptorKind::CombinedImageSampler
        | DescriptorKind::SampledImage
        | DescriptorKind::StorageImage
        | DescriptorKind::UniformTexelBuffer
        | DescriptorKind::StorageTexelBuffer
        | DescriptorKind::InputAttachment
        | DescriptorKind::AccelerationStructure => descriptor_type == default_descriptor_type(kind),
    }
}

pub fn default_descriptor_type(kind: DescriptorKind) -> vk::DescriptorType {
    match kind {
        DescriptorKind::Sampler => vk::DescriptorType::SAMPLER,
        DescriptorKind::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        DescriptorKind::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
        DescriptorKind::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        DescriptorKind::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        DescriptorKind::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        DescriptorKind::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        DescriptorKind::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        DescriptorKind::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
        DescriptorKind::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
    }
}

pub fn shader_stage_flags(stage: ShaderStage) -> vk::ShaderStageFlags {
    match stage {
        ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
        ShaderStage::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        ShaderStage::TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
        ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
        ShaderStage::RayGeneration => vk::ShaderStageFlags::RAYGEN_KHR,
        ShaderStage::Intersection => vk::ShaderStageFlags::INTERSECTION_KHR,
        ShaderStage::AnyHit => vk::ShaderStageFlags::ANY_HIT_KHR,
        ShaderStage::ClosestHit => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        ShaderStage::Miss => vk::ShaderStageFlags::MISS_KHR,
        ShaderStage::Callable => vk::ShaderStageFlags::CALLABLE_KHR,
        ShaderStage::Task => vk::ShaderStageFlags::TASK_EXT,
        ShaderStage::Mesh => vk::ShaderStageFlags::MESH_EXT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(set: u32, index: u32, kind: DescriptorKind) -> ReflectedBinding {
        ReflectedBinding {
            set,
            binding: index,
            name: format!("b{index}"),
            kind,
            count: DescriptorCount::Fixed(1),
            block: None,
        }
    }

    fn layout_entry(
        index: u32,
        descriptor_type: vk::DescriptorType,
        stages: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding::builder()
            .binding(index)
            .descriptor_type(descriptor_type)
            .descriptor_count(1)
            .stage_flags(stages)
            .build()
    }

    fn vertex_and_fragment_table() -> DescriptorSetTable {
        let vertex = ShaderReflection {
            stages: vec![ShaderStage::Vertex],
            bindings: vec![binding(0, 0, DescriptorKind::UniformBuffer)],
        };
        let fragment = ShaderReflection {
            stages: vec![ShaderStage::Fragment],
            bindings: vec![
                binding(0, 0, DescriptorKind::UniformBuffer),
                binding(0, 1, DescriptorKind::CombinedImageSampler),
            ],
        };
        DescriptorSetTable::from_reflections(&[vertex, fragment]).unwrap()
    }

    #[test]
    fn merges_stages_of_shared_bindings() {
        let table = vertex_and_fragment_table();
        assert_eq!(
            table.binding(0, 0).unwrap().stages,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
        );
        assert_eq!(
            table.binding(0, 1).unwrap().stages,
            vk::ShaderStageFlags::FRAGMENT
        );
    }

    #[test]
    fn rejects_conflicting_kinds_across_stages() {
        let vertex = ShaderReflection {
            stages: vec![ShaderStage::Vertex],
            bindings: vec![binding(0, 0, DescriptorKind::UniformBuffer)],
        };
        let fragment = ShaderReflection {
            stages: vec![ShaderStage::Fragment],
            bindings: vec![binding(0, 0, DescriptorKind::StorageBuffer)],
        };
        assert_eq!(
            DescriptorSetTable::from_reflections(&[vertex, fragment]),
            Err(ReflectError::ConflictingBinding { set: 0, binding: 0 })
        );
    }

    #[test]
    fn verify_layout_reports_every_kind_of_drift() {
        let table = vertex_and_fragment_table();
        let layout = [
            layout_entry(
                0,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                vk::ShaderStageFlags::VERTEX,
            ),
            layout_entry(
                2,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            ),
        ];

        let mismatches = table.verify_layout(0, &layout);
        assert_eq!(
            mismatches,
            vec![
                LayoutMismatch::StageMissing {
                    binding: 0,
                    shader_stages: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    layout_stages: vk::ShaderStageFlags::VERTEX,
                },
                LayoutMismatch::MissingInLayout {
                    binding: 1,
                    shader_kind: DescriptorKind::CombinedImageSampler,
                },
                LayoutMismatch::UnusedInShaders { binding: 2 },
            ]
        );
    }

    #[test]
    fn verify_layout_accepts_matching_layout() {
        let table = vertex_and_fragment_table();
        assert!(table.verify_layout(0, &table.layout_bindings(0)).is_empty());
    }
}
