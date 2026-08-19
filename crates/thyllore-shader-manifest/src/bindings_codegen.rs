use std::fmt::Write;

use thiserror::Error;
use thyllore_spirv_reflect::{
    binding_const_name, DescriptorCount, ReflectedBinding, ShaderReflection,
};

use crate::manifest::{PassDefinition, PassManifest};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BindingCodegenError {
    #[error("pass `{pass}`: no reflection provided for shader `{file}`")]
    MissingReflection { pass: String, file: String },
    #[error("pass `{pass}`: set {set} binding {binding} is `{first}` in one stage but `{second}` in another; stages of one pass must agree on descriptor names")]
    NameConflict {
        pass: String,
        set: u32,
        binding: u32,
        first: String,
        second: String,
    },
    #[error("pass `{pass}`: `{first}` and `{second}` both map to constant `{constant}`; rename one descriptor in GLSL")]
    ConstantCollision {
        pass: String,
        first: String,
        second: String,
        constant: String,
    },
}

pub fn generate_shader_bindings_rust(
    manifest: &PassManifest,
    reflection_of: impl Fn(&str) -> Option<ShaderReflection>,
) -> Result<String, BindingCodegenError> {
    let mut out = String::new();
    out.push_str("use thyllore_spirv_reflect::{DescriptorCount, DescriptorKind, ShaderBinding};\n");
    for pass in &manifest.passes {
        write_pass_module(&mut out, pass, &reflection_of)?;
    }
    Ok(out)
}

fn write_pass_module(
    out: &mut String,
    pass: &PassDefinition,
    reflection_of: &impl Fn(&str) -> Option<ShaderReflection>,
) -> Result<(), BindingCodegenError> {
    let bindings = merge_pass_bindings(pass, reflection_of)?;

    let _ = writeln!(out, "\npub mod {} {{\n    use super::*;\n", pass.name);
    let mut emitted: Vec<(String, String)> = Vec::new();
    for binding in &bindings {
        let constant = binding_const_name(&binding.name);
        if let Some((_, previous)) = emitted.iter().find(|(name, _)| *name == constant) {
            return Err(BindingCodegenError::ConstantCollision {
                pass: pass.name.clone(),
                first: previous.clone(),
                second: binding.name.clone(),
                constant,
            });
        }
        emitted.push((constant.clone(), binding.name.clone()));
        let _ = writeln!(
            out,
            "    pub const {constant}: ShaderBinding = ShaderBinding {{ set: {}, binding: {}, kind: DescriptorKind::{:?}, count: {} }};",
            binding.set,
            binding.binding,
            binding.kind,
            count_expression(binding.count),
        );
    }
    out.push_str("}\n");
    Ok(())
}

fn merge_pass_bindings(
    pass: &PassDefinition,
    reflection_of: &impl Fn(&str) -> Option<ShaderReflection>,
) -> Result<Vec<ReflectedBinding>, BindingCodegenError> {
    let mut merged: Vec<ReflectedBinding> = Vec::new();
    for stage in &pass.stages {
        let reflection = reflection_of(&stage.source_file).ok_or_else(|| {
            BindingCodegenError::MissingReflection {
                pass: pass.name.clone(),
                file: stage.source_file.clone(),
            }
        })?;
        for binding in reflection.bindings {
            match merged
                .iter()
                .find(|known| known.set == binding.set && known.binding == binding.binding)
            {
                Some(known) if known.name != binding.name => {
                    return Err(BindingCodegenError::NameConflict {
                        pass: pass.name.clone(),
                        set: binding.set,
                        binding: binding.binding,
                        first: known.name.clone(),
                        second: binding.name,
                    });
                }
                Some(_) => {}
                None => merged.push(binding),
            }
        }
    }
    merged.sort_by_key(|binding| (binding.set, binding.binding));
    Ok(merged)
}

fn count_expression(count: DescriptorCount) -> String {
    match count {
        DescriptorCount::Fixed(count) => format!("DescriptorCount::Fixed({count})"),
        DescriptorCount::Unbounded => "DescriptorCount::Unbounded".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_spirv_reflect::{DescriptorKind, ShaderStage};

    fn reflection(bindings: Vec<ReflectedBinding>) -> ShaderReflection {
        ShaderReflection {
            stages: vec![ShaderStage::Fragment],
            bindings,
        }
    }

    fn binding(set: u32, index: u32, name: &str, kind: DescriptorKind) -> ReflectedBinding {
        ReflectedBinding {
            set,
            binding: index,
            name: name.to_string(),
            kind,
            count: DescriptorCount::Fixed(1),
            block_size: None,
        }
    }

    #[test]
    fn generates_one_module_per_pass_with_merged_constants() {
        let manifest = PassManifest::parse(
            "[pass.flame_resolve]\nstages = [\"tonemapVertex.vert\", \"flameResolveFragment.frag\"]\nsets = { 0 = \"local\" }\n",
        )
        .unwrap();
        let code = generate_shader_bindings_rust(&manifest, |file| match file {
            "tonemapVertex.vert" => Some(reflection(vec![])),
            "flameResolveFragment.frag" => Some(reflection(vec![
                binding(0, 0, "flame", DescriptorKind::UniformBuffer),
                binding(0, 4, "historySampler", DescriptorKind::CombinedImageSampler),
            ])),
            _ => None,
        })
        .unwrap();
        assert!(code.contains("pub mod flame_resolve {"));
        assert!(code.contains("pub const FLAME: ShaderBinding = ShaderBinding { set: 0, binding: 0, kind: DescriptorKind::UniformBuffer, count: DescriptorCount::Fixed(1) };"));
        assert!(code.contains("pub const HISTORY_SAMPLER: ShaderBinding = ShaderBinding { set: 0, binding: 4, kind: DescriptorKind::CombinedImageSampler, count: DescriptorCount::Fixed(1) };"));
    }

    #[test]
    fn rejects_same_slot_with_different_names_across_stages() {
        let manifest = PassManifest::parse(
            "[pass.model]\nstages = [\"vertex.vert\", \"fragment.frag\"]\nsets = { 0 = \"frame\" }\n",
        )
        .unwrap();
        let error = generate_shader_bindings_rust(&manifest, |file| match file {
            "vertex.vert" => Some(reflection(vec![binding(
                0,
                0,
                "frameData",
                DescriptorKind::UniformBuffer,
            )])),
            "fragment.frag" => Some(reflection(vec![binding(
                0,
                0,
                "frame",
                DescriptorKind::UniformBuffer,
            )])),
            _ => None,
        })
        .unwrap_err();
        assert!(matches!(error, BindingCodegenError::NameConflict { .. }));
    }

    #[test]
    fn rejects_two_descriptors_mapping_to_one_constant() {
        let manifest = PassManifest::parse(
            "[pass.dof]\nstages = [\"tonemapVertex.vert\", \"dofFragment.frag\"]\nsets = { 0 = \"local\" }\n",
        )
        .unwrap();
        let error = generate_shader_bindings_rust(&manifest, |file| match file {
            "tonemapVertex.vert" => Some(reflection(vec![])),
            "dofFragment.frag" => Some(reflection(vec![
                binding(0, 0, "hdrSampler", DescriptorKind::CombinedImageSampler),
                binding(0, 1, "hdr_sampler", DescriptorKind::CombinedImageSampler),
            ])),
            _ => None,
        })
        .unwrap_err();
        assert!(matches!(
            error,
            BindingCodegenError::ConstantCollision { .. }
        ));
    }
}
