use std::collections::BTreeMap;
use std::fmt;

use crate::glsl::{
    scan_glsl_descriptor_declarations, GlslArrayCount, GlslDescriptorClass,
    GlslDescriptorDeclaration,
};
use crate::parser::reflect_shader_bytes;
use crate::types::{DescriptorCount, DescriptorKind, ReflectError, ReflectedBinding};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeclarationMismatch {
    DeclaredButNotReflected {
        set: u32,
        binding: u32,
        name: String,
    },
    ReflectedButNotDeclared {
        set: u32,
        binding: u32,
        name: String,
    },
    ClassDiffers {
        set: u32,
        binding: u32,
        name: String,
        declared: GlslDescriptorClass,
        reflected: DescriptorKind,
    },
    CountDiffers {
        set: u32,
        binding: u32,
        name: String,
        declared: u32,
        reflected: DescriptorCount,
    },
}

impl fmt::Display for DeclarationMismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeclaredButNotReflected { set, binding, name } => write!(
                formatter,
                "set {set} binding {binding} `{name}`: declared in GLSL but missing from SPIR-V reflection"
            ),
            Self::ReflectedButNotDeclared { set, binding, name } => write!(
                formatter,
                "set {set} binding {binding} `{name}`: reflected from SPIR-V but not declared in GLSL"
            ),
            Self::ClassDiffers {
                set,
                binding,
                name,
                declared,
                reflected,
            } => write!(
                formatter,
                "set {set} binding {binding} `{name}`: GLSL declares {declared:?} but SPIR-V reflects {reflected:?}"
            ),
            Self::CountDiffers {
                set,
                binding,
                name,
                declared,
                reflected,
            } => write!(
                formatter,
                "set {set} binding {binding} `{name}`: GLSL declares [{declared}] but SPIR-V reflects {reflected:?}"
            ),
        }
    }
}

pub fn verify_spirv_against_glsl(
    preprocessed_glsl: &str,
    spirv_bytes: &[u8],
) -> Result<Vec<DeclarationMismatch>, ReflectError> {
    let reflection = reflect_shader_bytes(spirv_bytes)?;
    let declared: BTreeMap<(u32, u32), GlslDescriptorDeclaration> =
        scan_glsl_descriptor_declarations(preprocessed_glsl)
            .into_iter()
            .map(|declaration| ((declaration.set, declaration.binding), declaration))
            .collect();
    let reflected: BTreeMap<(u32, u32), ReflectedBinding> = reflection
        .bindings
        .into_iter()
        .map(|binding| ((binding.set, binding.binding), binding))
        .collect();

    let mut mismatches = Vec::new();
    for (key, declaration) in &declared {
        match reflected.get(key) {
            Some(binding) => mismatches.extend(compare_declaration(declaration, binding)),
            None => mismatches.push(DeclarationMismatch::DeclaredButNotReflected {
                set: declaration.set,
                binding: declaration.binding,
                name: declaration.name.clone(),
            }),
        }
    }
    for (key, binding) in &reflected {
        if !declared.contains_key(key) {
            mismatches.push(DeclarationMismatch::ReflectedButNotDeclared {
                set: binding.set,
                binding: binding.binding,
                name: binding.name.clone(),
            });
        }
    }
    Ok(mismatches)
}

fn compare_declaration(
    declaration: &GlslDescriptorDeclaration,
    binding: &ReflectedBinding,
) -> Vec<DeclarationMismatch> {
    let mut mismatches = Vec::new();
    if !class_matches(declaration.class, binding.kind) {
        mismatches.push(DeclarationMismatch::ClassDiffers {
            set: declaration.set,
            binding: declaration.binding,
            name: declaration.name.clone(),
            declared: declaration.class,
            reflected: binding.kind,
        });
    }
    if let GlslArrayCount::Fixed(declared) = declaration.count {
        if binding.count != DescriptorCount::Fixed(declared) {
            mismatches.push(DeclarationMismatch::CountDiffers {
                set: declaration.set,
                binding: declaration.binding,
                name: declaration.name.clone(),
                declared,
                reflected: binding.count,
            });
        }
    }
    mismatches
}

fn class_matches(class: GlslDescriptorClass, kind: DescriptorKind) -> bool {
    match class {
        GlslDescriptorClass::UniformBlock => kind == DescriptorKind::UniformBuffer,
        GlslDescriptorClass::StorageBlock => kind == DescriptorKind::StorageBuffer,
        GlslDescriptorClass::Opaque => !matches!(
            kind,
            DescriptorKind::UniformBuffer | DescriptorKind::StorageBuffer
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::tests::fragment_module_with_block_and_sampler;

    fn module_bytes() -> Vec<u8> {
        fragment_module_with_block_and_sampler()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }

    const MATCHING_GLSL: &str = r#"
        layout(set = 0, binding = 0) uniform sampler2D texSampler;
        layout(set = 0, binding = 1) uniform sampler2D shadowMaps[4];
        layout(set = 1, binding = 2) uniform Frame { mat4 view; vec4 tint; vec4 rows[3]; } frame;
        layout(set = 2, binding = 0) buffer Particles { vec4 positions[]; } particles;
    "#;

    #[test]
    fn matching_declarations_produce_no_mismatch() {
        let mismatches = verify_spirv_against_glsl(MATCHING_GLSL, &module_bytes()).unwrap();
        assert!(mismatches.is_empty(), "{mismatches:?}");
    }

    #[test]
    fn reports_missing_extra_class_and_count_drift() {
        let drifted = r#"
            layout(set = 0, binding = 0) uniform sampler2D texSampler;
            layout(set = 0, binding = 1) uniform sampler2D shadowMaps[2];
            layout(set = 1, binding = 2) buffer Frame { mat4 view; } frame;
            layout(set = 3, binding = 0) uniform sampler2D orphan;
        "#;

        let mismatches = verify_spirv_against_glsl(drifted, &module_bytes()).unwrap();

        assert_eq!(
            mismatches,
            vec![
                DeclarationMismatch::CountDiffers {
                    set: 0,
                    binding: 1,
                    name: "shadowMaps".into(),
                    declared: 2,
                    reflected: DescriptorCount::Fixed(4),
                },
                DeclarationMismatch::ClassDiffers {
                    set: 1,
                    binding: 2,
                    name: "frame".into(),
                    declared: GlslDescriptorClass::StorageBlock,
                    reflected: DescriptorKind::UniformBuffer,
                },
                DeclarationMismatch::DeclaredButNotReflected {
                    set: 3,
                    binding: 0,
                    name: "orphan".into(),
                },
                DeclarationMismatch::ReflectedButNotDeclared {
                    set: 2,
                    binding: 0,
                    name: "particles".into(),
                },
            ]
        );
    }
}
