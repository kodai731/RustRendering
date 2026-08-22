use std::collections::BTreeMap;
use std::fmt::Write;

use thiserror::Error;
use thyllore_spirv_reflect::{ReflectedBlock, ReflectedMember};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GpuBlockCodegenError {
    #[error("member `{member}` has SPIR-V type `{type_name}` which has no Rust mapping")]
    UnmappedType { member: String, type_name: String },
    #[error("padding run at offset {offset} in `{block}` is not 4-byte float scalars")]
    IrregularPadding { block: String, offset: u32 },
}

#[derive(Clone, Debug, Default)]
pub struct GpuBlockCodegenConfig {
    pub regenerate_command: String,
    pub imports: Vec<String>,
    pub extra_derives: BTreeMap<String, Vec<String>>,
}

const BASE_DERIVES: &str = "Clone, Copy, Debug";
const PADDING_SCALAR: &str = "float32";

pub fn generate_gpu_blocks_rust(
    block: &ReflectedBlock,
    config: &GpuBlockCodegenConfig,
) -> Result<String, GpuBlockCodegenError> {
    let mut nested = BTreeMap::new();
    collect_nested_structs(&block.members, &mut nested);

    let mut out = String::new();
    let _ = writeln!(
        out,
        "// Generated from SPIR-V by `{}`; do not edit.",
        config.regenerate_command
    );
    for import in &config.imports {
        let _ = writeln!(out, "use {import};");
    }
    let _ = writeln!(out, "use thyllore_spirv_reflect::declare_gpu_block;");
    for (name, members) in &nested {
        write_struct(&mut out, name, members, &nested, config)?;
    }
    write_struct(&mut out, &block.type_name, &block.members, &nested, config)?;
    Ok(out)
}

fn collect_nested_structs(
    members: &[ReflectedMember],
    structs: &mut BTreeMap<String, Vec<ReflectedMember>>,
) {
    for member in members {
        if member.members.is_empty() {
            continue;
        }
        let name = element_type_name(&member.type_name).to_string();
        if !structs.contains_key(&name) {
            structs.insert(name, member.members.clone());
            collect_nested_structs(&member.members, structs);
        }
    }
}

fn element_type_name(type_name: &str) -> &str {
    type_name.split('[').next().unwrap_or(type_name)
}

fn write_struct(
    out: &mut String,
    name: &str,
    members: &[ReflectedMember],
    nested: &BTreeMap<String, Vec<ReflectedMember>>,
    config: &GpuBlockCodegenConfig,
) -> Result<(), GpuBlockCodegenError> {
    let mut derives = BASE_DERIVES.to_string();
    for extra in config.extra_derives.get(name).into_iter().flatten() {
        derives.push_str(", ");
        derives.push_str(extra);
    }

    let _ = writeln!(out, "\ndeclare_gpu_block! {{");
    let _ = writeln!(out, "    #[derive({derives})]");
    let _ = writeln!(out, "    pub struct {name} {{");
    let padding_runs = count_padding_runs(members);
    let mut padding_index = 0;
    let mut index = 0;
    while index < members.len() {
        let member = &members[index];
        if is_padding(&member.name) {
            let run = padding_run_length(name, &members[index..])?;
            let field = padding_field_name(padding_runs, padding_index);
            let _ = writeln!(out, "        pub {field}: {},", padding_type(run));
            padding_index += 1;
            index += run;
            continue;
        }
        let field = snake_case(&member.name);
        let rust_type = rust_type(member, nested)?;
        let nested_suffix = if member.members.is_empty() {
            String::new()
        } else {
            format!(" = nested {}", element_type_name(&member.type_name))
        };
        let _ = writeln!(out, "        pub {field}: {rust_type}{nested_suffix},");
        index += 1;
    }
    let _ = writeln!(out, "    }}\n}}");
    Ok(())
}

fn padding_run_length(
    block: &str,
    members: &[ReflectedMember],
) -> Result<usize, GpuBlockCodegenError> {
    let run = members
        .iter()
        .take_while(|member| is_padding(&member.name))
        .count();
    for member in &members[..run] {
        if member.type_name != PADDING_SCALAR {
            return Err(GpuBlockCodegenError::IrregularPadding {
                block: block.to_string(),
                offset: member.offset,
            });
        }
    }
    Ok(run)
}

fn count_padding_runs(members: &[ReflectedMember]) -> usize {
    members
        .iter()
        .enumerate()
        .filter(|(index, member)| {
            is_padding(&member.name)
                && index
                    .checked_sub(1)
                    .is_none_or(|prev| !is_padding(&members[prev].name))
        })
        .count()
}

fn padding_field_name(runs: usize, index: usize) -> String {
    if runs == 1 {
        "_padding".to_string()
    } else {
        format!("_padding{index}")
    }
}

fn padding_type(run: usize) -> String {
    if run == 1 {
        "f32".to_string()
    } else {
        format!("[f32; {run}]")
    }
}

fn rust_type(
    member: &ReflectedMember,
    nested: &BTreeMap<String, Vec<ReflectedMember>>,
) -> Result<String, GpuBlockCodegenError> {
    map_type(&member.type_name, nested).ok_or_else(|| GpuBlockCodegenError::UnmappedType {
        member: member.name.clone(),
        type_name: member.type_name.clone(),
    })
}

fn map_type(type_name: &str, nested: &BTreeMap<String, Vec<ReflectedMember>>) -> Option<String> {
    if let Some((element, length)) = type_name.rsplit_once('[') {
        let length = length.strip_suffix(']')?;
        return Some(format!("[{}; {length}]", map_type(element, nested)?));
    }
    let scalar = match type_name {
        "float32" => "f32",
        "int32" => "i32",
        "uint32" => "u32",
        "bool" => "u32",
        "float32x2" => "[f32; 2]",
        "float32x3" => "[f32; 3]",
        "float32x4" => "[f32; 4]",
        "float32x4x4" => "Matrix4<f32>",
        other if nested.contains_key(other) => other,
        _ => return None,
    };
    Some(scalar.to_string())
}

fn is_padding(name: &str) -> bool {
    name.trim_start_matches('_').starts_with("pad")
}

fn snake_case(name: &str) -> String {
    let characters: Vec<char> = name.chars().collect();
    let mut out = String::with_capacity(name.len() + 4);
    for (index, &current) in characters.iter().enumerate() {
        if current.is_ascii_uppercase() && starts_word(&characters, index) {
            out.push('_');
        }
        out.push(current.to_ascii_lowercase());
    }
    out
}

fn starts_word(characters: &[char], index: usize) -> bool {
    let Some(previous) = index.checked_sub(1).map(|prev| characters[prev]) else {
        return false;
    };
    let follows_lowercase = previous.is_ascii_lowercase() || previous.is_ascii_digit();
    let ends_acronym = previous.is_ascii_uppercase()
        && characters
            .get(index + 1)
            .is_some_and(|next| next.is_ascii_lowercase());
    follows_lowercase || ends_acronym
}

#[cfg(test)]
mod tests {
    use super::*;

    fn member(name: &str, offset: u32, size: u32, type_name: &str) -> ReflectedMember {
        ReflectedMember {
            name: name.into(),
            offset,
            size,
            type_name: type_name.into(),
            members: Vec::new(),
        }
    }

    fn inner(base: u32) -> Vec<ReflectedMember> {
        vec![
            member("rgb", base, 12, "float32x3"),
            member("occlusionLumRef", base + 12, 4, "float32"),
        ]
    }

    fn block() -> ReflectedBlock {
        let mut color = member("colorBase", 64, 16, "Color");
        color.members = inner(64);
        let mut modes = member("modes", 80, 32, "Color[2]");
        modes.members = inner(80);
        ReflectedBlock {
            type_name: "Outer".into(),
            size: 144,
            members: vec![
                member("view", 0, 64, "float32x4x4"),
                color,
                modes,
                member("warpYScale", 112, 16, "float32x4"),
                member("pad0", 128, 4, "float32"),
                member("pad1", 132, 4, "float32"),
                member("wienCK", 136, 4, "float32"),
                member("_pad2", 140, 4, "float32"),
            ],
        }
    }

    #[test]
    fn generates_nested_structs_before_the_block() {
        let mut config = GpuBlockCodegenConfig {
            regenerate_command: "cargo run --bin gen".into(),
            imports: vec!["cgmath::Matrix4".into()],
            extra_derives: BTreeMap::new(),
        };
        config
            .extra_derives
            .insert("Color".into(), vec!["PartialEq".into()]);

        let generated = generate_gpu_blocks_rust(&block(), &config).unwrap();
        assert_eq!(
            generated,
            "// Generated from SPIR-V by `cargo run --bin gen`; do not edit.\n\
             use cgmath::Matrix4;\n\
             use thyllore_spirv_reflect::declare_gpu_block;\n\
             \n\
             declare_gpu_block! {\n\
             \x20   #[derive(Clone, Copy, Debug, PartialEq)]\n\
             \x20   pub struct Color {\n\
             \x20       pub rgb: [f32; 3],\n\
             \x20       pub occlusion_lum_ref: f32,\n\
             \x20   }\n\
             }\n\
             \n\
             declare_gpu_block! {\n\
             \x20   #[derive(Clone, Copy, Debug)]\n\
             \x20   pub struct Outer {\n\
             \x20       pub view: Matrix4<f32>,\n\
             \x20       pub color_base: Color = nested Color,\n\
             \x20       pub modes: [Color; 2] = nested Color,\n\
             \x20       pub warp_y_scale: [f32; 4],\n\
             \x20       pub _padding0: [f32; 2],\n\
             \x20       pub wien_ck: f32,\n\
             \x20       pub _padding1: f32,\n\
             \x20   }\n\
             }\n"
        );
    }

    #[test]
    fn rejects_unmapped_types_and_irregular_padding() {
        let mut unmapped = block();
        unmapped.members[0].type_name = "float64".into();
        assert_eq!(
            generate_gpu_blocks_rust(&unmapped, &GpuBlockCodegenConfig::default()),
            Err(GpuBlockCodegenError::UnmappedType {
                member: "view".into(),
                type_name: "float64".into()
            })
        );

        let mut irregular = block();
        irregular.members[4].type_name = "int32".into();
        assert_eq!(
            generate_gpu_blocks_rust(&irregular, &GpuBlockCodegenConfig::default()),
            Err(GpuBlockCodegenError::IrregularPadding {
                block: "Outer".into(),
                offset: 128
            })
        );
    }
}
