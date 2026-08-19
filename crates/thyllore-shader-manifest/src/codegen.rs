use std::fmt::Write;

use crate::manifest::{PassDefinition, PassManifest};
use crate::naming::spirv_output_name;

pub fn generate_pass_manifest_rust(manifest: &PassManifest, spirv_dir: &str) -> String {
    let mut out = String::new();
    write_pass_id_enum(&mut out, manifest);
    for pass in &manifest.passes {
        write_pass_const(&mut out, pass, spirv_dir);
    }
    write_all_passes(&mut out, manifest);
    out
}

fn write_pass_id_enum(out: &mut String, manifest: &PassManifest) {
    out.push_str("#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]\npub enum PassId {\n");
    for pass in &manifest.passes {
        let _ = writeln!(out, "    {},", variant_name(&pass.name));
    }
    out.push_str(
        "}\n\nimpl PassId {\n    pub const fn name(self) -> &'static str {\n        match self {\n",
    );
    for pass in &manifest.passes {
        let _ = writeln!(
            out,
            "            PassId::{} => \"{}\",",
            variant_name(&pass.name),
            pass.name
        );
    }
    out.push_str("        }\n    }\n\n    pub const fn shaders(self) -> &'static PassShaders {\n        match self {\n");
    for pass in &manifest.passes {
        let _ = writeln!(
            out,
            "            PassId::{} => &{},",
            variant_name(&pass.name),
            const_name(&pass.name)
        );
    }
    out.push_str("        }\n    }\n}\n\n");
}

fn write_pass_const(out: &mut String, pass: &PassDefinition, spirv_dir: &str) {
    let _ = writeln!(
        out,
        "pub const {}: PassShaders = PassShaders {{\n    id: PassId::{},\n    stages: &[",
        const_name(&pass.name),
        variant_name(&pass.name)
    );
    for stage in &pass.stages {
        let spirv_name = spirv_output_name(&stage.source_file)
            .expect("manifest validation guarantees a shader extension");
        let _ = writeln!(
            out,
            "        ShaderFile {{ path: \"{spirv_dir}/{spirv_name}\", stage: ShaderStage::{} }},",
            stage.stage.reflect_variant()
        );
    }
    out.push_str("    ],\n    set_roles: &[");
    for (set, role) in &pass.sets {
        let _ = write!(out, "({set}, SetRole::{}), ", role.variant_name());
    }
    out.push_str("],\n};\n\n");
}

fn write_all_passes(out: &mut String, manifest: &PassManifest) {
    out.push_str("pub const ALL_PASSES: &[&PassShaders] = &[\n");
    for pass in &manifest.passes {
        let _ = writeln!(out, "    &{},", const_name(&pass.name));
    }
    out.push_str("];\n");
}

fn const_name(pass_name: &str) -> String {
    pass_name.to_ascii_uppercase()
}

fn variant_name(pass_name: &str) -> String {
    pass_name
        .split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_enum_consts_and_registry() {
        let manifest = PassManifest::parse(
            "[pass.onion_skin_ghost]\nstages = [\"gbufferVertex.vert\", \"onionSkinFragment.frag\"]\nsets = { 0 = \"frame\", 2 = \"object\" }\n",
        )
        .unwrap();
        let code = generate_pass_manifest_rust(&manifest, "assets/shaders");
        assert!(code.contains("pub enum PassId {\n    OnionSkinGhost,\n}"));
        assert!(code.contains("PassId::OnionSkinGhost => \"onion_skin_ghost\""));
        assert!(code.contains("pub const ONION_SKIN_GHOST: PassShaders"));
        assert!(code.contains(
            "ShaderFile { path: \"assets/shaders/gbufferVert.spv\", stage: ShaderStage::Vertex }"
        ));
        assert!(code.contains("set_roles: &[(0, SetRole::Frame), (2, SetRole::Object), ]"));
        assert!(
            code.contains("pub const ALL_PASSES: &[&PassShaders] = &[\n    &ONION_SKIN_GHOST,\n];")
        );
    }
}
