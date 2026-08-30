use std::path::{Path, PathBuf};

use thyllore_effect_core::FlameUBO;
use thyllore_render_core::{FrameUBO, MaterialUBO, ObjectUBO};
use thyllore_shader_manifest::{
    gpu_blocks_source, GpuBlockTarget, GPU_BLOCK_TARGETS, REGENERATE_GPU_BLOCKS_COMMAND,
};
use thyllore_spirv_reflect::{
    compare_block_layout, BlockCoverage, DescriptorKind, GpuBlock, LayoutDifference, ReflectedBlock,
};
use thyllore_vulkan_core::data::{SceneUniformData, UniformBufferObject};
use thyllore_vulkan_core::descriptor::{
    reflect_shader_bytes, DescriptorSetTable, LayoutMismatch, PassId, PassShaders,
    ReflectedLayoutSpec, SelectionUBO, ShaderFile, ShaderReflection, ALL_PASSES,
};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn enter_workspace_root() {
    std::env::set_current_dir(workspace_root()).expect("chdir to workspace root");
}

fn load_reflection(shader: &ShaderFile) -> ShaderReflection {
    let path = workspace_root().join(shader.path);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read {} (run cargo build first): {error}", path.display()));
    let reflection = reflect_shader_bytes(&bytes)
        .unwrap_or_else(|error| panic!("reflect {}: {error}", shader.path));
    assert_eq!(
        reflection.stages,
        [shader.stage],
        "{}: passes.toml stage differs from the SPIR-V entry point",
        shader.path
    );
    reflection
}

fn build_table(pass: &PassShaders) -> DescriptorSetTable {
    let reflections: Vec<ShaderReflection> = pass.stages.iter().map(load_reflection).collect();
    DescriptorSetTable::from_reflections(&reflections)
        .unwrap_or_else(|error| panic!("{}: merge shader stages: {error}", pass.name()))
}

struct RustBlock {
    glsl_name: &'static str,
    compare: fn(&ReflectedBlock, BlockCoverage) -> Vec<LayoutDifference>,
}

fn rust_block<T: GpuBlock>(glsl_name: &'static str) -> RustBlock {
    RustBlock {
        glsl_name,
        compare: compare_block_layout::<T>,
    }
}

fn rust_blocks() -> Vec<RustBlock> {
    vec![
        rust_block::<FrameUBO>("FrameUBO"),
        rust_block::<MaterialUBO>("MaterialUBO"),
        rust_block::<ObjectUBO>("ObjectUBO"),
        rust_block::<FlameUBO>("FlameUBO"),
        rust_block::<SceneUniformData>("SceneData"),
        rust_block::<SelectionUBO>("SelectionData"),
        rust_block::<UniformBufferObject>("UniformBufferObject"),
    ]
}

fn block_coverage(pass: &PassShaders, set: u32, binding: u32) -> BlockCoverage {
    if (pass.id, set, binding) == (PassId::RayQueryShadow, 0, 4) {
        BlockCoverage::ShaderReadsPrefix
    } else {
        BlockCoverage::Exact
    }
}

fn describe_mismatch(pass: &str, set: u32, mismatch: &LayoutMismatch) -> String {
    match mismatch {
        LayoutMismatch::MissingInLayout {
            binding,
            shader_kind,
        } => format!("{pass} set {set} binding {binding}: shader declares {shader_kind:?} but the layout has no such binding"),
        LayoutMismatch::TypeDiffers {
            binding,
            shader_kind,
            layout_type,
        } => format!("{pass} set {set} binding {binding}: shader {shader_kind:?} vs layout {layout_type:?}"),
        LayoutMismatch::CountDiffers {
            binding,
            shader_count,
            layout_count,
        } => format!("{pass} set {set} binding {binding}: shader count {shader_count:?} vs layout count {layout_count}"),
        LayoutMismatch::StageMissing {
            binding,
            shader_stages,
            layout_stages,
        } => format!("{pass} set {set} binding {binding}: shader stages {shader_stages:?} not covered by layout stages {layout_stages:?}"),
        LayoutMismatch::UnusedInShaders { binding } => {
            format!("{pass} set {set} binding {binding}: layout binding is not declared by any shader of this pass")
        }
    }
}

#[test]
fn every_pass_layout_covers_its_shaders() {
    enter_workspace_root();
    let mut failures = Vec::new();

    for pass in ALL_PASSES {
        let pass_table = build_table(pass);
        let specs: Vec<(u32, ReflectedLayoutSpec)> = pass
            .set_roles
            .iter()
            .map(|(set, role)| (*set, ReflectedLayoutSpec::for_role(pass, *role)))
            .collect();

        for (set, spec) in &specs {
            match spec.set_index() {
                Ok(index) if index == *set => {}
                Ok(index) => failures.push(format!(
                    "{}: layout spec targets set {index} but the pass binds it at set {set}",
                    pass.name()
                )),
                Err(error) => failures.push(format!("{} set {set}: {error:#}", pass.name())),
            }
            match spec.resolve_bindings() {
                Ok(layout) => {
                    for mismatch in pass_table.verify_layout(*set, &layout) {
                        if matches!(mismatch, LayoutMismatch::UnusedInShaders { .. }) {
                            continue;
                        }
                        failures.push(describe_mismatch(pass.name(), *set, &mismatch));
                    }
                }
                Err(error) => failures.push(format!("{} set {set}: {error:#}", pass.name())),
            }
        }

        for set in pass_table.set_indices() {
            if !specs.iter().any(|(declared, _)| *declared == set) {
                failures.push(format!(
                    "{}: shaders use descriptor set {set} but passes.toml binds no role for it",
                    pass.name()
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "descriptor layout drift against SPIR-V:\n{}",
        failures.join("\n")
    );
}

#[test]
fn rust_uniform_structs_match_every_shader_block_member() {
    enter_workspace_root();
    let rust_blocks = rust_blocks();
    let mut failures = Vec::new();

    for pass in ALL_PASSES {
        let table = build_table(pass);
        for set in table.set_indices() {
            let Some(bindings) = table.bindings(set) else {
                continue;
            };
            for (binding_index, merged) in bindings {
                if merged.kind != DescriptorKind::UniformBuffer {
                    continue;
                }
                let Some(block) = &merged.block else {
                    continue;
                };
                let location = format!(
                    "{} (set={set} binding={binding_index}) `{}`",
                    pass.name(),
                    block.type_name
                );
                let Some(rust) = rust_blocks
                    .iter()
                    .find(|rust| rust.glsl_name == block.type_name)
                else {
                    failures.push(format!("{location}: no Rust GpuBlock registered"));
                    continue;
                };

                let coverage = block_coverage(pass, set, *binding_index);
                let differences = (rust.compare)(block, coverage);
                if !differences.is_empty() {
                    let listed: Vec<String> =
                        differences.iter().map(|d| format!("  {d}")).collect();
                    failures.push(format!("{location}:\n{}", listed.join("\n")));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "uniform block layout drift against SPIR-V:\n{}",
        failures.join("\n")
    );
}

fn first_line_difference(left: &str, right: &str) -> Option<(usize, String, String)> {
    let mut left_lines = left.lines();
    let mut right_lines = right.lines();
    let mut line_number = 1;
    loop {
        match (left_lines.next(), right_lines.next()) {
            (None, None) => return None,
            (l, r) if l == r => line_number += 1,
            (l, r) => {
                return Some((
                    line_number,
                    l.unwrap_or("<end>").to_string(),
                    r.unwrap_or("<end>").to_string(),
                ))
            }
        }
    }
}

#[test]
fn generated_flame_gpu_blocks_match_spirv() {
    enter_workspace_root();
    let mut failures: Vec<String> = Vec::new();

    for target in GPU_BLOCK_TARGETS {
        let generated = gpu_blocks_source(Path::new("assets/shaders"), target)
            .unwrap_or_else(|error| panic!("generate {} gpu blocks: {error}", target.block_name));
        let checked_in = std::fs::read_to_string(target.output_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", target.output_path));

        if let Some((line, expected, current)) = first_line_difference(&generated, &checked_in) {
            failures.push(format!(
                "{} is stale at line {line}; run `{REGENERATE_GPU_BLOCKS_COMMAND}`\n  generated : {expected}\n  checked in: {current}",
                target.output_path
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "generated gpu blocks differ from checked-in files:\n{}",
        failures.join("\n")
    );
}
