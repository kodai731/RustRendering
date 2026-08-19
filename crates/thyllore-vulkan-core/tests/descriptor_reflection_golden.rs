use std::path::{Path, PathBuf};

use thyllore_effect_core::flame::analytic::ubo::FlameUBO;
use thyllore_render_core::{FrameUBO, MaterialUBO, ObjectUBO};
use thyllore_vulkan_core::data::{SceneUniformData, UniformBufferObject};
use thyllore_vulkan_core::descriptor::{
    reflect_shader_bytes, DescriptorSetTable, LayoutMismatch, PassId, PassShaders,
    ReflectedLayoutSpec, SelectionUBO, ShaderFile, ShaderReflection, ALL_PASSES,
};

#[derive(Clone, Copy, Debug)]
enum BlockCoverage {
    Exact,
    ShaderReadsPrefix,
}

struct BlockGolden {
    pass: PassId,
    set: u32,
    binding: u32,
    rust_type: &'static str,
    rust_size: usize,
    coverage: BlockCoverage,
}

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

fn block_goldens() -> Vec<BlockGolden> {
    vec![
        BlockGolden {
            pass: PassId::Model,
            set: 0,
            binding: 0,
            rust_type: "FrameUBO",
            rust_size: std::mem::size_of::<FrameUBO>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: PassId::Model,
            set: 1,
            binding: 1,
            rust_type: "MaterialUBO",
            rust_size: std::mem::size_of::<MaterialUBO>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: PassId::Model,
            set: 2,
            binding: 0,
            rust_type: "ObjectUBO",
            rust_size: std::mem::size_of::<ObjectUBO>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: PassId::FlameResolve,
            set: 1,
            binding: 0,
            rust_type: "FlameUBO",
            rust_size: std::mem::size_of::<FlameUBO>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: PassId::Tonemap,
            set: 0,
            binding: 3,
            rust_type: "SceneUniformData",
            rust_size: std::mem::size_of::<SceneUniformData>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: PassId::RayQueryShadow,
            set: 0,
            binding: 4,
            rust_type: "SceneUniformData",
            rust_size: std::mem::size_of::<SceneUniformData>(),
            coverage: BlockCoverage::ShaderReadsPrefix,
        },
        BlockGolden {
            pass: PassId::Composite,
            set: 0,
            binding: 4,
            rust_type: "SceneUniformData",
            rust_size: std::mem::size_of::<SceneUniformData>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: PassId::Composite,
            set: 0,
            binding: 6,
            rust_type: "SelectionUBO",
            rust_size: std::mem::size_of::<SelectionUBO>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: PassId::Billboard,
            set: 0,
            binding: 0,
            rust_type: "UniformBufferObject",
            rust_size: std::mem::size_of::<UniformBufferObject>(),
            coverage: BlockCoverage::Exact,
        },
    ]
}

fn check_block_coverage(golden: &BlockGolden, block_name: &str, block_size: u32) -> Option<String> {
    let block_size = block_size as usize;
    let padded_block_size = block_size.div_ceil(16) * 16;
    let covers = match golden.coverage {
        BlockCoverage::Exact => {
            golden.rust_size >= block_size && golden.rust_size <= padded_block_size
        }
        BlockCoverage::ShaderReadsPrefix => golden.rust_size >= block_size,
    };
    if covers {
        return None;
    }
    Some(format!(
        "{}: shader block `{}` is {} bytes (padded {}), Rust `{}` is {} bytes ({:?})",
        golden.pass.name(),
        block_name,
        block_size,
        padded_block_size,
        golden.rust_type,
        golden.rust_size,
        golden.coverage
    ))
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
fn rust_uniform_structs_cover_shader_blocks() {
    enter_workspace_root();
    let mut failures = Vec::new();

    for golden in block_goldens() {
        let table = build_table(golden.pass.shaders());
        let Some(binding) = table.binding(golden.set, golden.binding) else {
            failures.push(format!(
                "{}: set {} binding {} is not declared by its shaders",
                golden.pass.name(),
                golden.set,
                golden.binding
            ));
            continue;
        };
        let Some(block_size) = binding.block_size else {
            failures.push(format!(
                "{}: set {} binding {} ({}) is not a buffer block",
                golden.pass.name(),
                golden.set,
                golden.binding,
                binding.name
            ));
            continue;
        };

        if let Some(failure) = check_block_coverage(&golden, &binding.name, block_size) {
            failures.push(failure);
        }
    }

    assert!(
        failures.is_empty(),
        "uniform block size drift against SPIR-V:\n{}",
        failures.join("\n")
    );
}
