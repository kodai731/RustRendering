use std::path::{Path, PathBuf};

use thyllore_effect_core::flame::analytic::ubo::FlameUBO;
use thyllore_render_core::{FrameUBO, MaterialUBO, ObjectUBO};
use thyllore_vulkan_core::data::{SceneUniformData, UniformBufferObject};
use thyllore_vulkan_core::descriptor::{
    reflect_shader_bytes, DescriptorSetTable, DescriptorTypeOverride, FrameDescriptorSet,
    LayoutMismatch, MaterialManager, ObjectDescriptorSet, RRAutoExposureAverageDescriptorSet,
    RRAutoExposureHistogramDescriptorSet, RRBillboardDescriptorSet, RRBloomDescriptorSets,
    RRCompositeDescriptorSet, RRDofDescriptorSet, RRFlameDescriptorSet, RRRayQueryDescriptorSet,
    RRToneMapDescriptorSet, ReflectedSetLayout, SelectionUBO, ShaderReflection,
    FLAME_DESCRIPTOR_SET, FLAME_RESOLVE_SHADERS,
};
use thyllore_vulkan_core::resource::OnionSkinPassResources;
use vulkanalia::vk;

enum LayoutSource {
    Handwritten(Vec<vk::DescriptorSetLayoutBinding>),
    Reflected(Vec<DescriptorTypeOverride>),
}

struct PassGolden {
    name: &'static str,
    shaders: Vec<String>,
    sets: Vec<(u32, LayoutSource)>,
}

#[derive(Clone, Copy, Debug)]
enum BlockCoverage {
    Exact,
    ShaderReadsPrefix,
}

struct BlockGolden {
    pass: &'static str,
    set: u32,
    binding: u32,
    rust_type: &'static str,
    rust_size: usize,
    coverage: BlockCoverage,
}

const STANDARD_GRAPHICS_SHADERS: &[&str] = &[
    "vert.spv",
    "frag.spv",
    "gbufferVert.spv",
    "gbufferFrag.spv",
    "gridVert.spv",
    "gridFrag.spv",
    "gizmoVert.spv",
    "gizmoFrag.spv",
    "boneVert.spv",
    "boneFrag.spv",
    "onionSkinFrag.spv",
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn shader_assets(file_names: &[&str]) -> Vec<String> {
    file_names
        .iter()
        .map(|file_name| format!("assets/shaders/{file_name}"))
        .collect()
}

fn load_reflection(shader_path: &str) -> ShaderReflection {
    let path = workspace_root().join(shader_path);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read {} (run cargo build first): {error}", path.display()));
    reflect_shader_bytes(&bytes).unwrap_or_else(|error| panic!("reflect {shader_path}: {error}"))
}

fn build_table(shaders: &[String]) -> DescriptorSetTable {
    let reflections: Vec<ShaderReflection> = shaders
        .iter()
        .map(|shader| load_reflection(shader))
        .collect();
    DescriptorSetTable::from_reflections(&reflections).expect("merge shader stages")
}

fn pass_goldens() -> Vec<PassGolden> {
    vec![
        PassGolden {
            name: "standard_graphics",
            shaders: shader_assets(STANDARD_GRAPHICS_SHADERS),
            sets: vec![
                (
                    0,
                    LayoutSource::Handwritten(FrameDescriptorSet::layout_bindings()),
                ),
                (
                    1,
                    LayoutSource::Handwritten(MaterialManager::layout_bindings()),
                ),
                (
                    2,
                    LayoutSource::Handwritten(ObjectDescriptorSet::layout_bindings()),
                ),
            ],
        },
        PassGolden {
            name: "flame_resolve",
            shaders: FLAME_RESOLVE_SHADERS
                .iter()
                .map(|path| path.to_string())
                .collect(),
            sets: vec![
                (
                    0,
                    LayoutSource::Handwritten(FrameDescriptorSet::layout_bindings()),
                ),
                (
                    FLAME_DESCRIPTOR_SET,
                    LayoutSource::Reflected(
                        RRFlameDescriptorSet::descriptor_type_overrides().to_vec(),
                    ),
                ),
            ],
        },
        PassGolden {
            name: "tonemap",
            shaders: shader_assets(&["tonemapVert.spv", "tonemapFrag.spv"]),
            sets: vec![(
                0,
                LayoutSource::Handwritten(RRToneMapDescriptorSet::layout_bindings()),
            )],
        },
        PassGolden {
            name: "bloom",
            shaders: shader_assets(&[
                "tonemapVert.spv",
                "bloomDownsampleFrag.spv",
                "bloomUpsampleFrag.spv",
            ]),
            sets: vec![(
                0,
                LayoutSource::Handwritten(RRBloomDescriptorSets::layout_bindings()),
            )],
        },
        PassGolden {
            name: "dof",
            shaders: shader_assets(&["tonemapVert.spv", "dofFrag.spv"]),
            sets: vec![(
                0,
                LayoutSource::Handwritten(RRDofDescriptorSet::layout_bindings()),
            )],
        },
        PassGolden {
            name: "auto_exposure_histogram",
            shaders: shader_assets(&["autoExposureHistogram.spv"]),
            sets: vec![(
                0,
                LayoutSource::Handwritten(RRAutoExposureHistogramDescriptorSet::layout_bindings()),
            )],
        },
        PassGolden {
            name: "auto_exposure_average",
            shaders: shader_assets(&["autoExposureAverage.spv"]),
            sets: vec![(
                0,
                LayoutSource::Handwritten(RRAutoExposureAverageDescriptorSet::layout_bindings()),
            )],
        },
        PassGolden {
            name: "ray_query_shadow",
            shaders: shader_assets(&["rayQueryShadow.spv"]),
            sets: vec![(
                0,
                LayoutSource::Handwritten(RRRayQueryDescriptorSet::layout_bindings()),
            )],
        },
        PassGolden {
            name: "composite",
            shaders: shader_assets(&["compositeVert.spv", "compositeFrag.spv"]),
            sets: vec![(
                0,
                LayoutSource::Handwritten(RRCompositeDescriptorSet::layout_bindings()),
            )],
        },
        PassGolden {
            name: "billboard",
            shaders: shader_assets(&["billboardVert.spv", "billboardFrag.spv"]),
            sets: vec![(
                0,
                LayoutSource::Handwritten(RRBillboardDescriptorSet::layout_bindings()),
            )],
        },
        PassGolden {
            name: "onion_skin_composite",
            shaders: shader_assets(&["tonemapVert.spv", "onionSkinCompositeFrag.spv"]),
            sets: vec![(
                0,
                LayoutSource::Handwritten(OnionSkinPassResources::composite_layout_bindings()),
            )],
        },
    ]
}

fn block_goldens() -> Vec<BlockGolden> {
    vec![
        BlockGolden {
            pass: "standard_graphics",
            set: 0,
            binding: 0,
            rust_type: "FrameUBO",
            rust_size: std::mem::size_of::<FrameUBO>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: "standard_graphics",
            set: 1,
            binding: 1,
            rust_type: "MaterialUBO",
            rust_size: std::mem::size_of::<MaterialUBO>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: "standard_graphics",
            set: 2,
            binding: 0,
            rust_type: "ObjectUBO",
            rust_size: std::mem::size_of::<ObjectUBO>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: "flame_resolve",
            set: 1,
            binding: 0,
            rust_type: "FlameUBO",
            rust_size: std::mem::size_of::<FlameUBO>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: "tonemap",
            set: 0,
            binding: 3,
            rust_type: "SceneUniformData",
            rust_size: std::mem::size_of::<SceneUniformData>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: "ray_query_shadow",
            set: 0,
            binding: 4,
            rust_type: "SceneUniformData",
            rust_size: std::mem::size_of::<SceneUniformData>(),
            coverage: BlockCoverage::ShaderReadsPrefix,
        },
        BlockGolden {
            pass: "composite",
            set: 0,
            binding: 4,
            rust_type: "SceneUniformData",
            rust_size: std::mem::size_of::<SceneUniformData>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: "composite",
            set: 0,
            binding: 6,
            rust_type: "SelectionUBO",
            rust_size: std::mem::size_of::<SelectionUBO>(),
            coverage: BlockCoverage::Exact,
        },
        BlockGolden {
            pass: "billboard",
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
        golden.pass,
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
            format!("{pass} set {set} binding {binding}: layout binding is not declared by any shader")
        }
    }
}

#[test]
fn handwritten_layouts_match_spirv_reflection() {
    let mut failures = Vec::new();

    for pass in pass_goldens() {
        let table = build_table(&pass.shaders);
        for (set, source) in &pass.sets {
            match source {
                LayoutSource::Handwritten(layout) => {
                    for mismatch in table.verify_layout(*set, layout) {
                        failures.push(describe_mismatch(pass.name, *set, &mismatch));
                    }
                }
                LayoutSource::Reflected(overrides) => {
                    if let Err(error) =
                        ReflectedSetLayout::resolve_bindings(&table, *set, overrides)
                    {
                        failures.push(format!("{} set {set}: {error:#}", pass.name));
                    }
                }
            }
        }

        let declared_sets: Vec<u32> = pass.sets.iter().map(|(set, _)| *set).collect();
        for set in table.set_indices() {
            if !declared_sets.contains(&set) {
                failures.push(format!(
                    "{}: shaders use descriptor set {set} but the pass binds no layout for it",
                    pass.name
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
    let mut failures = Vec::new();

    for golden in block_goldens() {
        let pass = pass_goldens()
            .into_iter()
            .find(|pass| pass.name == golden.pass)
            .expect("block golden refers to a known pass");
        let table = build_table(&pass.shaders);
        let Some(binding) = table.binding(golden.set, golden.binding) else {
            failures.push(format!(
                "{}: set {} binding {} is not declared by its shaders",
                golden.pass, golden.set, golden.binding
            ));
            continue;
        };
        let Some(block_size) = binding.block_size else {
            failures.push(format!(
                "{}: set {} binding {} ({}) is not a buffer block",
                golden.pass, golden.set, golden.binding, binding.name
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
