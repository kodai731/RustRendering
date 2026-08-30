use std::collections::BTreeMap;
use std::path::Path;

use thiserror::Error;
use thyllore_spirv_reflect::{reflect_shader_bytes, ReflectError, ReflectedBlock};

use crate::gpu_block_codegen::{
    generate_gpu_blocks_rust, GpuBlockCodegenConfig, GpuBlockCodegenError,
};

pub struct GpuBlockTarget {
    pub block_name: &'static str,
    pub output_path: &'static str,
    pub codegen_config: fn() -> GpuBlockCodegenConfig,
}

pub const GPU_BLOCK_TARGETS: &[GpuBlockTarget] = &[GpuBlockTarget {
    block_name: "FlameUBO",
    output_path: "crates/thyllore-effect-core/src/flame/gpu/components/generated.rs",
    codegen_config: flame_codegen_config,
}];

pub const REGENERATE_GPU_BLOCKS_COMMAND: &str =
    "cargo run -p thyllore-shader-manifest --bin generate_gpu_blocks";

#[derive(Debug, Error)]
pub enum FlameGpuBlocksError {
    #[error("read {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("reflect {path}: {source}")]
    Reflect { path: String, source: ReflectError },
    #[error("no SPIR-V under {spirv_dir} declares uniform block `{block}`")]
    BlockNotFound { spirv_dir: String, block: String },
    #[error("uniform block `{block}` differs between {first} and {second}")]
    BlockDiffers {
        block: String,
        first: String,
        second: String,
    },
    #[error(transparent)]
    Codegen(#[from] GpuBlockCodegenError),
}

fn flame_codegen_config() -> GpuBlockCodegenConfig {
    let mut extra_derives = BTreeMap::new();
    for name in ["FlameBranchElement", "FlameBranchField"] {
        extra_derives.insert(name.to_string(), vec!["Default".into(), "PartialEq".into()]);
    }
    extra_derives.insert("FlameBranchAgeProfile".into(), vec!["PartialEq".into()]);
    GpuBlockCodegenConfig {
        regenerate_command: REGENERATE_GPU_BLOCKS_COMMAND.into(),
        imports: vec!["cgmath::Matrix4".into()],
        extra_derives,
    }
}

pub fn gpu_blocks_source(
    spirv_dir: &Path,
    target: &GpuBlockTarget,
) -> Result<String, FlameGpuBlocksError> {
    let block = find_uniform_block(spirv_dir, target.block_name)?;
    Ok(generate_gpu_blocks_rust(
        &block,
        &(target.codegen_config)(),
    )?)
}

fn find_uniform_block(
    spirv_dir: &Path,
    block_name: &str,
) -> Result<ReflectedBlock, FlameGpuBlocksError> {
    let io_error = |path: &Path, source| FlameGpuBlocksError::Io {
        path: path.display().to_string(),
        source,
    };
    let mut paths: Vec<_> = std::fs::read_dir(spirv_dir)
        .map_err(|source| io_error(spirv_dir, source))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "spv"))
        .collect();
    paths.sort();

    let mut found: Option<(ReflectedBlock, &Path)> = None;
    for path in &paths {
        let bytes = std::fs::read(path).map_err(|source| io_error(path, source))?;
        let reflection =
            reflect_shader_bytes(&bytes).map_err(|source| FlameGpuBlocksError::Reflect {
                path: path.display().to_string(),
                source,
            })?;
        let Some(block) = reflection
            .bindings
            .into_iter()
            .filter_map(|binding| binding.block)
            .find(|block| block.type_name == block_name)
        else {
            continue;
        };

        match &found {
            None => found = Some((block, path)),
            Some((first, first_path)) if *first != block => {
                return Err(FlameGpuBlocksError::BlockDiffers {
                    block: block_name.to_string(),
                    first: first_path.display().to_string(),
                    second: path.display().to_string(),
                });
            }
            Some(_) => {}
        }
    }

    found
        .map(|(block, _)| block)
        .ok_or_else(|| FlameGpuBlocksError::BlockNotFound {
            spirv_dir: spirv_dir.display().to_string(),
            block: block_name.to_string(),
        })
}
