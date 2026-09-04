use std::path::Path;

use thyllore_shader_manifest::{gpu_blocks_source, GPU_BLOCK_TARGETS};

const SPIRV_DIR: &str = "assets/shaders";

fn main() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/<name> lives two levels below the workspace root");

    for target in GPU_BLOCK_TARGETS {
        let source = match gpu_blocks_source(&workspace_root.join(SPIRV_DIR), target) {
            Ok(source) => source,
            Err(error) => {
                eprintln!("{error}");
                eprintln!("build thyllore-vulkan-core first so {SPIRV_DIR} holds current SPIR-V");
                std::process::exit(1);
            }
        };

        let out_path = workspace_root.join(target.output_path);
        if let Err(error) = std::fs::write(&out_path, &source) {
            eprintln!("write {}: {error}", out_path.display());
            std::process::exit(1);
        }
        println!(
            "wrote {} ({} lines)",
            target.output_path,
            source.lines().count()
        );
    }
}
