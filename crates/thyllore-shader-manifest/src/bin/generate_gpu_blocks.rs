use std::path::Path;

use thyllore_shader_manifest::{flame_gpu_blocks_source, FLAME_GPU_BLOCKS_PATH};

const SPIRV_DIR: &str = "assets/shaders";

fn main() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/<name> lives two levels below the workspace root");

    let source = match flame_gpu_blocks_source(&workspace_root.join(SPIRV_DIR)) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("{error}");
            eprintln!("build thyllore-vulkan-core first so {SPIRV_DIR} holds current SPIR-V");
            std::process::exit(1);
        }
    };

    let out_path = workspace_root.join(FLAME_GPU_BLOCKS_PATH);
    if let Err(error) = std::fs::write(&out_path, &source) {
        eprintln!("write {}: {error}", out_path.display());
        std::process::exit(1);
    }
    println!(
        "wrote {FLAME_GPU_BLOCKS_PATH} ({} lines)",
        source.lines().count()
    );
}
