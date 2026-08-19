use std::path::{Path, PathBuf};

use thyllore_shader_manifest::{generate_pass_manifest_rust, PassManifest};

const SPIRV_DIR: &str = "assets/shaders";

fn main() {
    let workspace_root = workspace_root();
    let shader_dir = workspace_root.join("shaders");
    let manifest_path = shader_dir.join("passes.toml");
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    println!("cargo:rerun-if-changed={}", shader_dir.display());

    let manifest = read_manifest(&manifest_path, &shader_dir);
    let generated = generate_pass_manifest_rust(&manifest, SPIRV_DIR);
    let out_path = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"))
        .join("pass_manifest.rs");
    std::fs::write(&out_path, generated).unwrap_or_else(|error| {
        eprintln!("failed to write {}: {error}", out_path.display());
        std::process::exit(1);
    });
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/<name> lives two levels below the workspace root")
        .to_path_buf()
}

fn read_manifest(manifest_path: &Path, shader_dir: &Path) -> PassManifest {
    let text = std::fs::read_to_string(manifest_path).unwrap_or_else(|error| {
        eprintln!("failed to read {}: {error}", manifest_path.display());
        std::process::exit(1);
    });
    let manifest = PassManifest::parse(&text).unwrap_or_else(|error| {
        eprintln!("{}: {error}", manifest_path.display());
        std::process::exit(1);
    });
    if let Err(error) = manifest.validate_against_sources(shader_dir) {
        eprintln!("{}: {error}", manifest_path.display());
        std::process::exit(1);
    }
    manifest
}
