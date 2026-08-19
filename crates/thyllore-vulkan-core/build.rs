use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use thyllore_shader_manifest::{
    generate_pass_manifest_rust, generate_shader_bindings_rust, spirv_output_name, PassManifest,
};
use thyllore_spirv_reflect::{reflect_shader_bytes, verify_spirv_against_glsl, ShaderReflection};

const SPIRV_DIR: &str = "assets/shaders";

fn main() {
    let workspace_root = workspace_root();
    let shader_dir = workspace_root.join("shaders");
    let manifest_path = shader_dir.join("passes.toml");
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    println!("cargo:rerun-if-changed={}", shader_dir.display());
    println!("cargo:rerun-if-env-changed=THYLLORE_FLAME_NOISE_ROT_DEG");

    let manifest = read_manifest(&manifest_path, &shader_dir);
    let reflections = compile_shaders(&shader_dir, &workspace_root.join(SPIRV_DIR));

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    write_generated(
        &out_dir.join("pass_manifest.rs"),
        generate_pass_manifest_rust(&manifest, SPIRV_DIR),
    );
    let bindings = generate_shader_bindings_rust(&manifest, |source_file| {
        reflections.get(source_file).cloned()
    })
    .unwrap_or_else(|error| {
        eprintln!("shader binding generation failed: {error}");
        std::process::exit(1);
    });
    write_generated(&out_dir.join("shader_bindings.rs"), bindings);
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/<name> lives two levels below the workspace root")
        .to_path_buf()
}

fn write_generated(out_path: &Path, content: String) {
    std::fs::write(out_path, content).unwrap_or_else(|error| {
        eprintln!("failed to write {}: {error}", out_path.display());
        std::process::exit(1);
    });
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

fn glslc_command(source_path: &Path) -> Command {
    let mut cmd = Command::new("glslc");
    cmd.arg(source_path.to_str().unwrap());
    if let Ok(rotation) = std::env::var("THYLLORE_FLAME_NOISE_ROT_DEG") {
        cmd.arg(format!("-DFLAME_NOISE_ROT_DEG_OVERRIDE={}", rotation));
    }
    cmd
}

fn compile_shaders(shader_dir: &Path, spirv_dir: &Path) -> BTreeMap<String, ShaderReflection> {
    let entries = std::fs::read_dir(shader_dir).unwrap_or_else(|error| {
        eprintln!("failed to read {}: {error}", shader_dir.display());
        std::process::exit(1);
    });

    let mut reflections = BTreeMap::new();
    let mut expected_outputs = Vec::new();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let file_name = path.file_name().unwrap().to_str().unwrap().to_string();
        let Some(out_name) = spirv_output_name(&file_name) else {
            continue;
        };
        let out_path = spirv_dir.join(&out_name);
        expected_outputs.push(out_name);

        compile_shader(&path, &out_path);
        let reflection = verify_descriptor_declarations(&path, &out_path);
        reflections.insert(file_name, reflection);
    }

    remove_stale_spirv(spirv_dir, &expected_outputs);
    reflections
}

fn compile_shader(source_path: &Path, out_path: &Path) {
    let file_name = source_path.file_name().unwrap().to_str().unwrap();
    let mut cmd = glslc_command(source_path);
    cmd.arg("-o").arg(out_path.to_str().unwrap());
    match cmd.output() {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            eprintln!(
                "シェーダーコンパイルエラー ({}):\n{}",
                file_name,
                String::from_utf8_lossy(&output.stderr)
            );
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("glslcの実行に失敗しました: {}", error);
            eprintln!(
                "VulkanSDKがインストールされ、glslcがPATHに含まれていることを確認してください。"
            );
            std::process::exit(1);
        }
    }
}

fn verify_descriptor_declarations(source_path: &Path, spirv_path: &Path) -> ShaderReflection {
    let file_name = source_path.file_name().unwrap().to_str().unwrap();
    let preprocessed = match glslc_command(source_path).arg("-E").output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        Ok(output) => {
            eprintln!(
                "glslc -E に失敗しました ({}):\n{}",
                file_name,
                String::from_utf8_lossy(&output.stderr)
            );
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("glslc -E の実行に失敗しました ({}): {}", file_name, error);
            std::process::exit(1);
        }
    };
    let spirv = std::fs::read(spirv_path).unwrap_or_else(|error| {
        eprintln!(
            "SPIR-V の読み取りに失敗しました ({}): {}",
            spirv_path.display(),
            error
        );
        std::process::exit(1);
    });

    let mismatches = match verify_spirv_against_glsl(&preprocessed, &spirv) {
        Ok(mismatches) => mismatches,
        Err(error) => {
            eprintln!(
                "SPIR-V reflection に失敗しました ({}): {}\nglslc / SPIR-V の版が上がった可能性があります。crates/thyllore-spirv-reflect を更新してください。",
                file_name, error
            );
            std::process::exit(1);
        }
    };
    if !mismatches.is_empty() {
        eprintln!(
            "GLSL 宣言と SPIR-V reflection が一致しません ({}):",
            file_name
        );
        for mismatch in &mismatches {
            eprintln!("  {}", mismatch);
        }
        eprintln!("glslc の出力形式が変わったか、reflection parser が新しい構造を扱えていません。");
        std::process::exit(1);
    }

    reflect_shader_bytes(&spirv).expect("verified above")
}

fn remove_stale_spirv(spirv_dir: &Path, expected_outputs: &[String]) {
    let Ok(entries) = std::fs::read_dir(spirv_dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let is_spirv = path.extension().is_some_and(|extension| extension == "spv");
        let is_expected = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| expected_outputs.iter().any(|expected| expected == name));
        if is_spirv && !is_expected {
            if let Err(error) = std::fs::remove_file(&path) {
                eprintln!(
                    "古い SPIR-V の削除に失敗しました ({}): {}",
                    path.display(),
                    error
                );
            }
        }
    }
}
