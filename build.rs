use std::fs;
use std::path::Path;
use std::process::Command;

use thyllore_shader_manifest::spirv_output_name;
use thyllore_spirv_reflect::verify_spirv_against_glsl;

fn main() {
    compile_shaders();
}

fn glslc_command(source_path: &Path) -> Command {
    let mut cmd = Command::new("glslc");
    cmd.arg(source_path.to_str().unwrap());
    if let Ok(val) = std::env::var("THYLLORE_FLAME_NOISE_ROT_DEG") {
        cmd.arg(format!("-DFLAME_NOISE_ROT_DEG_OVERRIDE={}", val));
    }
    cmd
}

fn verify_descriptor_declarations(source_path: &Path, spirv_path: &Path) {
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
        Err(e) => {
            eprintln!("glslc -E の実行に失敗しました ({}): {}", file_name, e);
            std::process::exit(1);
        }
    };
    let spirv = fs::read(spirv_path).unwrap_or_else(|e| {
        eprintln!(
            "SPIR-V の読み取りに失敗しました ({}): {}",
            spirv_path.display(),
            e
        );
        std::process::exit(1);
    });

    let mismatches = match verify_spirv_against_glsl(&preprocessed, &spirv) {
        Ok(mismatches) => mismatches,
        Err(e) => {
            eprintln!(
                "SPIR-V reflection に失敗しました ({}): {}\nglslc / SPIR-V の版が上がった可能性があります。crates/thyllore-spirv-reflect を更新してください。",
                file_name, e
            );
            std::process::exit(1);
        }
    };
    if mismatches.is_empty() {
        return;
    }
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

fn compile_shaders() {
    let shader_src_dir = "shaders";
    let shader_out_dir = "assets/shaders";

    println!("cargo:rerun-if-changed={}", shader_src_dir);
    println!("cargo:rerun-if-env-changed=THYLLORE_FLAME_NOISE_ROT_DEG");

    let entries = match fs::read_dir(shader_src_dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("シェーダーディレクトリの読み取りに失敗しました: {}", e);
            return;
        }
    };

    let mut shader_count = 0;

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        let file_name = path.file_name().unwrap().to_str().unwrap();
        let Some(out_name) = spirv_output_name(file_name) else {
            continue;
        };
        let out_path = Path::new(shader_out_dir).join(&out_name);

        println!("cargo:rerun-if-changed={}", path.display());

        let mut cmd = glslc_command(&path);
        cmd.arg("-o").arg(out_path.to_str().unwrap());
        match cmd.output() {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    eprintln!("シェーダーコンパイルエラー ({}):\n{}", file_name, stderr);
                    std::process::exit(1);
                }
                println!(
                    "cargo:warning=シェーダーをコンパイルしました: {} -> {}",
                    file_name, out_name
                );
                verify_descriptor_declarations(&path, &out_path);
                shader_count += 1;
            }
            Err(e) => {
                eprintln!("glslcの実行に失敗しました: {}", e);
                eprintln!("VulkanSDKがインストールされ、glslcがPATHに含まれていることを確認してください。");
                std::process::exit(1);
            }
        }
    }

    if shader_count > 0 {
        println!(
            "cargo:warning={}個のシェーダーのコンパイルが完了しました。",
            shader_count
        );
    }
}
