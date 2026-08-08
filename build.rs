use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    compile_shaders();
}

fn compile_shaders() {
    // シェーダーディレクトリのパス
    let shader_src_dir = "shaders";
    let shader_out_dir = "assets/shaders";

    // シェーダーディレクトリが変更されたら再ビルド
    println!("cargo:rerun-if-changed={}", shader_src_dir);
    println!("cargo:rerun-if-env-changed=THYLLORE_FLAME_NOISE_ROT_DEG");
    println!("cargo:rerun-if-env-changed=THYLLORE_FLAME_WAVE_SEGMENTS");

    // ディレクトリ内の全ての.vertと.fragファイルを取得
    let entries = match fs::read_dir(shader_src_dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("シェーダーディレクトリの読み取りに失敗しました: {}", e);
            return;
        }
    };

    let mut shader_count = 0;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();

        // .vert、.frag、.geom、または.compファイルのみ処理
        if let Some(extension) = path.extension() {
            let ext_str = extension.to_str().unwrap_or("");
            if ext_str != "vert" && ext_str != "frag" && ext_str != "geom" && ext_str != "comp" {
                continue;
            }

            let file_name = path.file_name().unwrap().to_str().unwrap();
            let file_stem = path.file_stem().unwrap().to_str().unwrap();

            // 出力ファイル名を生成
            // ルール: ファイル名から"Vertex"/"vertex"、"Fragment"/"fragment"、
            // または"Geometry"/"geometry"を削除し、
            // 拡張子に応じて"Vert.spv"、"Frag.spv"、または"Geom.spv"を追加
            // 例: vertex.vert -> vert.spv (vertexを削除)
            //     gridVertex.vert -> gridVert.spv (末尾のVertexを削除)
            //     imguiFragment.frag -> imguiFrag.spv (末尾のFragmentを削除)
            let base_name = file_stem
                .trim_end_matches("Vertex")
                .trim_end_matches("vertex")
                .trim_end_matches("Fragment")
                .trim_end_matches("fragment")
                .trim_end_matches("Geometry")
                .trim_end_matches("geometry");

            let out_name = if file_name.ends_with(".vert") {
                if base_name.is_empty() {
                    "vert.spv".to_string()
                } else {
                    format!("{}Vert.spv", base_name)
                }
            } else if file_name.ends_with(".frag") {
                if base_name.is_empty() {
                    "frag.spv".to_string()
                } else {
                    format!("{}Frag.spv", base_name)
                }
            } else if file_name.ends_with(".geom") {
                if base_name.is_empty() {
                    "geom.spv".to_string()
                } else {
                    format!("{}Geom.spv", base_name)
                }
            } else if file_name.ends_with(".comp") {
                // コンピュートシェーダーは単純に{file_stem}.spvとする
                format!("{}.spv", file_stem)
            } else {
                continue;
            };

            let out_path = Path::new(shader_out_dir).join(&out_name);

            // 各シェーダーファイルが変更されたら再ビルド
            println!("cargo:rerun-if-changed={}", path.display());

            // シェーダーをコンパイル
            let mut cmd = Command::new("glslc");
            cmd.arg(path.to_str().unwrap());
            if let Ok(val) = std::env::var("THYLLORE_FLAME_NOISE_ROT_DEG") {
                cmd.arg(format!("-DFLAME_NOISE_ROT_DEG_OVERRIDE={}", val));
            }
            if let Ok(val) = std::env::var("THYLLORE_FLAME_WAVE_SEGMENTS") {
                cmd.arg(format!("-DFLAME_WAVE_SEGMENTS_OVERRIDE={}", val));
            }
            cmd.arg("-o").arg(out_path.to_str().unwrap());
            let output = cmd.output();

            match output {
                Ok(output) => {
                    if !output.status.success() {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        eprintln!("シェーダーコンパイルエラー ({}):\n{}", file_name, stderr);
                        std::process::exit(1);
                    } else {
                        // cargo:warning=を使うことでビルド時に表示される
                        println!(
                            "cargo:warning=シェーダーをコンパイルしました: {} -> {}",
                            file_name, out_name
                        );
                        shader_count += 1;
                    }
                }
                Err(e) => {
                    eprintln!("glslcの実行に失敗しました: {}", e);
                    eprintln!("VulkanSDKがインストールされ、glslcがPATHに含まれていることを確認してください。");
                    std::process::exit(1);
                }
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
