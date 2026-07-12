//! MVP (-Variant lite) addon ZIP structural smoke.
//!
//! Phase 5.5 Step A: this test runs in CI on all three platforms and proves
//! that the lite Variant build produces a structurally correct extension ZIP
//! before any Blender install / Operator execution. The Blender-side pieces
//! (curve_copilot Operator launch, ABI handshake, FCurve assertion) are added
//! in Step B once the test_to_motion PyO3 rewrite + light_t2m ONNX land --
//! at that point the file gains an additional `#[ignore]`-d Blender-driven
//! test alongside the structural one.
//!
//! Verifies on a freshly built lite ZIP:
//!   1. ``scripts/build_blender_addon.ps1 -Variant lite`` succeeds.
//!   2. The resulting ZIP exists at the expected path.
//!   3. ZIP size is below 50 MB (Blender Market / Gumroad upload-friendly).
//!   4. ``blender_manifest.toml`` inside the ZIP carries
//!      ``id = "thyllore_animation_lite"`` and the GPL SPDX.
//!   5. ``LICENSE.md`` and ``EULA.md`` are present.
//!   6. Tier A files are absent (operators/auto_rig.py, text_to_mesh.py,
//!      _grpc_helpers.py, grpc_client/, license/).
//!   7. text_to_motion.py is absent (gRPC implementation excluded until the
//!      PyO3 rewrite ships).
//!   8. wheels/ contains thyllore_ml_core but no grpcio / protobuf / certifi
//!      wheels (Tier A wheel families dropped).
//!   9. curve_copilot.py is present (the only Tier B operator currently shipping).
//!
//! Skip semantics mirror ``blender_addon_linux_validate.rs``:
//! - No ``thyllore_ml_core`` wheel collected -> early skip with a hint to
//!   run ``scripts/collect_wheels.ps1 -Variant lite``.
//! - PowerShell missing -> early skip (CI ubuntu/macos provide ``pwsh``).
//!
//! Run with::
//!
//!     cargo test -p thyllore-grpc-client --test mvp_addon_zip_smoke -- --nocapture

use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn host_platform_tag() -> &'static str {
    if cfg!(target_os = "windows") {
        "win_amd64"
    } else if cfg!(target_os = "linux") {
        "linux_x86_64"
    } else if cfg!(target_os = "macos") {
        "macosx_arm64"
    } else {
        "unsupported"
    }
}

fn powershell_executable() -> Option<&'static str> {
    let candidates = if cfg!(target_os = "windows") {
        ["powershell.exe", "pwsh.exe", "pwsh"].as_slice()
    } else {
        ["pwsh"].as_slice()
    };
    for cand in candidates {
        if Command::new(cand).arg("-Help").output().is_ok() {
            return Some(cand);
        }
    }
    None
}

fn ml_core_wheel_present() -> bool {
    let dir = workspace_root().join("blender_addon/wheels");
    fs::read_dir(&dir)
        .map(|it| {
            it.filter_map(Result::ok).any(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("thyllore_ml_core-")
            })
        })
        .unwrap_or(false)
}

fn skip(reason: &str) {
    eprintln!("Skipping mvp_addon_zip_smoke: {reason}");
}

#[test]
fn mvp_addon_zip_lite_variant_is_structurally_valid() {
    let platform = host_platform_tag();
    if platform == "unsupported" {
        skip("target_os not in {windows, linux, macos}");
        return;
    }

    let pwsh = match powershell_executable() {
        Some(p) => p,
        None => {
            skip("PowerShell not found on PATH");
            return;
        }
    };

    if !ml_core_wheel_present() {
        skip(
            "no thyllore_ml_core wheel in blender_addon/wheels/ -- \
             run `scripts/collect_wheels.ps1 -Variant lite` first",
        );
        return;
    }

    let root = workspace_root();
    let build_script = root.join("scripts/build_blender_addon.ps1");
    if !build_script.exists() {
        skip("build_blender_addon.ps1 not found");
        return;
    }

    let mut args: Vec<String> = vec![
        "-NoProfile".into(),
        "-ExecutionPolicy".into(),
        "Bypass".into(),
        "-File".into(),
        build_script.to_string_lossy().to_string(),
        "-Platform".into(),
        platform.into(),
        "-Variant".into(),
        "lite".into(),
        "-SkipBlenderValidate".into(),
    ];
    if cfg!(unix) {
        args.insert(0, "-Command".into());
        args.insert(1, "&".into());
    }

    let build_output = Command::new(pwsh)
        .args(if cfg!(target_os = "windows") {
            &args[..]
        } else {
            &args[..]
        })
        .current_dir(&root)
        .output()
        .expect("failed to invoke build_blender_addon.ps1");

    let stdout = String::from_utf8_lossy(&build_output.stdout);
    let stderr = String::from_utf8_lossy(&build_output.stderr);
    if !build_output.status.success() {
        if stderr.contains("Expected at least") {
            skip(&format!(
                "build_blender_addon.ps1 -Variant lite failed -- wheels missing.\n{stderr}"
            ));
            return;
        }
        panic!(
            "build_blender_addon.ps1 -Variant lite failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    let sentinel = root.join("dist/.last_built_zip");
    let zip_path_str = fs::read_to_string(&sentinel).unwrap_or_else(|e| {
        panic!(
            "build_blender_addon.ps1 reported success but sentinel {} is unreadable: {e}",
            sentinel.display()
        )
    });
    let zip_path = PathBuf::from(zip_path_str.trim_start_matches('\u{FEFF}').trim());
    assert!(
        zip_path.exists(),
        "sentinel {} points to missing file: {}",
        sentinel.display(),
        zip_path.display()
    );

    let zip_name = zip_path
        .file_name()
        .expect("zip path has no filename")
        .to_string_lossy();
    let expected_prefix = format!("thyllore_animation_lite-0.0.1-{platform}");
    assert!(
        zip_name.starts_with(&expected_prefix) && zip_name.ends_with(".zip"),
        "lite ZIP filename {zip_name} does not match expected prefix {expected_prefix}*.zip"
    );

    let metadata = fs::metadata(&zip_path).expect("stat ZIP");
    let size_mib = metadata.len() as f64 / (1024.0 * 1024.0);
    assert!(
        metadata.len() < 50 * 1024 * 1024,
        "lite ZIP exceeds 50 MiB ceiling: {size_mib:.2} MiB at {}",
        zip_path.display()
    );
    eprintln!("lite ZIP size: {size_mib:.2} MiB");

    let inspection = inspect_zip(&zip_path).expect("read ZIP entries");
    assert_inspection(&inspection);

    run_blender_extension_validate(&zip_path);
}

enum BlenderRunner {
    Native(PathBuf),
    Wsl { blender_path_wsl: String },
}

fn run_blender_extension_validate(zip_path: &Path) {
    let runner = match locate_blender_42_runner() {
        Some(r) => r,
        None => {
            eprintln!(
                "Skipping `blender --command extension validate`: \
                 set THYLLORE_BLENDER_PATH to a Blender 4.2 binary, \
                 install Blender 4.2 at the standard path, \
                 or install into WSL2 at ~/blender_test/blender/blender \
                 (override with THYLLORE_WSL_BLENDER_PATH). \
                 CI runs this check on every platform."
            );
            return;
        }
    };

    let mut command = match &runner {
        BlenderRunner::Native(path) => {
            eprintln!("Running `extension validate` via native {}", path.display());
            let mut cmd = Command::new(path);
            cmd.args([
                "--command",
                "extension",
                "validate",
                zip_path.to_str().expect("zip path utf-8"),
            ]);
            cmd
        }
        BlenderRunner::Wsl { blender_path_wsl } => {
            let zip_wsl = win_path_to_wsl(zip_path);
            eprintln!("Running `extension validate` via WSL2 {blender_path_wsl} on {zip_wsl}");
            let mut cmd = Command::new("wsl.exe");
            cmd.args([
                blender_path_wsl,
                "--command",
                "extension",
                "validate",
                &zip_wsl,
            ]);
            cmd
        }
    };

    let output = command
        .output()
        .expect("spawn blender for extension validate");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "blender --command extension validate failed for {}\n\
         exit={:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        zip_path.display(),
        output.status.code()
    );
    assert!(
        !stdout.contains("FATAL_ERROR") && !stderr.contains("FATAL_ERROR"),
        "blender extension validate emitted FATAL_ERROR despite exit 0\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
}

fn locate_blender_42_runner() -> Option<BlenderRunner> {
    if let Ok(p) = env::var("THYLLORE_BLENDER_PATH") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(BlenderRunner::Native(path));
        }
    }

    let native_candidates: Vec<PathBuf> = if cfg!(target_os = "windows") {
        vec![PathBuf::from(
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        )]
    } else if cfg!(target_os = "macos") {
        vec![PathBuf::from(
            "/Applications/Blender.app/Contents/MacOS/Blender",
        )]
    } else {
        let mut v = vec![PathBuf::from("/opt/blender/blender")];
        if let Ok(home) = env::var("HOME") {
            v.push(PathBuf::from(home).join("blender_test/blender/blender"));
        }
        v
    };

    if let Some(p) = native_candidates.into_iter().find(|p| p.is_file()) {
        return Some(BlenderRunner::Native(p));
    }

    if cfg!(target_os = "windows") && wsl_available() {
        let blender_wsl = env::var("THYLLORE_WSL_BLENDER_PATH")
            .unwrap_or_else(|_| "~/blender_test/blender/blender".into());
        if wsl_blender_present(&blender_wsl) {
            return Some(BlenderRunner::Wsl {
                blender_path_wsl: blender_wsl,
            });
        }
    }

    None
}

fn wsl_available() -> bool {
    Command::new("wsl.exe")
        .args(["--status"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn wsl_blender_present(blender_path_wsl: &str) -> bool {
    Command::new("wsl.exe")
        .args(["bash", "-lc", &format!("test -x {blender_path_wsl}")])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn win_path_to_wsl(p: &Path) -> String {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        return win_path_to_wsl(Path::new(rest));
    }
    let bytes = s.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' {
        let drive = (bytes[0] as char).to_ascii_lowercase();
        let tail = &s[2..].replace('\\', "/");
        return format!("/mnt/{drive}{tail}");
    }
    s.replace('\\', "/")
}

#[derive(Debug, Default)]
struct ZipInspection {
    entry_paths: Vec<String>,
    manifest_text: Option<String>,
    has_license_md: bool,
    has_eula_md: bool,
    has_curve_copilot: bool,
    wheel_basenames: Vec<String>,
}

fn inspect_zip(zip_path: &Path) -> std::io::Result<ZipInspection> {
    let file = fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut inspection = ZipInspection::default();

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let name = entry.name().to_string();
        inspection.entry_paths.push(name.clone());

        if name == "blender_manifest.toml" {
            let mut buf = String::new();
            entry.read_to_string(&mut buf)?;
            inspection.manifest_text = Some(buf);
        } else if name == "LICENSE.md" {
            inspection.has_license_md = true;
        } else if name == "EULA.md" {
            inspection.has_eula_md = true;
        } else if name == "operators/curve_copilot.py" {
            inspection.has_curve_copilot = true;
        } else if let Some(rest) = name.strip_prefix("wheels/") {
            if rest.ends_with(".whl") && !rest.contains('/') {
                inspection.wheel_basenames.push(rest.to_string());
            }
        }
    }

    Ok(inspection)
}

fn assert_inspection(insp: &ZipInspection) {
    let manifest = insp
        .manifest_text
        .as_deref()
        .expect("blender_manifest.toml absent from ZIP");
    assert!(
        manifest.contains(r#"id = "thyllore_animation_lite""#),
        "manifest does not declare lite id:\n{manifest}"
    );
    assert!(
        manifest.contains("SPDX:GPL-2.0-or-later"),
        "manifest does not declare GPL-2.0-or-later:\n{manifest}"
    );

    assert!(insp.has_license_md, "LICENSE.md missing from lite ZIP");
    assert!(insp.has_eula_md, "EULA.md missing from lite ZIP");
    assert!(
        insp.has_curve_copilot,
        "operators/curve_copilot.py missing from lite ZIP"
    );

    let forbidden_files = [
        "operators/auto_rig.py",
        "operators/text_to_mesh.py",
        "operators/_grpc_helpers.py",
        "operators/text_to_motion.py",
    ];
    for f in &forbidden_files {
        assert!(
            !insp.entry_paths.iter().any(|p| p == f),
            "lite ZIP must not contain {f} (Step A excludes Tier A + gRPC text_to_motion)"
        );
    }

    let forbidden_prefixes = ["grpc_client/", "license/"];
    for prefix in &forbidden_prefixes {
        assert!(
            !insp.entry_paths.iter().any(|p| p.starts_with(prefix)),
            "lite ZIP must not contain anything under {prefix}"
        );
    }

    assert!(
        insp.wheel_basenames
            .iter()
            .any(|w| w.starts_with("thyllore_ml_core-")),
        "lite ZIP must contain a thyllore_ml_core wheel; wheels = {:?}",
        insp.wheel_basenames
    );
    let banned_wheel_prefixes = ["grpcio-", "grpcio_status-", "protobuf-", "certifi-"];
    for w in &insp.wheel_basenames {
        for banned in &banned_wheel_prefixes {
            assert!(
                !w.starts_with(banned),
                "lite ZIP must not bundle Tier A wheel {w}"
            );
        }
    }
}
