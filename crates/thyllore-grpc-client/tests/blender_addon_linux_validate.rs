//! Linux extension ZIP validation via WSL2 Blender 4.2 LTS.
//!
//! Mirrors `blender_grpc_smoke_tests.rs` for graceful skipping when WSL2 or
//! Blender are not available. Marked `#[ignore]` so it does not run by default
//! (it builds a Linux ZIP and shells out to WSL Blender, which takes ~10s and
//! requires a specific dev environment).
//!
//! Run with:
//!     cargo test -p thyllore-grpc-client --test blender_addon_linux_validate \
//!         -- --ignored --nocapture
//!
//! Skip conditions (each prints a reason and returns OK):
//! - Not running on Windows.
//! - `wsl.exe` not on PATH.
//! - WSL2 Blender not at `$HOME/blender_test/blender/blender` (override with
//!   the `THYLLORE_WSL_BLENDER_PATH` env var).
//! - Linux wheels not collected in `blender_addon/wheels/` (build script
//!   reports "Expected at least 4 wheels for linux_x86_64"). Run
//!   `wsl bash -lc 'cd <repo> && python -m pip download --platform
//!   manylinux2014_x86_64 ...'` plus `maturin build` to populate.

use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root (two levels above grpc-client crate)")
        .to_path_buf()
}

#[test]
#[ignore = "requires Windows host with WSL2 Ubuntu + Blender 4.2 LTS; run with --ignored"]
fn blender_addon_linux_zip_validates_via_wsl() {
    if !cfg!(target_os = "windows") {
        eprintln!("Skipping: WSL-based Linux validation only runs on Windows hosts");
        return;
    }

    let blender_path = std::env::var("THYLLORE_WSL_BLENDER_PATH")
        .unwrap_or_else(|_| "$HOME/blender_test/blender/blender".to_string());

    if !is_wsl_available() {
        eprintln!("Skipping: wsl.exe is not available on PATH");
        return;
    }
    if !is_wsl_blender_present(&blender_path) {
        eprintln!(
            "Skipping: WSL Blender not found at {}. Install Blender 4.2 LTS \
             into WSL Ubuntu or set THYLLORE_WSL_BLENDER_PATH.",
            blender_path
        );
        return;
    }

    let root = workspace_root();
    let build_script = root.join("scripts/build_blender_addon.ps1");
    if !build_script.exists() {
        eprintln!(
            "Skipping: build script missing at {}",
            build_script.display()
        );
        return;
    }

    let build_output = Command::new("powershell.exe")
        .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"])
        .arg(&build_script)
        .args(["-Platform", "linux_x86_64", "-SkipBlenderValidate"])
        .current_dir(&root)
        .output()
        .expect("failed to invoke build_blender_addon.ps1");

    let build_stdout = String::from_utf8_lossy(&build_output.stdout);
    let build_stderr = String::from_utf8_lossy(&build_output.stderr);

    if !build_output.status.success() {
        if build_stderr.contains("Expected at least") {
            eprintln!(
                "Skipping: Linux wheels not present in blender_addon/wheels/. \
                 Populate via WSL2:\n\
                 \n\
                 wsl -d Ubuntu -- bash -lc '\\\n\
                     cd /mnt/c/.../ThylloreAnimation && \\\n\
                     python3 -m pip download grpcio==1.71.2 grpcio-status==1.71.2 \\\n\
                                          protobuf==5.29.6 certifi==2024.12.14 \\\n\
                         --platform manylinux2014_x86_64 --abi cp310 --implementation cp \\\n\
                         --only-binary=:all: --dest blender_addon/wheels --no-deps && \\\n\
                     cd crates/thyllore-ml-core && \\\n\
                     maturin build --release --features python --out ../../blender_addon/wheels'"
            );
            return;
        }
        panic!(
            "build_blender_addon.ps1 -Platform linux_x86_64 failed:\n\
             stdout:\n{}\n\
             stderr:\n{}",
            build_stdout, build_stderr
        );
    }

    let zip_path_win = root.join("dist/thyllore_animation_addon-0.0.1-linux_x86_64.zip");
    if !zip_path_win.exists() {
        panic!(
            "build script reported success but ZIP missing at {}",
            zip_path_win.display()
        );
    }

    let zip_path_wsl = win_path_to_wsl(&zip_path_win);

    let validate_output = Command::new("wsl.exe")
        .args(["-d", "Ubuntu", "--", "bash", "-lc"])
        .arg(format!(
            "{} --command extension validate {}",
            shell_escape(&blender_path),
            shell_escape(&zip_path_wsl)
        ))
        .output()
        .expect("failed to invoke wsl Blender");

    let validate_stdout = String::from_utf8_lossy(&validate_output.stdout);
    let validate_stderr = String::from_utf8_lossy(&validate_output.stderr);
    eprintln!(
        "WSL Blender stdout:\n{}\nstderr:\n{}",
        validate_stdout, validate_stderr
    );

    let combined = format!("{}\n{}", validate_stdout, validate_stderr);
    assert!(
        combined.contains("Success parsing TOML"),
        "Blender extension validate did not report success.\nOutput:\n{}",
        combined
    );
}

fn is_wsl_available() -> bool {
    Command::new("wsl.exe")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn is_wsl_blender_present(blender_path: &str) -> bool {
    let probe = format!("test -x {}", shell_escape(blender_path));
    Command::new("wsl.exe")
        .args(["-d", "Ubuntu", "--", "bash", "-lc"])
        .arg(probe)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Translate `C:\path\to\file` into `/mnt/c/path/to/file` for WSL consumption.
fn win_path_to_wsl(p: &Path) -> String {
    let s = p.to_string_lossy().replace('\\', "/");
    if let Some(rest) = s.strip_prefix(|c: char| c.is_ascii_alphabetic()) {
        if let Some(rest) = rest.strip_prefix(':') {
            let drive_letter = s.chars().next().expect("drive letter").to_ascii_lowercase();
            return format!("/mnt/{}{}", drive_letter, rest);
        }
    }
    s
}

/// Single-quote the string for safe inclusion in a `bash -c` command.
fn shell_escape(s: &str) -> String {
    if s.contains('\'') {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        format!("'{}'", s)
    }
}

#[cfg(test)]
mod helper_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn win_path_to_wsl_translates_drive_letter() {
        let p = PathBuf::from(r"C:\Users\kodai\Projects\ThylloreAnimation\dist\foo.zip");
        assert_eq!(
            win_path_to_wsl(&p),
            "/mnt/c/Users/kodai/Projects/ThylloreAnimation/dist/foo.zip"
        );
    }

    #[test]
    fn win_path_to_wsl_preserves_already_posix_path() {
        let p = PathBuf::from("/home/user/file");
        assert_eq!(win_path_to_wsl(&p), "/home/user/file");
    }

    #[test]
    fn shell_escape_wraps_simple_strings_in_single_quotes() {
        assert_eq!(shell_escape("foo bar"), "'foo bar'");
    }
}
