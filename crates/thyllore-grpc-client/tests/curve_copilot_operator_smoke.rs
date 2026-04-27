//! curve_copilot Operator headless smoke.
//!
//! Drives ``bpy.ops.thyllore.curve_copilot`` synchronously from a real Blender
//! 4.2 LTS process with the addon enabled. The bit-identical parity between
//! ``MlCoreImpl::run_curve_copilot`` and the wheel's typed/call_op_json paths
//! is already covered by ``thyllore-ml-core``'s ``curve_copilot_blender_parity``
//! test; this smoke is responsible for proving that the Modal/Operator wiring
//! keeps working under ``--background`` (i.e. the headless flag actually
//! shortcuts the worker thread + Timer pipeline).
//!
//! Skip semantics mirror ``blender_addon_linux_validate.rs``: each missing
//! prerequisite is reported and the test returns OK.
//!
//! Run with::
//!
//!     cargo test -p thyllore-grpc-client --test curve_copilot_operator_smoke \
//!         --features text-to-motion -- --ignored --nocapture

#![cfg(feature = "text-to-motion")]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SMOKE_RESULT_NAME: &str = "curve_copilot_operator_smoke.json";
const SMOKE_SCRIPT_RELATIVE: &str = "blender_addon/tests/curve_copilot_operator_smoke.py";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn read_paths_md_value(key: &str) -> Option<String> {
    let path = workspace_root().join(".claude/local/paths.md");
    let content = fs::read_to_string(path).ok()?;
    let prefix = format!("- {key} = ");
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix(prefix.as_str()) {
            let value = rest.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn resolve_blender_executable() -> Option<PathBuf> {
    if let Ok(p) = env::var("THYLLORE_BLENDER_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    if cfg!(unix) {
        if let Ok(home) = env::var("HOME") {
            let wsl = PathBuf::from(home).join("blender_test/blender/blender");
            if wsl.exists() {
                return Some(wsl);
            }
        }
    }
    if let Some(value) = read_paths_md_value("BlenderPath") {
        let path = PathBuf::from(value);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn resolve_fixture_root() -> PathBuf {
    if let Ok(p) = env::var("THYLLORE_PARITY_FIXTURE_OUTPUT") {
        return PathBuf::from(p);
    }
    workspace_root().join("fixtures").join("ml_parity")
}

fn resolve_ort_dylib_path() -> Option<PathBuf> {
    let vendor_candidate = if cfg!(windows) {
        workspace_root()
            .join("vendor/onnxruntime/onnxruntime-win-x64-1.23.2/lib/onnxruntime.dll")
    } else {
        workspace_root()
            .join("vendor/onnxruntime/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so")
    };
    if vendor_candidate.exists() {
        return Some(vendor_candidate);
    }
    if let Ok(p) = env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn assert_smoke_result_ok(result_path: &Path, stdout: &str, stderr: &str) {
    let text = fs::read_to_string(result_path).unwrap_or_else(|e| {
        panic!(
            "smoke result missing at {}: {e}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            result_path.display()
        )
    });

    let parsed: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("smoke result not JSON: {e}\n{text}"));

    let ok = parsed
        .get("ok")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(
        ok,
        "curve_copilot operator smoke reported ok=false:\n{text}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );

    let inserted = parsed
        .get("inserted_count")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1);
    assert!(
        inserted >= 0,
        "curve_copilot operator smoke produced negative inserted_count: {inserted}\n{text}"
    );
}

#[test]
#[ignore]
fn curve_copilot_operator_runs_under_background() {
    let Some(blender) = resolve_blender_executable() else {
        eprintln!(
            "skip: Blender executable not found (set THYLLORE_BLENDER_PATH or check paths.md)"
        );
        return;
    };

    let fixture_root = resolve_fixture_root();
    let onnx = fixture_root.join("onnx/curve_copilot.onnx");
    if !onnx.exists() {
        eprintln!(
            "skip: curve_copilot.onnx missing at {}; run scripts/generate_parity_fixtures.sh",
            onnx.display()
        );
        return;
    }

    let script = workspace_root().join(SMOKE_SCRIPT_RELATIVE);
    if !script.exists() {
        panic!(
            "curve_copilot_operator_smoke.py missing at {}",
            script.display()
        );
    }

    let temp = tempfile::tempdir().expect("create temp dir");
    let result_path = temp.path().join(SMOKE_RESULT_NAME);

    let mut command = Command::new(&blender);
    command
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script)
        .arg("--")
        .arg("--result")
        .arg(&result_path)
        .arg("--onnx")
        .arg(&onnx)
        .env("THYLLORE_HEADLESS", "1")
        .env("THYLLORE_FORCE_MOCK_SERVER", "1")
        .env("THYLLORE_TEST_BYPASS_LICENSE", "1");

    if let Some(dylib_path) = resolve_ort_dylib_path() {
        command.env("ORT_DYLIB_PATH", &dylib_path);
    }

    let output = command
        .output()
        .expect("spawn Blender for Tier B smoke");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        panic!(
            "Blender Tier B smoke failed: status={:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            output.status.code()
        );
    }

    assert_smoke_result_ok(&result_path, &stdout, &stderr);
}
