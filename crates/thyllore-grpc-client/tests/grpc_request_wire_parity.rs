#![cfg(feature = "text-to-motion")]

mod common;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::common::mock_server::{MockServerConfig, MockServerHandle};

const SERVICE_AUTO_RIG: &str = "AutoRiggingService";
const SERVICE_MOTION: &str = "TextToMotionService";
const SERVICE_MESH: &str = "MeshGenerationService";

const METHOD_GENERATE_RIG: &str = "GenerateRig";
const METHOD_GENERATE_MOTION: &str = "GenerateMotion";
const METHOD_GENERATE_MESH: &str = "GenerateMesh";

const RUST_RIGGING_RESULT: &str = "rigging_response_rust.json";
const RUST_MOTION_RESULT: &str = "motion_response_rust.json";
const RUST_MESH_RESULT: &str = "mesh_response_rust.json";

const BLENDER_RIGGING_RESULT: &str = "rigging_response_blender.json";
const BLENDER_MOTION_RESULT: &str = "motion_response_blender.json";
const BLENDER_MESH_RESULT: &str = "mesh_response_blender.json";

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

fn resolve_fixture_root() -> PathBuf {
    if let Ok(p) = env::var("THYLLORE_PARITY_FIXTURE_OUTPUT") {
        return PathBuf::from(p);
    }
    workspace_root().join("fixtures").join("ml_parity")
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
            let wsl_blender = PathBuf::from(home).join("blender_test/blender/blender");
            if wsl_blender.exists() {
                return Some(wsl_blender);
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

#[test]
#[ignore]
fn grpc_request_wire_bytes_match_across_rust_and_blender() {
    let fixture_root = resolve_fixture_root();
    if !fixture_root.exists() {
        eprintln!(
            "skip: fixture root {} not found; run scripts/generate_parity_fixtures.sh",
            fixture_root.display()
        );
        return;
    }

    for required in [
        "proto/rigging_request.bin",
        "proto/rigging_response.bin",
        "proto/motion_request.bin",
        "proto/motion_response.bin",
        "proto/mesh_request.bin",
        "proto/mesh_response.bin",
    ] {
        if !fixture_root.join(required).exists() {
            eprintln!(
                "skip: required fixture missing: {}",
                fixture_root.join(required).display()
            );
            return;
        }
    }

    let Some(blender) = resolve_blender_executable() else {
        eprintln!(
            "skip: Blender executable not found (set THYLLORE_BLENDER_PATH or check paths.md)"
        );
        return;
    };

    let blender_script = workspace_root().join("blender_addon/tests/grpc_parity_blender_client.py");
    if !blender_script.exists() {
        panic!(
            "grpc_parity_blender_client.py missing at {}",
            blender_script.display()
        );
    }

    let server = MockServerHandle::start(MockServerConfig::DeterministicFromFixtures(
        fixture_root.clone(),
    ));
    let server_url = server.address.clone();

    let temp = tempfile::tempdir().expect("create temp dir");
    let result_dir = temp.path().to_path_buf();

    run_rust_client(&server_url, &fixture_root, &result_dir);
    let rust_request_sha = collect_request_sha(&server, "rust");

    run_blender_client(
        &blender,
        &blender_script,
        &server_url,
        &fixture_root,
        &result_dir,
    );
    let blender_request_sha = collect_request_sha(&server, "blender");

    drop(server);

    assert_request_sha_pairs_match(&rust_request_sha, &blender_request_sha);
    assert_canonical_responses_match(&result_dir);
}

#[derive(Debug)]
struct RecordedRequestShas {
    auto_rig: String,
    motion: String,
    mesh: String,
}

fn collect_request_sha(server: &MockServerHandle, label: &str) -> RecordedRequestShas {
    let buffer = server.recv_buffer.lock().expect("recv_buffer poisoned");
    let auto_rig_index = buffer.count(SERVICE_AUTO_RIG, METHOD_GENERATE_RIG);
    let motion_index = buffer.count(SERVICE_MOTION, METHOD_GENERATE_MOTION);
    let mesh_index = buffer.count(SERVICE_MESH, METHOD_GENERATE_MESH);

    assert!(
        auto_rig_index > 0,
        "[{label}] no AutoRig request was recorded"
    );
    assert!(motion_index > 0, "[{label}] no Motion request was recorded");
    assert!(mesh_index > 0, "[{label}] no Mesh request was recorded");

    RecordedRequestShas {
        auto_rig: buffer
            .sha256_of(SERVICE_AUTO_RIG, METHOD_GENERATE_RIG, auto_rig_index - 1)
            .expect("sha256 auto_rig"),
        motion: buffer
            .sha256_of(SERVICE_MOTION, METHOD_GENERATE_MOTION, motion_index - 1)
            .expect("sha256 motion"),
        mesh: buffer
            .sha256_of(SERVICE_MESH, METHOD_GENERATE_MESH, mesh_index - 1)
            .expect("sha256 mesh"),
    }
}

fn run_rust_client(server_url: &str, fixture_root: &Path, result_dir: &Path) {
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());

    let output = Command::new(&cargo)
        .current_dir(workspace_root())
        .args([
            "test",
            "-p",
            "thyllore-grpc-client",
            "--features",
            "auto-rig,text-to-motion",
            "--test",
            "grpc_parity_rust_client",
            "run_for_orchestrator",
            "--",
            "--ignored",
            "--nocapture",
            "--exact",
        ])
        .env("THYLLORE_PARITY_SERVER_URL", server_url)
        .env("THYLLORE_PARITY_FIXTURE_ROOT", fixture_root)
        .env("THYLLORE_PARITY_RESULT_DIR", result_dir)
        .output()
        .expect("spawn rust client cargo test");

    if !output.status.success() {
        panic!(
            "rust client failed: status={:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    for required in [RUST_RIGGING_RESULT, RUST_MOTION_RESULT, RUST_MESH_RESULT] {
        let path = result_dir.join(required);
        assert!(
            path.exists(),
            "rust client did not produce {}",
            path.display()
        );
    }
}

fn run_blender_client(
    blender: &Path,
    script: &Path,
    server_url: &str,
    fixture_root: &Path,
    result_dir: &Path,
) {
    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(script)
        .args(["--", "--server-url", server_url, "--fixture-root"])
        .arg(fixture_root)
        .arg("--result-dir")
        .arg(result_dir)
        .env("THYLLORE_HEADLESS", "1")
        .env("THYLLORE_FORCE_MOCK_SERVER", "1")
        // No license module: mode C is fully offline, licensing was retired for
        // good (design doc 20260719_curve_copilot_private_mode_offline).
        .output()
        .expect("spawn blender");

    if !output.status.success() {
        panic!(
            "blender client failed: status={:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    for required in [
        BLENDER_RIGGING_RESULT,
        BLENDER_MOTION_RESULT,
        BLENDER_MESH_RESULT,
    ] {
        let path = result_dir.join(required);
        assert!(
            path.exists(),
            "blender client did not produce {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

fn assert_request_sha_pairs_match(rust: &RecordedRequestShas, blender: &RecordedRequestShas) {
    assert_eq!(
        rust.auto_rig, blender.auto_rig,
        "AutoRig request wire bytes mismatch:\n  rust:    {}\n  blender: {}",
        rust.auto_rig, blender.auto_rig,
    );
    assert_eq!(
        rust.motion, blender.motion,
        "Motion request wire bytes mismatch:\n  rust:    {}\n  blender: {}",
        rust.motion, blender.motion,
    );
    assert_eq!(
        rust.mesh, blender.mesh,
        "Mesh request wire bytes mismatch:\n  rust:    {}\n  blender: {}",
        rust.mesh, blender.mesh,
    );
}

fn assert_canonical_responses_match(result_dir: &Path) {
    for (label, rust_name, blender_name) in [
        ("AutoRig", RUST_RIGGING_RESULT, BLENDER_RIGGING_RESULT),
        ("Motion", RUST_MOTION_RESULT, BLENDER_MOTION_RESULT),
        ("Mesh", RUST_MESH_RESULT, BLENDER_MESH_RESULT),
    ] {
        let rust_text = fs::read_to_string(result_dir.join(rust_name))
            .unwrap_or_else(|e| panic!("read {rust_name}: {e}"));
        let blender_text = fs::read_to_string(result_dir.join(blender_name))
            .unwrap_or_else(|e| panic!("read {blender_name}: {e}"));
        assert_eq!(
            rust_text.trim(),
            blender_text.trim(),
            "{label} canonical response mismatch (sort_keys=true should make text-equal)\n--- {rust_name}\n{rust_text}\n--- {blender_name}\n{blender_text}"
        );
    }
}
