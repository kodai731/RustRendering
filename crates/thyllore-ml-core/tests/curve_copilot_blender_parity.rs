use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const CASES: &[&str] = &["short", "medium", "long"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn read_paths_md_value(key: &str) -> Option<String> {
    let paths_md = workspace_root().join(".claude/local/paths.md");
    let content = std::fs::read_to_string(paths_md).ok()?;
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
            let wsl_path = PathBuf::from(home).join("blender_test/blender/blender");
            if wsl_path.exists() {
                return Some(wsl_path);
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

fn resolve_wheel_dir() -> PathBuf {
    workspace_root().join("blender_addon").join("wheels")
}

fn resolve_ort_dylib_path() -> Option<PathBuf> {
    let vendor_candidate = if cfg!(windows) {
        workspace_root()
            .join("vendor")
            .join("onnxruntime")
            .join("onnxruntime-win-x64-1.23.2")
            .join("lib")
            .join("onnxruntime.dll")
    } else {
        workspace_root()
            .join("vendor")
            .join("onnxruntime")
            .join("onnxruntime-linux-x64-1.23.2")
            .join("lib")
            .join("libonnxruntime.so")
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

#[test]
#[ignore]
fn curve_copilot_typed_and_call_op_paths_match_in_blender() {
    let fixture_root = resolve_fixture_root();
    if !fixture_root.exists() {
        eprintln!(
            "skip: fixture root {} not found; run scripts/generate_parity_fixtures.sh",
            fixture_root.display()
        );
        return;
    }

    let Some(blender) = resolve_blender_executable() else {
        eprintln!("skip: Blender executable not found (set THYLLORE_BLENDER_PATH or check paths.md)");
        return;
    };

    let wheel_dir = resolve_wheel_dir();
    if !wheel_dir.exists() {
        eprintln!(
            "skip: wheel dir {} not found; run scripts/build_blender_addon.ps1 first",
            wheel_dir.display()
        );
        return;
    }

    let script_path = workspace_root()
        .join("crates/thyllore-ml-core/tests/curve_copilot_blender_runner.py");
    if !script_path.exists() {
        panic!("curve_copilot_blender_runner.py missing at {}", script_path.display());
    }

    let temp_dir = tempfile::tempdir().expect("create temp dir");

    for case_id in CASES {
        let typed_output = temp_dir.path().join(format!("typed_{case_id}.json"));
        let call_op_output = temp_dir.path().join(format!("call_op_{case_id}.json"));

        run_blender_parity_script(
            &blender,
            &script_path,
            &wheel_dir,
            &fixture_root,
            case_id,
            &typed_output,
            &call_op_output,
        );

        let golden_path = fixture_root
            .join("numpy")
            .join(format!("curve_copilot_golden_{case_id}.json"));

        assert_predictions_bit_identical(&golden_path, &typed_output, case_id, "typed_vs_golden");
        assert_predictions_bit_identical(
            &golden_path,
            &call_op_output,
            case_id,
            "call_op_vs_golden",
        );
        assert_predictions_bit_identical(
            &typed_output,
            &call_op_output,
            case_id,
            "typed_vs_call_op",
        );
    }
}

fn run_blender_parity_script(
    blender: &Path,
    script: &Path,
    wheel_dir: &Path,
    fixture_root: &Path,
    case_id: &str,
    typed_output: &Path,
    call_op_output: &Path,
) {
    let mut command = Command::new(blender);
    command.args([
        "--background",
        "--factory-startup",
        "--python",
        script.to_str().expect("script path utf-8"),
        "--",
        "--wheel-dir",
        wheel_dir.to_str().expect("wheel dir utf-8"),
        "--fixture-root",
        fixture_root.to_str().expect("fixture root utf-8"),
        "--case-id",
        case_id,
        "--typed-output",
        typed_output.to_str().expect("typed output utf-8"),
        "--call-op-output",
        call_op_output.to_str().expect("call_op output utf-8"),
    ]);

    if let Some(dylib_path) = resolve_ort_dylib_path() {
        command.env("ORT_DYLIB_PATH", &dylib_path);
    }

    let output = command.output().expect("spawn blender");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let typed_exists = typed_output.exists();
    let call_op_exists = call_op_output.exists();

    if !output.status.success() || !typed_exists || !call_op_exists {
        panic!(
            "blender failed for case {case_id}:\n  exit={:?}\n  typed_output exists={typed_exists}\n  call_op_output exists={call_op_exists}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            output.status.code()
        );
    }
    eprintln!("blender stdout for {case_id}: {}", stdout.trim());
}

fn assert_predictions_bit_identical(
    left_path: &Path,
    right_path: &Path,
    case_id: &str,
    label: &str,
) {
    let left = parse_predictions_json(left_path);
    let right = parse_predictions_json(right_path);

    assert_eq!(
        left.len(),
        right.len(),
        "[{case_id}/{label}] prediction count differs ({} vs {})",
        left.len(),
        right.len()
    );

    for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(
            l.value_bits, r.value_bits,
            "[{case_id}/{label}] prediction[{i}].value_bits differ: {:#x} vs {:#x}",
            l.value_bits, r.value_bits
        );
        assert_eq!(
            l.tangent_in_bits, r.tangent_in_bits,
            "[{case_id}/{label}] prediction[{i}].tangent_in_bits differ"
        );
        assert_eq!(
            l.tangent_out_bits, r.tangent_out_bits,
            "[{case_id}/{label}] prediction[{i}].tangent_out_bits differ"
        );
        assert_eq!(
            l.confidence_bits, r.confidence_bits,
            "[{case_id}/{label}] prediction[{i}].confidence_bits differ: {:#x} vs {:#x}",
            l.confidence_bits, r.confidence_bits
        );
    }
}

#[derive(Debug)]
struct ParsedPrediction {
    value_bits: u32,
    tangent_in_bits: [u32; 2],
    tangent_out_bits: [u32; 2],
    confidence_bits: u32,
}

fn parse_predictions_json(path: &Path) -> Vec<ParsedPrediction> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {} failed: {e}", path.display()));
    let value: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("parse {} as JSON failed: {e}", path.display()));

    let predictions = value
        .get("predictions")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("{} missing top-level predictions array", path.display()));

    predictions
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let value_bits = expect_u32(item, "value_bits", path, i);
            let confidence_bits = expect_u32(item, "confidence_bits", path, i);
            let tangent_in_bits = expect_u32_pair(item, "tangent_in_bits", path, i);
            let tangent_out_bits = expect_u32_pair(item, "tangent_out_bits", path, i);
            ParsedPrediction {
                value_bits,
                tangent_in_bits,
                tangent_out_bits,
                confidence_bits,
            }
        })
        .collect()
}

fn expect_u32(item: &serde_json::Value, field: &str, path: &Path, idx: usize) -> u32 {
    let value = item
        .get(field)
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("{}[{idx}].{field} missing or not a number", path.display()));
    value as u32
}

fn expect_u32_pair(item: &serde_json::Value, field: &str, path: &Path, idx: usize) -> [u32; 2] {
    let array = item
        .get(field)
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("{}[{idx}].{field} not an array", path.display()));
    if array.len() != 2 {
        panic!(
            "{}[{idx}].{field} must have length 2 (got {})",
            path.display(),
            array.len()
        );
    }
    [
        array[0]
            .as_u64()
            .unwrap_or_else(|| panic!("{}[{idx}].{field}[0] not a number", path.display())) as u32,
        array[1]
            .as_u64()
            .unwrap_or_else(|| panic!("{}[{idx}].{field}[1] not a number", path.display())) as u32,
    ]
}
