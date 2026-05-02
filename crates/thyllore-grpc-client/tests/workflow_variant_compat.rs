//! Static check: every `.github/workflows/*.yml` job that installs a Variant
//! of the Blender addon must invoke only tests whose declared Variant
//! requirement is satisfied by that install.
//!
//! Bug class caught: PR #100 had `blender_parity.yml` install lite (the
//! default) yet invoke `grpc_request_wire_parity`, which imports
//! `grpc_client.stubs` -- a directory the lite Variant intentionally
//! excludes. CI surfaced this as `Add-on not loaded`. The static check
//! below catches the same mismatch in milliseconds, with no Blender / wheel /
//! ONNX setup required.
//!
//! Detection model:
//!   - Walk every `.github/workflows/*.yml` line by line.
//!   - When a line contains `build_blender_addon.ps1`, snapshot
//!     `current_variant` from the `-Variant <X>` argument (defaulting to
//!     `lite` -- mirrors the build script default).
//!   - When a line contains `cargo test ... --test <name>`, look up `<name>`
//!     in the inline registry below and verify
//!     `current_variant.satisfies(required)`.
//!   - Tests not in the registry are assumed not to load the addon and are
//!     skipped silently.
//!
//! Adding new tests:
//!   - If your test installs / loads the addon, add a row to
//!     `TEST_REQUIREMENTS` with the required Variant and a one-line reason.
//!   - Tests that only exercise wheels / fixtures / proto without going
//!     through Blender's addon loader do NOT need a row.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variant {
    Lite,
    Full,
    Any,
}

impl Variant {
    fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "lite" => Some(Variant::Lite),
            "full" => Some(Variant::Full),
            "any" => Some(Variant::Any),
            _ => None,
        }
    }

    fn satisfies(self, required: Variant) -> bool {
        match (self, required) {
            (_, Variant::Any) => true,
            (Variant::Full, _) => true,
            (Variant::Lite, Variant::Lite) => true,
            (Variant::Lite, Variant::Full) => false,
            (Variant::Any, _) => false,
        }
    }
}

const TEST_REQUIREMENTS: &[(&str, Variant, &str)] = &[
    (
        "grpc_request_wire_parity",
        Variant::Full,
        "Imports grpc_client.stubs (Tier A); lite excludes blender_addon/grpc_client/.",
    ),
    (
        "curve_copilot_operator_smoke",
        Variant::Any,
        "Tier B Operator via PyO3; works under any Variant once the addon id is registered.",
    ),
    (
        "curve_copilot_blender_parity",
        Variant::Any,
        "Loads thyllore_ml_core wheel directly; never enables the addon.",
    ),
    (
        "mvp_addon_zip_smoke",
        Variant::Lite,
        "Explicitly builds and validates the lite Variant ZIP.",
    ),
    (
        "blender_addon_linux_validate",
        Variant::Lite,
        "WSL2-driven smoke that mirrors blender_addon_build.yml (lite default).",
    ),
];

#[test]
fn workflows_install_variants_satisfy_invoked_test_requirements() {
    let workflow_dir = workspace_root().join(".github/workflows");
    assert!(
        workflow_dir.is_dir(),
        ".github/workflows missing at {}",
        workflow_dir.display()
    );

    let mut violations: Vec<String> = Vec::new();
    let mut workflow_count = 0;

    for entry in fs::read_dir(&workflow_dir).expect("read workflow dir") {
        let entry = entry.expect("read workflow entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yml") {
            continue;
        }
        workflow_count += 1;
        scan_workflow(&path, &mut violations);
    }

    assert!(workflow_count > 0, "no workflow YAML files found");
    assert!(
        violations.is_empty(),
        "Variant <-> test compatibility violations found ({}):\n  {}\n\
         Either change the workflow's `-Variant` to satisfy the requirement, \
         or update TEST_REQUIREMENTS in this file if the test no longer needs \
         that Variant feature.",
        violations.len(),
        violations.join("\n  ")
    );
}

#[test]
fn registry_test_names_correspond_to_real_test_files() {
    let mut missing: Vec<&str> = Vec::new();
    for (name, _, _) in TEST_REQUIREMENTS {
        if locate_test_file(name).is_none() {
            missing.push(name);
        }
    }
    assert!(
        missing.is_empty(),
        "TEST_REQUIREMENTS contains entries with no corresponding crates/<crate>/tests/<name>.rs file: {missing:?}"
    );
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn locate_test_file(test_name: &str) -> Option<PathBuf> {
    const CRATES: &[&str] = &[
        "thyllore-grpc-client",
        "thyllore-ml-core",
        "thyllore-ml-api",
        "thyllore-anim-core",
        "thyllore-model-core",
    ];
    for c in CRATES {
        let path =
            workspace_root().join(format!("crates/{c}/tests/{test_name}.rs"));
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn scan_workflow(path: &Path, violations: &mut Vec<String>) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => panic!("read {} failed: {e}", path.display()),
    };

    let mut current_variant: Option<Variant> = None;
    for (idx, line) in source.lines().enumerate() {
        if is_build_invocation(line) {
            let extracted = extract_variant_arg(line);
            current_variant = Some(extracted.unwrap_or(Variant::Lite));
            let _ = idx;
        }

        if let Some(test_name) = extract_cargo_test_name(line) {
            let Some((_, required, reason)) = TEST_REQUIREMENTS
                .iter()
                .find(|(n, _, _)| *n == test_name.as_str())
            else {
                continue;
            };

            let installed = current_variant.unwrap_or(Variant::Lite);
            if !installed.satisfies(*required) {
                violations.push(format!(
                    "{}:{}: installs Variant `{:?}` but `--test {}` requires `{:?}` ({})",
                    path.display(),
                    idx + 1,
                    installed,
                    test_name,
                    required,
                    reason
                ));
            }
        }
    }
}

fn is_build_invocation(line: &str) -> bool {
    // An actual `build_blender_addon.ps1` call always passes `-Platform`.
    // Path-filter listings (`- 'scripts/build_blender_addon.ps1'`) and
    // diagnostic echoes (`echo "::error::build_blender_addon.ps1 ..."`)
    // mention the script name without invoking it -- they would otherwise
    // reset state to default Lite and mask later -Variant settings.
    line.contains("build_blender_addon.ps1") && line.contains("-Platform")
}

fn extract_variant_arg(line: &str) -> Option<Variant> {
    let idx = line.find("-Variant")?;
    let rest = line[idx + "-Variant".len()..].trim_start();
    let token = rest
        .split(|c: char| c.is_whitespace() || c == '`' || c == '\\')
        .find(|s| !s.is_empty())?;
    Variant::parse(token)
}

fn extract_cargo_test_name(line: &str) -> Option<String> {
    if !line.contains("cargo test") && !line.contains("--test") {
        return None;
    }
    let idx = line.find("--test")?;
    let after = line[idx + "--test".len()..].trim_start();
    let token = after
        .split(|c: char| c.is_whitespace() || c == '`' || c == '\\')
        .find(|s| !s.is_empty())?;
    let cleaned = token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.to_string())
    }
}

#[cfg(test)]
mod parser_self_tests {
    use super::*;

    #[test]
    fn satisfies_truth_table() {
        assert!(Variant::Full.satisfies(Variant::Lite));
        assert!(Variant::Full.satisfies(Variant::Full));
        assert!(Variant::Full.satisfies(Variant::Any));
        assert!(Variant::Lite.satisfies(Variant::Lite));
        assert!(Variant::Lite.satisfies(Variant::Any));
        assert!(!Variant::Lite.satisfies(Variant::Full));
    }

    #[test]
    fn parses_variant_arg_in_pwsh_continuation() {
        let line = "          ./scripts/build_blender_addon.ps1 -Platform x -Variant full -SkipBlenderValidate";
        assert_eq!(extract_variant_arg(line), Some(Variant::Full));
    }

    #[test]
    fn parses_variant_arg_with_lite() {
        let line = "    pwsh ... build_blender_addon.ps1 ... -Variant lite";
        assert_eq!(extract_variant_arg(line), Some(Variant::Lite));
    }

    #[test]
    fn no_variant_arg_returns_none() {
        let line = "          ./scripts/build_blender_addon.ps1 -Platform x";
        assert_eq!(extract_variant_arg(line), None);
    }

    #[test]
    fn extracts_cargo_test_name_with_continuation() {
        let line = "cargo test -p thyllore-grpc-client --test grpc_request_wire_parity \\";
        assert_eq!(
            extract_cargo_test_name(line),
            Some("grpc_request_wire_parity".to_string())
        );
    }

    #[test]
    fn extracts_cargo_test_name_with_function_filter() {
        let line = "cargo test -p thyllore-grpc-client --test grpc_request_wire_parity grpc_request_wire_bytes_match -- --ignored";
        assert_eq!(
            extract_cargo_test_name(line),
            Some("grpc_request_wire_parity".to_string())
        );
    }
}
