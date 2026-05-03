//! Static SSOT check for `ABI_MARKER` across the L2/L3/L4 layers.
//!
//! Why this lives at the workspace level instead of inside one of the
//! consumers: the bug class this guards against is "value bumped in one
//! file, forgotten in the others." Running here makes the test crate-
//! agnostic and forces a 3-file diff whenever the marker bumps.
//!
//! Files inspected:
//!   - `crates/thyllore-ml-api/src/lib.rs`                           (L2 SSOT)
//!   - `blender_addon/__init__.py`                                   (L4 addon)
//!   - `crates/thyllore-ml-core/tests/curve_copilot_blender_runner.py` (test
//!     runner -- must read from env, NOT hardcode the marker)
//!
//! Run::
//!
//!     cargo test -p thyllore-grpc-client --test abi_marker_consistency

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("read {} failed: {e}", path.display()))
}

fn extract_rust_const(source: &str) -> Option<u32> {
    for line in source.lines() {
        let trimmed = line.trim();
        let prefix = "pub const ABI_MARKER: u32 = ";
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let value = rest.trim_end_matches(';').trim();
            return value.parse().ok();
        }
    }
    None
}

fn extract_python_int(source: &str, identifier: &str) -> Option<u32> {
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(identifier) {
            let value = rest.trim().split_whitespace().next()?;
            return value.parse().ok();
        }
    }
    None
}

#[test]
fn rust_abi_marker_matches_addon_expected_value() {
    let root = workspace_root();

    let rust_source = read(&root.join("crates/thyllore-ml-api/src/lib.rs"));
    let rust_value = extract_rust_const(&rust_source).expect(
        "ABI_MARKER not found in crates/thyllore-ml-api/src/lib.rs -- \
         expected `pub const ABI_MARKER: u32 = N;`",
    );

    let addon_source = read(&root.join("blender_addon/__init__.py"));
    let addon_value = extract_python_int(&addon_source, "EXPECTED_ABI_MARKER: int =").expect(
        "EXPECTED_ABI_MARKER not found in blender_addon/__init__.py -- \
             expected `EXPECTED_ABI_MARKER: int = N`",
    );

    assert_eq!(
        rust_value, addon_value,
        "ABI_MARKER mismatch: thyllore-ml-api says {rust_value}, \
         blender_addon/__init__.py says {addon_value}. \
         When bumping, update BOTH files in the same commit."
    );
}

#[test]
fn blender_runner_does_not_hardcode_abi_marker_value() {
    let root = workspace_root();
    let runner_path = root.join("crates/thyllore-ml-core/tests/curve_copilot_blender_runner.py");
    let source = read(&runner_path);

    let lines: Vec<&str> = source.lines().collect();
    for (idx, line) in lines.iter().enumerate() {
        if !line.contains("__abi_marker__") {
            continue;
        }
        let window_end = (idx + 12).min(lines.len());
        let window = lines[idx..window_end].join("\n");
        if window.contains("THYLLORE_EXPECTED_ABI_MARKER") {
            return;
        }
    }

    panic!(
        "{} references __abi_marker__ but does not read \
         THYLLORE_EXPECTED_ABI_MARKER from the environment. \
         Hardcoding the expected value reintroduces the bug fixed in \
         PR #100 -- the Rust caller injects the env var via \
         thyllore_ml_api::ABI_MARKER.",
        runner_path.display()
    );
}
