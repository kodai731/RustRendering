//! Static SSOT check for the Blender addon module name across manifests and
//! test scripts.
//!
//! Phase 5.5 introduced a `lite` Variant whose manifest id is
//! `thyllore_animation_lite`. Forgotten consumers that still hardcoded the
//! `thyllore_animation` (full Variant) id silently lose the addon at runtime
//! with `Add-on not loaded: "thyllore_animation"`. The CI parity workflows
//! catch this only after a full ZIP build + Blender install round-trip.
//! This test catches the same regressions in milliseconds without a Blender
//! or wheel build.
//!
//! Files inspected:
//!   - `blender_addon/blender_manifest.toml`           (full Variant id)
//!   - `blender_addon/blender_manifest.lite.toml`      (lite Variant id)
//!   - `blender_addon/tests/curve_copilot_operator_smoke.py` (smoke runner
//!     `candidates` list -- must enumerate every Variant id, both as the
//!     `bl_ext.user_default.<id>` form and the bare `<id>` legacy form)
//!
//! Run::
//!
//!     cargo test -p thyllore-grpc-client --test addon_id_consistency

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
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {} failed: {e}", path.display()))
}

fn extract_toml_id(toml_text: &str) -> Option<String> {
    for line in toml_text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("id") {
            let after_eq = rest.split('=').nth(1)?.trim();
            let value = after_eq.trim_matches(|c| c == '"' || c == '\'').trim();
            if !value.is_empty() && !value.starts_with('"') {
                return Some(value.to_string());
            }
        }
    }
    None
}

#[test]
fn smoke_runner_handles_every_addon_variant() {
    let root = workspace_root();

    let full_id = extract_toml_id(&read(&root.join("blender_addon/blender_manifest.toml")))
        .expect("id not found in blender_manifest.toml");
    let lite_id =
        extract_toml_id(&read(&root.join("blender_addon/blender_manifest.lite.toml")))
            .expect("id not found in blender_manifest.lite.toml");

    let smoke_path = root.join("blender_addon/tests/curve_copilot_operator_smoke.py");
    let smoke_source = read(&smoke_path);

    let candidates_section = locate_candidates_list(&smoke_source).unwrap_or_else(|| {
        panic!(
            "could not locate `candidates = [...]` block in {}",
            smoke_path.display()
        )
    });

    let mut missing: Vec<String> = Vec::new();
    for id in [full_id.as_str(), lite_id.as_str()] {
        let canonical = format!("bl_ext.user_default.{id}");
        let bare = id.to_string();
        if !candidates_section.contains(&format!("\"{canonical}\"")) {
            missing.push(canonical);
        }
        if !candidates_section.contains(&format!("\"{bare}\"")) {
            missing.push(bare);
        }
    }

    assert!(
        missing.is_empty(),
        "{} is missing addon id candidates: {}\n\
         When introducing a new manifest Variant, every id must appear in the \
         candidates list as both the `bl_ext.user_default.<id>` and `<id>` \
         forms so addon_utils.check / addon_enable can resolve regardless of \
         install style.",
        smoke_path.display(),
        missing.join(", ")
    );
}

fn locate_candidates_list(source: &str) -> Option<&str> {
    let key_pos = source.find("candidates")?;
    let bracket_open = source[key_pos..].find('[')? + key_pos;
    let bracket_close = source[bracket_open..].find(']')? + bracket_open;
    Some(&source[bracket_open..=bracket_close])
}
