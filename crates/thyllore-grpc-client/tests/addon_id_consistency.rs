//! Static SSOT check for the Blender addon module name across the manifest
//! and every Python script under `blender_addon/`.
//!
//! Consumers that hardcode a stale extension id silently lose the addon at
//! runtime with `Add-on not loaded: "..."`. The CI parity workflows catch
//! this only after a full ZIP build + Blender install round-trip. This test
//! catches the same regressions in milliseconds without a Blender or wheel
//! build.
//!
//! Auto-discovery scope:
//!   - Walks every `*.py` under `blender_addon/` for a `candidates = [...]`
//!     literal that references `thyllore_animation*`. Each id mentioned must
//!     resolve to the manifest, in either bare or
//!     `bl_ext.user_default.<id>` form. Catches typos and stale ids.
//!   - Smoke-runner explicit gate: `blender_addon/tests/
//!     curve_copilot_operator_smoke.py` must enumerate the manifest id in
//!     both forms so addon_utils.check / addon_enable can resolve it.
//!
//! Run::
//!
//!     cargo test -p thyllore-grpc-client --test addon_id_consistency

use std::collections::HashSet;
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

fn collect_manifest_ids() -> Vec<(PathBuf, String)> {
    let root = workspace_root().join("blender_addon");
    let manifest_paths = [root.join("blender_manifest.toml")];
    let mut ids = Vec::new();
    for path in &manifest_paths {
        if !path.exists() {
            continue;
        }
        let id = extract_toml_id(&read(path))
            .unwrap_or_else(|| panic!("id not found in {}", path.display()));
        ids.push((path.clone(), id));
    }
    assert!(
        !ids.is_empty(),
        "no manifest ids extracted from blender_addon/"
    );
    ids
}

fn discover_python_files(dir: &Path, into: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if name == "__pycache__" || name == "wheels" || name == "assets" {
                continue;
            }
            discover_python_files(&path, into);
        } else if path.extension().is_some_and(|ext| ext == "py") {
            into.push(path);
        }
    }
}

fn extract_thyllore_ids_from_py(source: &str) -> HashSet<String> {
    let mut out: HashSet<String> = HashSet::new();
    let mut cursor = 0;
    let bytes = source.as_bytes();
    while cursor < bytes.len() {
        let Some(rel) = source[cursor..].find("thyllore_animation") else {
            break;
        };
        let start = cursor + rel;
        let end = source[start..]
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .map(|n| start + n)
            .unwrap_or(source.len());
        let token = &source[start..end];
        out.insert(token.to_string());
        cursor = end;
    }
    out
}

#[test]
fn every_python_addon_id_resolves_to_a_manifest() {
    let manifests = collect_manifest_ids();
    let valid_ids: HashSet<String> = manifests.iter().map(|(_, id)| id.clone()).collect();

    let mut python_files: Vec<PathBuf> = Vec::new();
    discover_python_files(&workspace_root().join("blender_addon"), &mut python_files);
    assert!(
        !python_files.is_empty(),
        "no Python files discovered under blender_addon/"
    );

    let mut bad_refs: Vec<String> = Vec::new();
    for path in &python_files {
        let source = read(path);
        let Some(candidates_block) = locate_candidates_list(&source) else {
            continue;
        };
        let ids_in_file = extract_thyllore_ids_from_py(candidates_block);
        for id in &ids_in_file {
            if !valid_ids.contains(id) {
                bad_refs.push(format!(
                    "{}: references `{id}` which is not in any manifest",
                    path.display()
                ));
            }
        }
    }

    assert!(
        bad_refs.is_empty(),
        "unknown thyllore_animation id(s) referenced in Python sources \
         (typo or stale id?):\n  {}\n\
         Valid manifest ids: {}",
        bad_refs.join("\n  "),
        valid_ids.iter().cloned().collect::<Vec<_>>().join(", ")
    );
}

#[test]
fn smoke_runner_handles_the_addon_id() {
    let manifests = collect_manifest_ids();
    let smoke_path = workspace_root().join("blender_addon/tests/curve_copilot_operator_smoke.py");
    let smoke_source = read(&smoke_path);

    let candidates_section = locate_candidates_list(&smoke_source).unwrap_or_else(|| {
        panic!(
            "could not locate `candidates = [...]` block in {}",
            smoke_path.display()
        )
    });

    let mut missing: Vec<String> = Vec::new();
    for (_, id) in &manifests {
        let canonical = format!("bl_ext.user_default.{id}");
        let bare = id.clone();
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
         The manifest id must appear in the candidates list as both the \
         `bl_ext.user_default.<id>` and `<id>` forms so addon_utils.check / \
         addon_enable can resolve regardless of install style.",
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
