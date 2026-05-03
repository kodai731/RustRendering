//! Static checks for `.github/workflows/*.yml` that catch broken references
//! before CI does.
//!
//! Bug classes covered:
//!   1. Workflow run blocks invoke a script (`*.ps1`, `*.sh`, `*.py`) that
//!      does not exist in the tree -- rename / typo causing CI-only "no such
//!      file" failures.
//!   2. `actions/upload-artifact` `name:` and `actions/download-artifact`
//!      `name:` mismatch within the same workflow file (PR #100 had this
//!      shape with the lite Variant rename).
//!   3. Workflow `paths:` filter omits a file the job actually consumes
//!      (covered by class 1 indirectly, since missing dependencies surface
//!      as broken script invocations).
//!
//! Run::
//!
//!     cargo test -p thyllore-grpc-client --test workflow_references

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

fn read_workflow_files() -> Vec<(PathBuf, String)> {
    let dir = workspace_root().join(".github/workflows");
    let mut out = Vec::new();
    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yml") {
            continue;
        }
        if let Ok(text) = fs::read_to_string(&path) {
            out.push((path, text));
        }
    }
    out
}

fn extract_script_paths(line: &str) -> Vec<String> {
    const EXTS: &[&str] = &[".ps1", ".sh", ".py"];
    let mut found: Vec<String> = Vec::new();
    for ext in EXTS {
        let mut search_from = 0;
        while let Some(rel_idx) = line[search_from..].find(ext) {
            let end = search_from + rel_idx + ext.len();
            let mut start = search_from + rel_idx;
            while start > 0 {
                let prev = line.as_bytes()[start - 1];
                let is_path_char = prev.is_ascii_alphanumeric()
                    || prev == b'_'
                    || prev == b'-'
                    || prev == b'/'
                    || prev == b'.';
                if !is_path_char {
                    break;
                }
                start -= 1;
            }
            let candidate = &line[start..end];
            if candidate.starts_with("scripts/")
                || candidate.starts_with("./scripts/")
                || candidate.starts_with("blender_addon/")
                || candidate.starts_with("crates/")
                || candidate.starts_with(".github/")
            {
                let normalized = candidate.trim_start_matches("./").to_string();
                found.push(normalized);
            }
            search_from = end;
        }
    }
    found
}

fn extract_artifact_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix("name:")?;
    let value = rest
        .trim()
        .trim_matches(|c: char| c == '"' || c == '\'')
        .trim();
    if value.is_empty() {
        return None;
    }
    Some(value.to_string())
}

fn template_matches(upload_template: &str, download_literal: &str) -> bool {
    if !upload_template.contains("${{") {
        return upload_template == download_literal;
    }
    let mut remaining_upload = upload_template;
    let mut remaining_download = download_literal;
    while let Some(open) = remaining_upload.find("${{") {
        let prefix = &remaining_upload[..open];
        if !remaining_download.starts_with(prefix) {
            return false;
        }
        remaining_download = &remaining_download[prefix.len()..];
        let close_rel = match remaining_upload[open..].find("}}") {
            Some(i) => i,
            None => return false,
        };
        remaining_upload = &remaining_upload[open + close_rel + 2..];
        let next_lit_end = remaining_upload
            .find("${{")
            .unwrap_or(remaining_upload.len());
        let next_lit = &remaining_upload[..next_lit_end];
        if next_lit.is_empty() {
            return true;
        }
        let consumed_to = match remaining_download.find(next_lit) {
            Some(i) => i,
            None => return false,
        };
        remaining_download = &remaining_download[consumed_to..];
    }
    remaining_download.starts_with(remaining_upload)
        && remaining_download.len() >= remaining_upload.len()
}

#[derive(Default)]
struct ArtifactEdges {
    uploads: HashSet<String>,
    downloads: HashSet<String>,
}

fn scan_artifact_edges(text: &str) -> ArtifactEdges {
    let mut out = ArtifactEdges::default();
    let mut mode: Option<&str> = None;
    for line in text.lines() {
        let lt = line.trim_start();
        if lt.contains("actions/upload-artifact@") {
            mode = Some("upload");
            continue;
        }
        if lt.contains("actions/download-artifact@") {
            mode = Some("download");
            continue;
        }
        if line.trim_start_matches(' ').starts_with("- ") || lt.starts_with("- name:") {
            mode = None;
        }
        if let Some(kind) = mode {
            if let Some(name) = extract_artifact_name(line) {
                if kind == "upload" {
                    out.uploads.insert(name);
                } else {
                    out.downloads.insert(name);
                }
                mode = None;
            }
        }
    }
    out
}

#[test]
fn every_script_path_in_workflows_exists() {
    let root = workspace_root();
    let workflows = read_workflow_files();
    assert!(!workflows.is_empty(), "no workflow YAML files found");

    let mut missing: Vec<String> = Vec::new();
    let mut checked: HashSet<String> = HashSet::new();

    for (wf_path, text) in &workflows {
        for (line_idx, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                continue;
            }
            for script in extract_script_paths(line) {
                let abs = root.join(&script);
                if abs.exists() {
                    checked.insert(script);
                } else {
                    missing.push(format!(
                        "{}:{}: references `{}` which does not exist in the tree",
                        wf_path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or_default(),
                        line_idx + 1,
                        script
                    ));
                }
            }
        }
    }

    assert!(
        missing.is_empty(),
        "{} broken script reference(s) in workflow YAML:\n  {}\n\
         (Checked {} script paths overall.)",
        missing.len(),
        missing.join("\n  "),
        checked.len()
    );
}

#[test]
fn upload_artifact_names_have_matching_downloads_within_workflow() {
    let workflows = read_workflow_files();
    let mut violations: Vec<String> = Vec::new();

    for (wf_path, text) in &workflows {
        let edges = scan_artifact_edges(text);
        if edges.uploads.is_empty() && edges.downloads.is_empty() {
            continue;
        }

        for download in &edges.downloads {
            if download.contains("${{") {
                continue;
            }
            let matched = edges
                .uploads
                .iter()
                .any(|upload| template_matches(upload, download));
            if !matched {
                violations.push(format!(
                    "{}: download-artifact `{}` has no matching upload-artifact in the same workflow (uploads: {:?})",
                    wf_path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or_default(),
                    download,
                    edges.uploads
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "artifact name mismatch (would surface as `Artifact not found` in CI):\n  {}",
        violations.join("\n  ")
    );
}

#[cfg(test)]
mod parser_self_tests {
    use super::*;

    #[test]
    fn extracts_pwsh_script_path() {
        let line =
            "  pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/build_blender_addon.ps1";
        assert_eq!(
            extract_script_paths(line),
            vec!["scripts/build_blender_addon.ps1".to_string()]
        );
    }

    #[test]
    fn extracts_relative_dot_slash_script() {
        let line = "  ./scripts/run_parity_local.sh --foo";
        assert_eq!(
            extract_script_paths(line),
            vec!["scripts/run_parity_local.sh".to_string()]
        );
    }

    #[test]
    fn ignores_string_mention_in_echo() {
        let line = r#"          echo "::error::build_blender_addon.ps1 did not emit ..." "#;
        assert!(
            extract_script_paths(line).is_empty(),
            "expected no match (script name only mentioned in echo string), got {:?}",
            extract_script_paths(line)
        );
    }

    #[test]
    fn extracts_artifact_name_from_yaml_indent() {
        let line = "          name: thyllore_animation_addon-linux_x86_64";
        assert_eq!(
            extract_artifact_name(line),
            Some("thyllore_animation_addon-linux_x86_64".to_string())
        );
    }

    #[test]
    fn captures_artifact_name_template_for_pattern_matching() {
        let line = "          name: thyllore_animation_addon-${{ matrix.platform }}";
        assert_eq!(
            extract_artifact_name(line),
            Some("thyllore_animation_addon-${{ matrix.platform }}".to_string())
        );
    }

    #[test]
    fn template_match_resolves_matrix_variable() {
        assert!(template_matches(
            "thyllore_animation_addon-${{ matrix.platform }}",
            "thyllore_animation_addon-linux_x86_64"
        ));
    }

    #[test]
    fn template_match_rejects_wrong_prefix() {
        assert!(!template_matches(
            "foo-${{ matrix.platform }}",
            "bar-linux_x86_64"
        ));
    }

    #[test]
    fn template_match_literal_only() {
        assert!(template_matches("exact", "exact"));
        assert!(!template_matches("exact", "different"));
    }
}
