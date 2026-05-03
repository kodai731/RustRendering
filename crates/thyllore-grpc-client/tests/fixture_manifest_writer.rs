//! Walks `fixtures/ml_parity/` (or `THYLLORE_PARITY_FIXTURE_OUTPUT`) and
//! rewrites `manifest.json` with the current sha256 / size for every fixture
//! file. Mirrors the Python block in `scripts/generate_parity_fixtures.sh` so
//! the cargo-only flow (CI) produces a manifest in sync with the fixtures it
//! just generated.
//!
//! Marked `#[ignore]` because:
//!   - It mutates `fixtures/ml_parity/manifest.json`
//!   - It only makes sense AFTER the proto + numpy generators have run
//!     (`grpc_fixture_generator` + `curve_copilot_fixture_generator`)
//!
//! Run via::
//!
//!     cargo test -p thyllore-grpc-client --test fixture_manifest_writer \
//!         write_manifest_from_fixture_dir -- --ignored --nocapture
//!
//! Then `fixture_hash_check::fixture_root_matches_manifest` can verify
//! integrity.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const SCHEMA_VERSION: u32 = 1;
const PROTO_VERSION: &str = "v1";
const EXCLUDED_FILENAMES: &[&str] = &["manifest.json", "README.md", ".gitkeep"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn resolve_fixture_root() -> PathBuf {
    if let Ok(p) = std::env::var("THYLLORE_PARITY_FIXTURE_OUTPUT") {
        return PathBuf::from(p);
    }
    workspace_root().join("fixtures").join("ml_parity")
}

#[test]
#[ignore = "mutates fixtures/ml_parity/manifest.json; run after the generators"]
fn write_manifest_from_fixture_dir() {
    let root = resolve_fixture_root();
    assert!(
        root.exists(),
        "fixture root {} does not exist; run the generators first \
         (cargo test --test grpc_fixture_generator + \
         --test curve_copilot_fixture_generator with --ignored)",
        root.display()
    );

    let mut entries: BTreeMap<String, FixtureEntry> = BTreeMap::new();
    walk_dir(&root, &root, &mut entries);

    let commit = git_short_commit(&workspace_root()).unwrap_or_else(|| "unknown".into());
    let generated_at = current_utc_timestamp();
    let onnx_revision = std::env::var("ONNX_REVISION").unwrap_or_else(|_| "main".into());

    let json = render_manifest(
        SCHEMA_VERSION,
        PROTO_VERSION,
        &generated_at,
        &commit,
        &onnx_revision,
        &entries,
    );

    let manifest_path = root.join("manifest.json");
    fs::write(&manifest_path, json)
        .unwrap_or_else(|e| panic!("write {} failed: {e}", manifest_path.display()));

    eprintln!(
        "manifest_writer: wrote {} ({} fixtures)",
        manifest_path.display(),
        entries.len()
    );
}

struct FixtureEntry {
    sha256: String,
    size_bytes: u64,
}

fn walk_dir(root: &Path, dir: &Path, into: &mut BTreeMap<String, FixtureEntry>) {
    let read = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) => panic!("read_dir {} failed: {e}", dir.display()),
    };

    let mut sorted: Vec<PathBuf> = read.filter_map(|e| e.ok().map(|d| d.path())).collect();
    sorted.sort();

    for path in sorted {
        if path.is_dir() {
            walk_dir(root, &path, into);
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if EXCLUDED_FILENAMES.contains(&name) {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .expect("strip_prefix")
            .to_string_lossy()
            .replace('\\', "/");
        let bytes =
            fs::read(&path).unwrap_or_else(|e| panic!("read {} failed: {e}", path.display()));
        into.insert(
            rel,
            FixtureEntry {
                sha256: sha256_hex(&bytes),
                size_bytes: bytes.len() as u64,
            },
        );
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{:02x}", byte));
    }
    hex
}

fn git_short_commit(workspace_root: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .current_dir(workspace_root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn current_utc_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before UNIX_EPOCH")
        .as_secs();

    let days_since_epoch = secs / 86_400;
    let seconds_in_day = secs % 86_400;
    let h = seconds_in_day / 3600;
    let m = (seconds_in_day / 60) % 60;
    let s = seconds_in_day % 60;
    let (y, mo, d) = ymd_from_days_since_epoch(days_since_epoch);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn ymd_from_days_since_epoch(days: u64) -> (i32, u32, u32) {
    let days = days as i64 + 719468;
    let era = days.div_euclid(146097);
    let doe = (days - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y_final = if m <= 2 { y + 1 } else { y };
    (y_final as i32, m as u32, d as u32)
}

fn render_manifest(
    schema_version: u32,
    proto_version: &str,
    generated_at: &str,
    commit: &str,
    onnx_revision: &str,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> String {
    let mut out = String::new();
    out.push_str("{\n");

    let mut fixture_lines: Vec<String> = Vec::with_capacity(fixtures.len());
    for (rel, entry) in fixtures {
        fixture_lines.push(format!(
            "    \"{}\": {{\n      \"sha256\": \"{}\",\n      \"size_bytes\": {}\n    }}",
            json_escape(rel),
            entry.sha256,
            entry.size_bytes
        ));
    }
    out.push_str(&format!(
        "  \"fixtures\": {{\n{}\n  }},\n",
        fixture_lines.join(",\n")
    ));
    out.push_str(&format!(
        "  \"generated_at\": \"{}\",\n",
        json_escape(generated_at)
    ));
    out.push_str(
        "  \"generator\": \"crates/thyllore-grpc-client/tests/fixture_manifest_writer.rs\",\n",
    );
    out.push_str(&format!(
        "  \"onnx_huggingface_revision\": \"{}\",\n",
        json_escape(onnx_revision)
    ));
    out.push_str(&format!(
        "  \"proto_version\": \"{}\",\n",
        json_escape(proto_version)
    ));
    out.push_str(&format!("  \"schema_version\": {schema_version},\n"));
    out.push_str(&format!(
        "  \"thyllore_animation_commit\": \"{}\"\n",
        json_escape(commit)
    ));
    out.push_str("}\n");
    out
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}
