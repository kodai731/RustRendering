use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_PROTO_VERSION: &str = "v1";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root (two levels above grpc-client crate)")
        .to_path_buf()
}

fn resolve_fixture_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("THYLLORE_PHASE5_FIXTURE_OUTPUT") {
        return Some(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("THYLLORE_SHARED_DATA_PATH") {
        return Some(PathBuf::from(p).join("fixtures").join("ml_parity"));
    }
    let shared_data = read_shared_data_path_from_paths_md()?;
    Some(PathBuf::from(shared_data).join("fixtures").join("ml_parity"))
}

fn read_shared_data_path_from_paths_md() -> Option<String> {
    let paths_md = workspace_root().join(".claude/local/paths.md");
    let content = fs::read_to_string(&paths_md).ok()?;

    let primary_key = if cfg!(unix) {
        "SharedDataPathWSL"
    } else {
        "SharedDataPath"
    };
    let secondary_key = if cfg!(unix) {
        "SharedDataPath"
    } else {
        "SharedDataPathWSL"
    };

    for key in [primary_key, secondary_key] {
        let prefix = format!("- {key} = ");
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix(prefix.as_str()) {
                let value = rest.trim();
                if !value.is_empty() {
                    return Some(normalize_unc_for_rust(value));
                }
            }
        }
    }
    None
}

// `\\wsl.localhost\...` (canonical UNC) is rejected by std::fs on Windows for
// the wsl.localhost virtual provider; the forward-slash form is accepted.
fn normalize_unc_for_rust(value: &str) -> String {
    if cfg!(windows) && value.starts_with(r"\\") {
        value.replace('\\', "/")
    } else {
        value.to_string()
    }
}

#[test]
fn fixture_root_matches_manifest() {
    let Some(root) = resolve_fixture_root() else {
        eprintln!(
            "skip: cannot resolve fixture root. Set THYLLORE_SHARED_DATA_PATH \
             or THYLLORE_PHASE5_FIXTURE_OUTPUT, or ensure paths.md exists."
        );
        return;
    };

    if !root.exists() {
        eprintln!(
            "skip: fixture root {} does not exist; run scripts/generate_parity_fixtures.sh",
            root.display()
        );
        return;
    }

    let manifest_path = root.join("manifest.json");
    if !manifest_path.exists() {
        eprintln!(
            "skip: manifest.json missing at {}; regenerate fixtures",
            manifest_path.display()
        );
        return;
    }

    let manifest_text = fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("read manifest {}: {e}", manifest_path.display()));

    let manifest = parse_manifest(&manifest_text);

    assert_eq!(
        manifest.schema_version, EXPECTED_SCHEMA_VERSION,
        "manifest.json schema_version mismatch (expected {EXPECTED_SCHEMA_VERSION}, got {})",
        manifest.schema_version
    );
    assert_eq!(
        manifest.proto_version, EXPECTED_PROTO_VERSION,
        "manifest.json proto_version mismatch (expected {EXPECTED_PROTO_VERSION}, got {})",
        manifest.proto_version
    );

    let mut mismatches: Vec<String> = Vec::new();
    for (rel_path, expected) in &manifest.fixtures {
        let abs = root.join(rel_path);
        let bytes = match fs::read(&abs) {
            Ok(b) => b,
            Err(e) => {
                mismatches.push(format!("{rel_path}: read failed: {e}"));
                continue;
            }
        };
        if (bytes.len() as u64) != expected.size_bytes {
            mismatches.push(format!(
                "{rel_path}: size mismatch (expected {}, actual {})",
                expected.size_bytes,
                bytes.len()
            ));
        }
        let actual_sha = sha256_hex(&bytes);
        if actual_sha != expected.sha256 {
            mismatches.push(format!(
                "{rel_path}: sha256 mismatch\n        expected: {}\n        actual:   {}",
                expected.sha256, actual_sha
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "fixture hash check failed for {} entries:\n  {}",
        mismatches.len(),
        mismatches.join("\n  ")
    );

    eprintln!(
        "fixture_hash_check OK: {} entries verified at {}",
        manifest.fixtures.len(),
        root.display()
    );
}

struct Manifest {
    schema_version: u32,
    proto_version: String,
    fixtures: BTreeMap<String, FixtureEntry>,
}

struct FixtureEntry {
    sha256: String,
    size_bytes: u64,
}

fn parse_manifest(text: &str) -> Manifest {
    let schema_version = extract_u32(text, "\"schema_version\"")
        .expect("manifest.json missing or malformed schema_version");
    let proto_version = extract_string(text, "\"proto_version\"")
        .expect("manifest.json missing or malformed proto_version");

    let mut fixtures: BTreeMap<String, FixtureEntry> = BTreeMap::new();
    let body = slice_object_body(text, "\"fixtures\"")
        .expect("manifest.json missing or malformed fixtures object");

    let mut cursor = 0_usize;
    while let Some((rel_path, entry_body, next)) = next_keyed_object(body, cursor) {
        let sha256 = extract_string(entry_body, "\"sha256\"")
            .unwrap_or_else(|| panic!("entry {rel_path}: missing sha256"));
        let size_bytes = extract_u64(entry_body, "\"size_bytes\"")
            .unwrap_or_else(|| panic!("entry {rel_path}: missing size_bytes"));
        fixtures.insert(rel_path, FixtureEntry { sha256, size_bytes });
        cursor = next;
    }

    Manifest {
        schema_version,
        proto_version,
        fixtures,
    }
}

fn extract_u32(text: &str, key: &str) -> Option<u32> {
    let pos = text.find(key)? + key.len();
    let after_colon = text[pos..].find(':')? + pos + 1;
    let rest = &text[after_colon..];
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == '\n')
        .unwrap_or(rest.len());
    rest[..end].trim().parse::<u32>().ok()
}

fn extract_u64(text: &str, key: &str) -> Option<u64> {
    let pos = text.find(key)? + key.len();
    let after_colon = text[pos..].find(':')? + pos + 1;
    let rest = &text[after_colon..];
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == '\n')
        .unwrap_or(rest.len());
    rest[..end].trim().parse::<u64>().ok()
}

fn extract_string(text: &str, key: &str) -> Option<String> {
    let pos = text.find(key)? + key.len();
    let after_colon = text[pos..].find(':')? + pos + 1;
    let after_quote = text[after_colon..].find('"')? + after_colon + 1;
    let end_quote = text[after_quote..].find('"')? + after_quote;
    Some(text[after_quote..end_quote].to_string())
}

fn slice_object_body<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let pos = text.find(key)? + key.len();
    let open = text[pos..].find('{')? + pos;
    let close = find_matching_close_brace(text, open)?;
    Some(&text[open + 1..close])
}

fn find_matching_close_brace(text: &str, open_idx: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escape = false;
    for i in open_idx..bytes.len() {
        let c = bytes[i];
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match c {
                b'\\' => escape = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn next_keyed_object(body: &str, cursor: usize) -> Option<(String, &str, usize)> {
    let key_open = body[cursor..].find('"')? + cursor;
    let key_close = body[key_open + 1..].find('"')? + key_open + 1;
    let key = body[key_open + 1..key_close].to_string();

    let after_key = key_close + 1;
    let colon_pos = body[after_key..].find(':')? + after_key;
    let open = body[colon_pos..].find('{')? + colon_pos;
    let close = find_matching_close_brace(body, open)?;
    Some((key, &body[open + 1..close], close + 1))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{:02x}", byte));
    }
    hex
}

#[cfg(test)]
mod sha_self_test {
    use super::sha256_hex;

    #[test]
    fn sha256_of_empty_string() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_of_abc() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
