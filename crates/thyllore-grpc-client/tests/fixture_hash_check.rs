//! Phase 5 — Verifies that the shared fixture set on disk matches the SHA-256
//! recorded in `manifest.json`.
//!
//! Run from CI / WSL2 / Windows host:
//!     cargo test -p thyllore-grpc-client --test fixture_hash_check
//!
//! Skip behaviour: if the fixture root cannot be resolved, the test prints a
//! reason and returns OK so default `cargo test` does not fail on developer
//! machines without WSL2 set up. CI overrides `THYLLORE_SHARED_DATA_PATH`
//! explicitly so the skip never triggers there.
//!
//! Resolution order for the fixture root:
//! 1. `THYLLORE_PHASE5_FIXTURE_OUTPUT` env var (used by generators)
//! 2. `THYLLORE_SHARED_DATA_PATH` env var + `fixtures/ml_parity` suffix
//! 3. `paths.md`'s `SharedDataPathWSL` (POSIX, when running on Linux/WSL2)
//! 4. `paths.md`'s `SharedDataPath` (Windows UNC, last resort) — Rust on
//!    Windows requires the forward-slash form (`//wsl.localhost/Ubuntu/...`),
//!    which `paths.md` notes explicitly.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

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
    if let Some(p) = read_shared_data_path_from_paths_md() {
        return Some(PathBuf::from(p).join("fixtures").join("ml_parity"));
    }
    None
}

fn read_shared_data_path_from_paths_md() -> Option<String> {
    let paths_md = workspace_root().join(".claude/local/paths.md");
    let content = fs::read_to_string(&paths_md).ok()?;

    // Prefer POSIX form on Linux/WSL2 (faster I/O), fallback to UNC on Windows.
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
                    // On Windows, normalize backslash UNC to forward-slash UNC
                    // because `\\wsl.localhost\...` is not resolved by std::fs;
                    // see paths.md Notes.
                    if cfg!(windows) && value.starts_with(r"\\") {
                        return Some(value.replace('\\', "/"));
                    }
                    return Some(value.to_string());
                }
            }
        }
    }
    None
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
            "skip: fixture root {} does not exist. Run \
             scripts/generate_parity_fixtures.sh (recommended, WSL2) or \
             generate_parity_fixtures.ps1 first.",
            root.display()
        );
        return;
    }

    let manifest_path = root.join("manifest.json");
    if !manifest_path.exists() {
        eprintln!(
            "skip: manifest.json missing at {}. Regenerate fixtures.",
            manifest_path.display()
        );
        return;
    }

    let manifest_text = fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("read manifest {}: {e}", manifest_path.display()));

    let manifest = parse_manifest(&manifest_text);

    assert_eq!(
        manifest.schema_version, 1,
        "manifest.json schema_version mismatch (expected 1, got {})",
        manifest.schema_version
    );
    assert_eq!(
        manifest.proto_version, "v1",
        "manifest.json proto_version mismatch (expected v1, got {})",
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
        let actual = sha256_hex(&bytes);
        if actual != expected.sha256 {
            mismatches.push(format!(
                "{rel_path}: sha256 mismatch\n        expected: {}\n        actual:   {}",
                expected.sha256, actual
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

/// Minimal hand-rolled JSON reader for manifest.json. We avoid a `serde_json`
/// dev-dep on this crate to keep the dependency footprint tight; the format is
/// fixed and small.
fn parse_manifest(text: &str) -> Manifest {
    let schema_version = extract_u32(text, "\"schema_version\"")
        .expect("manifest.json missing or malformed schema_version");
    let proto_version = extract_string(text, "\"proto_version\"")
        .expect("manifest.json missing or malformed proto_version");

    let mut fixtures: BTreeMap<String, FixtureEntry> = BTreeMap::new();
    let body = slice_object(text, "\"fixtures\"")
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

fn slice_object<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let pos = text.find(key)? + key.len();
    let open = text[pos..].find('{')? + pos;
    let close = find_matching_brace(text, open)?;
    Some(&text[open + 1..close])
}

fn find_matching_brace(text: &str, open_idx: usize) -> Option<usize> {
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

/// Returns (key, slice_of_object_body, end_index_after_object) for the next
/// `"key": { ... }` pair starting at or after `cursor`. Returns None when no
/// more keyed objects exist.
fn next_keyed_object(body: &str, cursor: usize) -> Option<(String, &str, usize)> {
    let key_open = body[cursor..].find('"')? + cursor;
    let key_close = body[key_open + 1..].find('"')? + key_open + 1;
    let key = body[key_open + 1..key_close].to_string();

    let after_key = key_close + 1;
    let colon_pos = body[after_key..].find(':')? + after_key;
    let open = body[colon_pos..].find('{')? + colon_pos;
    let close = find_matching_brace(body, open)?;
    Some((key, &body[open + 1..close], close + 1))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::sync::OnceLock;
    static CACHE: OnceLock<()> = OnceLock::new();
    CACHE.get_or_init(|| {});
    // hand-rolled SHA-256 to avoid pulling in `sha2` as a dev-dep on grpc-client.
    let digest = sha256(bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{:02x}", byte));
    }
    hex
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    // Minimal SHA-256 implementation per RFC 6234. ~70 LOC.
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (bytes.len() as u64) * 8;
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod sha_self_test {
    use super::sha256_hex;

    #[test]
    fn sha256_of_empty_string() {
        // FIPS-180-4 reference vector: SHA-256("")
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_of_abc() {
        // FIPS-180-4 reference vector: SHA-256("abc")
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
