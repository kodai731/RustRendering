//! Closed-form guard: the flame runtime must stay analytic.
//!
//! Sample-based computation (ray-lattice quadrature, raymarch, LUT
//! interpolation) is only allowed in quarantined files or in entries of the
//! exception ledger below. Any new occurrence fails this test, so a sampled
//! path cannot slip back into the analytic pipeline unnoticed. Finite mode
//! sums, Chebyshev/Clenshaw evaluation, erf moments and fixed-count
//! Newton/bisection refinement are closed-form and are not flagged.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const PRODUCT_ENTRY: &str = "shaders/flameResolveFragment.frag";

/// Files whose whole purpose is sample-based reference integration. They may
/// contain lattice loops, but nothing outside this list may include them
/// except the product entry (which dispatches debug modes at runtime).
const SAMPLING_INCLUDES: &[&str] = &["include/flame_reference_march.glsl"];

const GLSL_BANNED_TOKENS: &[&str] = &["Raymarch", "raymarch", "FLAME_WAVE_SEGMENTS"];
const RUST_BANNED_TOKENS: &[&str] = &["Raymarch", "raymarch", "lut_lerp", "[f32; 33]"];

struct Exception {
    file_suffix: &'static str,
    token: &'static str,
    reason: &'static str,
}

/// Known sample-based remnants, each with the decision that keeps it alive.
/// Entries must still match a real occurrence; a stale entry fails the test.
const EXCEPTION_LEDGER: &[Exception] = &[
    Exception {
        file_suffix: "include/flame_radial_integral.glsl",
        token: "FLAME_WAVE_SEGMENTS",
        reason: "legacy 64-segment piecewise closed-form quadrature; the fully \
                 closed-form v5 replacement was rejected on look (2026-08-10) \
                 and stays env opt-in",
    },
    Exception {
        file_suffix: "flameResolveFragment.frag",
        token: "FLAME_WAVE_SEGMENTS",
        reason: "debug-view-only uses: segment-grid visualization (view 9) and \
                 the wave debug node search inside flameDebugViewColor",
    },
    Exception {
        file_suffix: "flameResolveFragment.frag",
        token: "Raymarch",
        reason: "runtime dispatch of push.mode 1/3 into the quarantined \
                 reference integrators; the entry routes but does not integrate",
    },
    Exception {
        file_suffix: "flame/settings.rs",
        token: "Raymarch",
        reason: "FlameShadingMode names the debug modes; selection data, not \
                 computation",
    },
    Exception {
        file_suffix: "flame/settings.rs",
        token: "raymarch",
        reason: "FlameShadingMode::parse accepts the debug mode name",
    },
    Exception {
        file_suffix: "flame/coefficients.rs",
        token: "lut_lerp",
        reason: "baked texture-fit envelope/radius LUT; scheduled for \
                 coefficient-form replacement or permanent registration in W9 S3",
    },
    Exception {
        file_suffix: "flame/coefficients.rs",
        token: "[f32; 33]",
        reason: "baked LUT storage, same pending decision as lut_lerp",
    },
    Exception {
        file_suffix: "flame/baked.rs",
        token: "[f32; 33]",
        reason: "FlameBaked component stores the texture-fit LUT; pending \
                 coefficient-form replacement (W9 S3 decision)",
    },
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn parse_includes(source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            line.strip_prefix("#include \"")
                .and_then(|rest| rest.strip_suffix('"'))
                .map(str::to_string)
        })
        .collect()
}

fn resolve_include(shader_dir: &Path, includer: &str, child: &str) -> String {
    let parent = Path::new(includer).parent().unwrap_or(Path::new(""));
    let joined = parent.join(child);
    if shader_dir.join(&joined).exists() {
        joined.to_string_lossy().into_owned()
    } else {
        child.to_string()
    }
}

fn collect_include_graph(root: &Path) -> BTreeSet<String> {
    let entry = root.join(PRODUCT_ENTRY);
    let shader_dir = entry.parent().unwrap().to_path_buf();
    let mut visited = BTreeSet::new();
    let mut queue: Vec<String> = parse_includes(&read(&entry))
        .into_iter()
        .map(|child| resolve_include(&shader_dir, "", &child))
        .collect();
    while let Some(rel) = queue.pop() {
        if !visited.insert(rel.clone()) {
            continue;
        }
        for child in parse_includes(&read(&shader_dir.join(&rel))) {
            queue.push(resolve_include(&shader_dir, &rel, &child));
        }
    }
    visited
}

fn is_excepted(file: &Path, token: &str, used: &mut BTreeSet<usize>) -> bool {
    let file = file.to_string_lossy();
    for (idx, e) in EXCEPTION_LEDGER.iter().enumerate() {
        if e.token == token && file.ends_with(e.file_suffix) {
            used.insert(idx);
            return true;
        }
    }
    false
}

fn scan_tokens(
    files: &[PathBuf],
    tokens: &[&str],
    used: &mut BTreeSet<usize>,
    violations: &mut Vec<String>,
) {
    for file in files {
        let source: String = read(file)
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        for token in tokens {
            if source.contains(token) && !is_excepted(file, token, used) {
                violations.push(format!("{}: `{token}`", file.display()));
            }
        }
    }
}

#[test]
fn flame_runtime_stays_closed_form() {
    let root = repo_root();
    let shader_dir = root.join("shaders");

    let graph = collect_include_graph(&root);
    for include in &graph {
        if SAMPLING_INCLUDES.contains(&include.as_str()) {
            continue;
        }
        for child in parse_includes(&read(&shader_dir.join(include))) {
            let child = resolve_include(&shader_dir, include, &child);
            assert!(
                !SAMPLING_INCLUDES.contains(&child.as_str()),
                "analytic include {include} pulls in sampling include {child}; \
                 sampling integrators may only be reached from the product \
                 entry's debug-mode dispatch"
            );
        }
    }

    let glsl_files: Vec<PathBuf> = std::iter::once(root.join(PRODUCT_ENTRY))
        .chain(
            graph
                .iter()
                .filter(|inc| !SAMPLING_INCLUDES.contains(&inc.as_str()))
                .map(|inc| shader_dir.join(inc)),
        )
        .collect();

    let flame_dir = root.join("crates/thyllore-render-core/src/flame");
    let rust_files: Vec<PathBuf> = fs::read_dir(&flame_dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", flame_dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "rs"))
        .filter(|p| p.file_name().is_some_and(|name| name != "tests.rs"))
        .collect();

    let mut used = BTreeSet::new();
    let mut violations = Vec::new();
    scan_tokens(&glsl_files, GLSL_BANNED_TOKENS, &mut used, &mut violations);
    scan_tokens(&rust_files, RUST_BANNED_TOKENS, &mut used, &mut violations);

    assert!(
        violations.is_empty(),
        "sample-computation tokens found outside the exception ledger:\n{}\n\
         Either restore the closed-form path or register a reasoned exception \
         in tests/closed_form_guard.rs",
        violations.join("\n")
    );

    let stale: Vec<String> = EXCEPTION_LEDGER
        .iter()
        .enumerate()
        .filter(|(idx, _)| !used.contains(idx))
        .map(|(_, e)| format!("{} / `{}`", e.file_suffix, e.token))
        .collect();
    assert!(
        stale.is_empty(),
        "stale exception ledger entries (token no longer present — remove them):\n{}",
        stale.join("\n")
    );
}
