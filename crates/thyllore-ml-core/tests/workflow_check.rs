use std::fs;
use std::path::PathBuf;

fn workflow_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(".github/workflows/python_parity.yml")
}

fn read_workflow() -> String {
    let path = workflow_path();
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e))
}

#[test]
fn ci_must_not_use_maturin_develop() {
    let content = read_workflow();
    assert!(
        !content.contains("maturin develop"),
        "python_parity.yml contains 'maturin develop' but it requires an active virtualenv \
         which GitHub Actions runners do NOT provide. \
         Use 'maturin build' + 'pip install <wheel>' instead. \
         See crates/thyllore-ml-core/scripts/verify_ci_workflow.ps1 for the correct pattern."
    );
}

#[test]
fn ci_must_use_maturin_build() {
    let content = read_workflow();
    assert!(
        content.contains("maturin build"),
        "python_parity.yml must use 'maturin build' to produce a wheel that can be \
         pip-installed without a virtualenv being active in CI."
    );
}

#[test]
fn ci_must_install_built_wheel() {
    let content = read_workflow();
    assert!(
        content.contains("pip install") && content.contains("dist/*.whl"),
        "python_parity.yml must install the wheel produced by maturin build via \
         'pip install $(ls dist/*.whl)' (or equivalent)."
    );
}

#[test]
fn ci_must_run_pytest_against_python_parity() {
    let content = read_workflow();
    assert!(
        content.contains("pytest tests/python_parity"),
        "python_parity.yml must run 'pytest tests/python_parity/' to verify Python \
         <-> Rust bit-identical parity."
    );
}

#[test]
fn ci_must_generate_rust_fixtures_before_pytest() {
    let content = read_workflow();
    let fixture_pos = content.find("parity_fixtures generate_parity_fixtures").expect(
        "python_parity.yml must run 'cargo test --test parity_fixtures generate_parity_fixtures' \
         before pytest so that Rust fixtures (JSON files compared by Python) exist",
    );
    let pytest_pos = content
        .find("pytest tests/python_parity")
        .expect("pytest step missing");
    assert!(
        fixture_pos < pytest_pos,
        "fixture generation must run BEFORE pytest in python_parity.yml \
         (otherwise Python tests will read stale or missing fixtures)"
    );
}

#[test]
fn ci_must_cover_python_3_10_3_11_3_12() {
    let content = read_workflow();
    for version in &["'3.10'", "'3.11'", "'3.12'"] {
        assert!(
            content.contains(version),
            "python_parity.yml must include {} in the matrix to cover Blender 4.0/4.1/4.2/4.3 \
             (abi3-py310 produces one wheel that targets all three Python versions)",
            version
        );
    }
}

#[test]
fn ci_must_cover_ubuntu_and_windows() {
    let content = read_workflow();
    assert!(
        content.contains("ubuntu-latest"),
        "python_parity.yml must include ubuntu-latest in the matrix"
    );
    assert!(
        content.contains("windows-latest"),
        "python_parity.yml must include windows-latest in the matrix \
         (Blender on Windows is the primary target platform)"
    );
}
