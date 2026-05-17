---
paths:
  - "tests/**"
  - "crates/*/tests/**"
  - ".github/workflows/**"
  - "build-with-tests.ps1"
---

# Testing

## CI-Verified Tests Must Live Under `crates/thyllore-grpc-client/tests/`

**IMPORTANT:** Any test that must be verified by GitHub Actions MUST be placed under
`crates/thyllore-grpc-client/tests/` (or another lightweight workspace crate that does
not depend on the main `thyllore-animation` crate).

### Reason

The workspace root crate (`thyllore-animation`) depends on `vendor/imgui-sys`, whose
`build.rs` is intentionally NOT tracked in git (covered by `vendor/` in `.gitignore`).
GitHub Actions runners therefore CANNOT compile the root crate — any `cargo test` that
forces the root crate to build will fail with:

```
error: couldn't read `vendor/imgui-sys/build.rs`: No such file or directory
```

Tests placed under `tests/` at the workspace root require the root crate to compile and
will break CI.

### How to Apply

When adding a new integration test that should run in CI:

1. Place the test file under `crates/thyllore-grpc-client/tests/` (or another core crate
   without a vendored-imgui dependency, e.g., `thyllore-math-core`, `thyllore-anim-core`).
2. Reference workspace-root paths via `env!("CARGO_MANIFEST_DIR")` and walk up to the
   workspace root:
   ```rust
   fn workspace_root() -> PathBuf {
       Path::new(env!("CARGO_MANIFEST_DIR"))
           .ancestors()
           .nth(2)
           .expect("workspace root")
           .to_path_buf()
   }
   ```
3. Invoke the test in the workflow with `-p <crate-name>`:
   ```yaml
   cargo test -p thyllore-grpc-client --features <feature> --test <test_name>
   ```
4. Keep `[dev-dependencies]` (e.g., `tonic`, `tokio`) on the sub-crate, NOT on the
   workspace root, to avoid pulling them into root-crate builds.

### When Tests Belong at Workspace Root

`tests/` at the workspace root is for tests that intentionally exercise the full main
crate (e.g., `gltf_export_tests.rs`, `ecs_tests.rs`). These can ONLY run on developer
machines where `vendor/imgui-sys/build.rs` exists locally. They must NOT be wired into
GitHub Actions workflows.

## CI Reproduction — Run `scripts/collect_wheels.sh` Before Pushing

**IMPORTANT:** Before pushing any change that touches `crates/thyllore-ml-core`,
`pybindings/`, the blender addon, or anything affecting wheel builds, reproduce
the GitHub Actions "Collect vendored wheels" step locally with:

```bash
scripts/collect_wheels.sh
```

Pushing first to "see what CI says" wastes minutes per round-trip and burns
runner time. The local script runs the same maturin invocation CI does and
surfaces compile errors directly (unlike the PowerShell version, which used
`Out-Null` and hid the real cause).

### Why

- The host machine is Linux (migrated from Windows on 2026-05-04). The same
  glibc + Python toolchain as GitHub Actions `ubuntu-latest` is available
  natively; manylinux wheel filename tags, ONNX Runtime layout, and maturin
  output reproduce 1:1.
- Local turnaround is ~30 seconds (incremental build). A CI failure round-trip
  is several minutes per push.
- The Linux Build / parity / blender_parity workflows all begin with this same
  `collect_wheels` step. If maturin fails locally, every CI matrix entry will
  fail too — no need to wait for the runner.
- PR #105 (Phase 12 ISAB) caught a stale `pybindings/session.rs` signature
  locally via this script after CI had already failed across all three
  platforms (Linux / macOS / Windows) with output the PowerShell script had
  swallowed.

### How to Apply

Before pushing CI-affecting changes:

1. **Run the local reproduction**:
   ```bash
   scripts/collect_wheels.sh
   ```
   On first run it creates `.venv-collect-wheels/` (gitignored) and bootstraps
   pip. Set `PYTHON=...` to override the host Python or
   `THYLLORE_COLLECT_WHEELS_VENV=...` to point at an existing venv.
2. **For the full Blender parity pipeline**, use `scripts/run_parity_local.sh`,
   which wraps `collect_wheels.sh` + `build_blender_addon.ps1` + Blender install
   + cargo parity tests.
3. **For lib + integration tests**:
   ```bash
   cargo test --lib                                              # 167 tests, ml enabled
   cargo test --test ecs_tests --no-default-features             # 76 tests, ml disabled
   ```
4. **Add a `cargo test` shim** when a workflow step needs orchestration the
   shell can't express — see
   `crates/thyllore-grpc-client/tests/blender_addon_linux_validate.rs`
   (`#[ignore]`, runs Blender + reports clear skip message when missing).

### One-Time Setup

- `python3-venv` (Ubuntu: `sudo apt install python3.12-venv`) — required for
  the venv created by `collect_wheels.sh`.
- ONNX Runtime in `vendor/onnxruntime/onnxruntime-linux-x64-*/lib/` —
  `scripts/run_parity_local.sh` installs this automatically on first run.

### When CI is Still the Right Choice

- The bug only reproduces on macOS/arm64 hardware not available locally.
- The defect needs the exact GitHub Actions runner image (rare).
- A platform-specific Windows or macOS path must be exercised end-to-end.

In all other cases, exhaust the local reproduction first.

The project includes integration tests in the `tests/` directory:

## Test Files

**`integration_tests.rs`** - Project structure and configuration tests

- Verifies required directories exist
- Checks Cargo files and configuration
- Validates font and vendor directory structure

**`model_loading_tests.rs`** - Model loader tests

- Tests glTF and FBX model file existence
- Verifies model files are not empty
- Checks texture file availability
- Validates model directory structure

**`shader_tests.rs`** - Shader compilation tests

- Verifies shader source files exist
- Checks compiled shader files (`.spv`)
- Validates SPIR-V header format
- Ensures shader count matches between source and compiled files

## Test Counts

- Unit tests: 58 (math: 35, gltf: 11, fbx: 12)
- Integration tests: 31 (project structure: 12, model: 9, shader: 10)

## Running Tests

```bash
cargo test                              # Run all tests
cargo test --test integration_tests     # Run specific test file
cargo test --test model_loading_tests
cargo test --test shader_tests
cargo test -- --nocapture               # Run tests with output
cargo test -- --ignored                 # Run ignored tests
```

## Build + Test

Use `build-with-tests.ps1` to run build and tests sequentially, saving results to `log/log_test.txt`

```powershell
.\build-with-tests.ps1            # Build and run tests
.\build-with-tests.ps1 -Release   # Release build
.\build-with-tests.ps1 -SkipTests # Skip tests
```
