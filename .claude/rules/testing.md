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

## Linux Verification — Prefer WSL2 Before Touching CI

**IMPORTANT:** When you need to verify Linux behavior from a Windows development
machine (Blender extension validate, manylinux wheel layout, glibc-dependent
binaries), **try WSL2 first** before adding new CI runs or asking the user to
spin up a Linux VM.

### Why

- WSL2 Ubuntu has the same glibc + Python toolchain that GitHub Actions
  `ubuntu-latest` uses; manylinux wheel filename tags, TOML parsing, and
  Blender CLI behavior reproduce 1:1 between the two environments.
- Local WSL2 turnaround is ~30 seconds (build + validate). A CI failure
  round-trip is several minutes per push and consumes runner minutes.
- Phase 4 PR #97 caught three TOML/manifest defects locally via WSL2 in one
  iteration after CI had already failed twice without surfacing the root cause.

### How to Apply

Before pushing CI-only fixes for Linux issues:

1. **Install Blender 4.2 LTS into WSL2 Ubuntu** (one-time):
   ```bash
   wsl -d Ubuntu -- bash -lc '
   mkdir -p ~/blender_test && cd ~/blender_test &&
   wget -q -O blender.tar.xz https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz &&
   tar -xf blender.tar.xz && mv blender-4.2.0-linux-x64 blender && rm blender.tar.xz
   '
   ```
2. **Reproduce the CI environment locally** by collecting Linux wheels via WSL2
   (`pip download --platform manylinux2014_x86_64 ...`, `maturin build`).
3. **Run the actual CI step** (`build_blender_addon.ps1`,
   `blender --command extension validate ...`) before pushing.
4. **Add a `cargo test` shim** that orchestrates this when feasible — the
   pattern used by
   `crates/thyllore-grpc-client/tests/blender_addon_linux_validate.rs`:
   - `#[ignore]` so default `cargo test` skips it.
   - Detect `wsl.exe` + WSL Blender at `~/blender_test/blender/blender`
     (overridable via `THYLLORE_WSL_BLENDER_PATH`) and skip with a clear
     message when missing.
   - Run with `cargo test -p thyllore-grpc-client --test
     blender_addon_linux_validate -- --ignored --nocapture`.

### When CI is Still the Right Choice

- The bug only reproduces on macOS/arm64 hardware that WSL2 does not provide.
- The defect needs the exact GitHub Actions runner image (rare for our scope).
- A long-running matrix is required and developer machine cost is too high.

In all other cases, exhaust local WSL2 verification first.

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
