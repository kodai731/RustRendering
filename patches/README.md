# Dependency patches

## openusd-jointweights.patch

Fixes a decode bug in the `openusd` crate (≤ 0.5.0): USDC **compressed
integer-valued float arrays** (single-char code `'i'`) are decoded with raw LZ4
(`read_compressed`) instead of the USD integer codec (`read_encoded_ints`). This
makes rigid-bind `jointWeights` (all `1.0`) read at the wrong length, so skinned
accessories (watch, earrings, jacket badges, fannypack zipper/buckle, eyes)
stretch or sit at the wrong place. The C++ reference
(`crateFile.cpp::_ReadPossiblyCompressedArray`) uses the integer codec, so this
is a Rust-crate-only defect (one line).

This is a cargo-patch-style setup with **no external tool**: a committed diff is
applied to a local copy of the crate that `[patch.crates-io]` points at. The
crate version in `Cargo.toml` stays normal (upgradable); only the diff is
maintained.

### Files
- `patches/openusd-jointweights.patch` — the one-line fix (committed)
- `scripts/vendor_openusd_patch.sh` — copies the locked openusd source from the
  cargo registry into `vendor/openusd` and applies the patch (committed)
- `vendor/openusd/` — generated, **git-ignored**
- root `Cargo.toml` — `[patch.crates-io] openusd = { path = "vendor/openusd" }`

### Setup — automatic via the build scripts (nothing to remember)
The vendor+patch step runs idempotently at the start of every build, so a fresh
checkout (where `vendor/openusd` is absent) and a post-`cargo update` build both
self-heal:

- **Linux**: `./build.sh <cargo-args>` (e.g. `./build.sh build`, `./build.sh test --lib`)
- **Windows**: `.\build-with-tests.ps1`

Both call `scripts/vendor_openusd_patch.sh` (Linux) / `Ensure-PatchedOpenUsd`
(PowerShell), which:
- fast-skips if `vendor/openusd` already matches the locked version and is patched,
- otherwise copies the locked openusd source from the cargo registry and applies
  the one-line fix.

Only required manually: run `cargo fetch` once if the openusd source is not yet in
the registry. If you bypass the wrappers and run bare `cargo build`, run
`./scripts/vendor_openusd_patch.sh` first.

If upstream changes the patched line, the script aborts loudly — update the fix
(`old`/`new` strings in the script + this `.patch`) then.

### When openusd fixes this upstream
Drop `[patch.crates-io]`, this `patches/` dir, and the script; bump the crate
version normally.
