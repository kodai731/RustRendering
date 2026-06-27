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

### Setup / required steps
1. `cargo fetch` (ensure the openusd source is in the cargo registry)
2. `./scripts/vendor_openusd_patch.sh` (regenerates `vendor/openusd`)
3. `cargo build`

Run step 2 again after a fresh checkout (the vendored copy is git-ignored) and
after `cargo update` bumps openusd. If upstream changed the patched line the
`patch` step fails loudly — update this diff then.

### When openusd fixes this upstream
Drop `[patch.crates-io]`, this `patches/` dir, and the script; bump the crate
version normally.
