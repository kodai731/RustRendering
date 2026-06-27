#!/usr/bin/env bash
# Regenerate a patched local copy of the `openusd` crate.
#
# Why: openusd <= 0.5.0 mis-decodes USDC compressed integer-valued float arrays
# (code 'i'): it uses raw LZ4 (`read_compressed`) instead of the USD integer
# codec (`read_encoded_ints`), so rigid-bind jointWeights (all 1.0) read at the
# wrong length and skinning breaks (watch/earrings/pins/zippers stretch).
# The C++ reference (crateFile.cpp `_ReadPossiblyCompressedArray`) uses the
# integer decompression, so this is a Rust-crate-only defect.
# See patches/openusd-jointweights.patch (one-line fix).
#
# This is a cargo-patch-equivalent without the external tool: it copies the
# crates.io source of the *locked* openusd version and applies the diff into
# vendor/openusd, which the workspace [patch.crates-io] points at. Re-run after
# `cargo update` bumps openusd; the diff re-applies if the upstream line is
# unchanged, and fails loudly otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."

VER="$(awk '/^name = "openusd"$/{f=1} f&&/^version = /{gsub(/["version =,]/,""); print; exit}' Cargo.lock 2>/dev/null || true)"
VER="${VER:-${1:-}}"
if [ -z "$VER" ]; then
  echo "could not determine openusd version from Cargo.lock; pass it explicitly: $0 0.5.0" >&2
  exit 1
fi

SRC="$(find "${CARGO_HOME:-$HOME/.cargo}/registry/src" -maxdepth 2 -type d -name "openusd-${VER}" 2>/dev/null | head -1)"
if [ -z "$SRC" ]; then
  echo "openusd-${VER} not found in the cargo registry. Run 'cargo fetch' first." >&2
  exit 1
fi

DEST="vendor/openusd"
rm -rf "$DEST"
mkdir -p vendor
cp -r "$SRC" "$DEST"
chmod -R u+w "$DEST"
patch -p1 -d "$DEST" < patches/openusd-jointweights.patch

echo "patched openusd ${VER} -> ${DEST}"
echo "ensure the workspace Cargo.toml has:"
echo "  [patch.crates-io]"
echo "  openusd = { path = \"vendor/openusd\" }"
