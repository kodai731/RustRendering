#!/usr/bin/env bash
# Idempotently vendor a patched copy of the `openusd` crate into vendor/openusd,
# which the workspace [patch.crates-io] points at.
#
# Why: openusd <= 0.5.0 mis-decodes USDC compressed integer-valued float arrays
# (code 'i'): it uses raw LZ4 (`read_compressed`) instead of the USD integer
# codec (`read_encoded_ints`), so rigid-bind jointWeights (all 1.0) read at the
# wrong length and skinned accessories (watch/earrings/badges/zipper/eyes) break.
# The C++ reference (crateFile.cpp) uses the integer codec, so this is a
# Rust-crate-only defect. See patches/openusd-jointweights.patch for the diff.
#
# Idempotent: skips immediately if vendor/openusd already matches the locked
# version and is already patched. Run it from the build script (it is cheap to
# call every build). Re-applies automatically after `cargo update` bumps openusd.
set -euo pipefail
cd "$(dirname "$0")/.."

OLD='let ints: Vec<i32> = self.read_compressed(count)?;'
NEW='let ints: Vec<i32> = self.read_encoded_ints(count)?;'
DEST="vendor/openusd"
TARGET="$DEST/src/usdc/reader.rs"

VER="$(awk '/^name = "openusd"$/{f=1} f&&/^version = /{gsub(/[",]/,"",$3); print $3; exit}' Cargo.lock 2>/dev/null || true)"
VER="${VER:-${1:-}}"
if [ -z "$VER" ]; then
  echo "[vendor_openusd] could not determine openusd version (run 'cargo fetch')" >&2
  exit 1
fi

# Fast path: already vendored at the locked version and already patched.
if [ -f "$DEST/Cargo.toml" ] \
   && grep -q "^version = \"$VER\"" "$DEST/Cargo.toml" \
   && grep -qF "$NEW" "$TARGET" 2>/dev/null; then
  echo "[vendor_openusd] openusd $VER already patched; skip"
  exit 0
fi

SRC="$(find "${CARGO_HOME:-$HOME/.cargo}/registry/src" -maxdepth 2 -type d -name "openusd-${VER}" 2>/dev/null | head -1)"
if [ -z "$SRC" ]; then
  echo "[vendor_openusd] openusd-${VER} not in cargo registry; run 'cargo fetch' first" >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p vendor
cp -r "$SRC" "$DEST"
chmod -R u+w "$DEST"

if grep -qF "$NEW" "$TARGET"; then
  : # upstream already fixed
elif grep -qF "$OLD" "$TARGET"; then
  sed -i "s|$OLD|$NEW|" "$TARGET"
else
  echo "[vendor_openusd] target line not found in $TARGET; upstream changed — update the fix" >&2
  exit 1
fi

grep -qF "$NEW" "$TARGET" || { echo "[vendor_openusd] patch failed" >&2; exit 1; }
echo "[vendor_openusd] patched openusd $VER -> $DEST"
