# Vendored Python Wheels

This directory holds the Python wheels bundled into the Blender extension ZIP.
The wheels themselves are **not committed** (large, OS-specific) and neither
is `HASHES.txt` (host-glibc-dependent for maturin output). Run
`scripts/collect_wheels.ps1` (or `.sh`) to populate this directory locally
and in CI.

## What gets bundled

| Wheel | Purpose | Source |
|---|---|---|
| `thyllore_ml_core-0.0.1-cp310-abi3-{win_amd64,manylinux_2_*,macosx_*}.whl` | PyO3 wheel for Curve Copilot | `crates/thyllore-ml-core` via maturin |

`thyllore_ml_core` is `abi3`, so a single cp310 wheel covers Python 3.10+
(Blender 4.0 = 3.10, 4.2 = 3.11, future = 3.12).

## Reproducible collection

```bash
bash scripts/collect_wheels.sh
```

The wheel filename's platform tag varies with the build host: maturin tags
with the host's glibc, e.g., Ubuntu 22 = `manylinux_2_35`, Ubuntu 24 =
`manylinux_2_34`. `build_blender_addon.sh/.ps1` accepts all forms
(multi-pattern filter).

## Supply-chain integrity

The build trusts the `maturin build` output for `thyllore_ml_core`.
`HASHES.txt` is emitted locally by `collect_wheels` for debugging, but it is
gitignored.

## ABI marker handshake

`thyllore_ml_core` exports `__abi_marker__` (sourced from
`crates/thyllore-ml-api/src/lib.rs::ABI_MARKER`). The build script verifies
that this value matches `EXPECTED_ABI_MARKER` in `blender_addon/__init__.py`
**before** producing a ZIP, so a wheel update without an addon update is
caught at build time rather than at user-facing register time.
