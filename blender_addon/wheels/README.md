# Vendored Python Wheels

This directory holds the Python wheels bundled into the Blender extension ZIP.
The wheels themselves are **not committed** (large, OS-specific) and neither
is `HASHES.txt` (host-glibc-dependent for maturin output). Run
`scripts/collect_wheels.ps1` to populate this directory locally and in CI.

## What gets bundled

| Wheel | Purpose | Source |
|---|---|---|
| `thyllore_ml_core-0.0.1-cp310-abi3-{win_amd64,manylinux_2_*,macosx_*}.whl` | L3 PyO3 wheel for Tier B (Curve Copilot) | `crates/thyllore-ml-core` via maturin |
| `grpcio-1.71.2-cp310-{platform}.whl` | Tier A gRPC runtime | `pip download grpcio==1.71.2` |
| `grpcio_status-1.71.2-py3-none-any.whl` | grpcio sub-package, pure Python | same |
| `protobuf-5.29.6-{platform}.whl` | grpc internal | same |
| `certifi-2024.12.14-py3-none-any.whl` | TLS root CAs | same |

`thyllore_ml_core` is `abi3`, so a single cp310 wheel covers Python 3.10+
(Blender 4.0 = 3.10, 4.2 = 3.11, future = 3.12).

## Reproducible collection

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/collect_wheels.ps1
```

This populates **all three platforms** under `blender_addon/wheels/`. The
build script (`scripts/build_blender_addon.ps1 -Platform <p>`) prunes the
list to the active platform when staging.

The wheel filename's platform tag varies with the build host:
- pip's grpcio uses PEP 600 dual-tag: `manylinux_2_17_x86_64.manylinux2014_x86_64`.
- maturin tags with the host's glibc, e.g., Ubuntu 22 = `manylinux_2_35`,
  Ubuntu 24 = `manylinux_2_34`.

`build_blender_addon.ps1` accepts both forms (multi-pattern filter).

## Supply-chain integrity

Phase 4 trusts pip's PyPI signature checks for runtime wheels and trusts the
`maturin build` output for `thyllore_ml_core`. **No committed hash manifest
exists**, so `verify_wheel_hashes.ps1` has been removed.

`HASHES.txt` is still emitted locally by `collect_wheels.ps1` for debugging,
but it is gitignored.

Phase 6 will introduce a stronger guarantee using
[`pip install --require-hashes`](https://pip.pypa.io/en/stable/topics/secure-installs/)
with per-platform `requirements.lock` files committed to the repo. See
`SharedData/document/Rust_Rendering/Design/20260421_CommonaizeBlenderAddon/
20260421_BlenderAddonCommonizationDesign.md` (Phase 6 section) for the plan.

## ABI marker handshake

`thyllore_ml_core` exports `__abi_marker__` (sourced from
`crates/thyllore-ml-api/src/lib.rs::ABI_MARKER`). The build script verifies
that this value matches `EXPECTED_ABI_MARKER` in `blender_addon/__init__.py`
**before** producing a ZIP, so a wheel update without an addon update is
caught at build time rather than at user-facing register time.
