"""Vendored wheel bootstrap for the Thyllore Animation addon.

Inserts blender_addon/wheels/ into sys.path BEFORE any grpc / thyllore_ml_core
import so the addon's vendored versions take priority over any system-installed
copies. Imported first by __init__.py and idempotent — re-importing has no
effect after the first call.

Layer responsibility (see Phase4_AddonRegistration.md):
- This module is the only place that touches sys.path.
- Operators must NEVER call sys.path.append themselves.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import List

_WHEELS_INSERTED: bool = False


def _wheels_dir() -> Path:
    """Resolve wheels/ relative to this file.

    Works both for the source-tree layout (`blender_addon/wheels/`) and for the
    extension ZIP layout where this module sits at the package root.
    """
    return Path(__file__).resolve().parent / "wheels"


def _is_already_imported(module_name: str) -> bool:
    prefix = module_name + "."
    return module_name in sys.modules or any(
        m == module_name or m.startswith(prefix) for m in sys.modules
    )


def insert_wheels_to_sys_path() -> None:
    """Insert vendored wheels into sys.path. Idempotent."""
    global _WHEELS_INSERTED
    if _WHEELS_INSERTED:
        return

    wheels_dir = _wheels_dir()
    if not wheels_dir.is_dir():
        # Development source layout without wheels/ populated yet.
        # Fall through silently — Tier A may still work via system grpc and
        # Tier B will gray-out via tml.capabilities().
        _WHEELS_INSERTED = True
        return

    conflict_modules: List[str] = [
        m
        for m in ("grpc", "google.protobuf", "thyllore_ml_core")
        if _is_already_imported(m)
    ]
    if conflict_modules:
        # We cannot safely override an already-loaded grpcio (different ABI).
        # Surface the conflict to the user log; the existing modules will be
        # used and the addon may still work if they are compatible.
        print(
            f"[Thyllore] WARNING: modules already imported by another addon — "
            f"vendored wheels may be ignored: {conflict_modules}"
        )

    inserted = 0
    for wheel_path in sorted(str(p) for p in wheels_dir.glob("*.whl")):
        if wheel_path not in sys.path:
            sys.path.insert(0, wheel_path)
            inserted += 1

    if inserted > 0:
        importlib.invalidate_caches()
        print(f"[Thyllore] Inserted {inserted} vendored wheels into sys.path")

    _WHEELS_INSERTED = True


def remove_wheels_from_sys_path() -> None:
    """Remove vendored wheels from sys.path. Optional — Blender unregister
    typically does not need to call this, but it is exposed for tests."""
    global _WHEELS_INSERTED
    if not _WHEELS_INSERTED:
        return

    wheels_dir = str(_wheels_dir())
    sys.path[:] = [p for p in sys.path if not p.startswith(wheels_dir)]
    _WHEELS_INSERTED = False


def is_wheel_inserted() -> bool:
    """Test helper."""
    return _WHEELS_INSERTED
