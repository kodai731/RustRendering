"""Vendored wheel bootstrap for the Thyllore Animation addon.

Extracts the .whl files in ``wheels/`` next to this module into a flat
``wheels-extracted/`` directory and inserts the per-wheel directories into
``sys.path`` BEFORE any grpc / thyllore_ml_core import, so the addon's
vendored versions take priority over any system-installed copies.

Why extract: Python's zipimport cannot load compiled extensions
(``cygrpc.so`` / ``thyllore_ml_core.abi3.so`` etc.) from inside a ``.whl``
zip archive — the dynamic loader requires a real on-disk file. Inserting the
``.whl`` path into sys.path therefore works for pure-Python wheels but fails
for grpcio and the L3 PyO3 wheel.

Extraction is idempotent: each wheel is unpacked once into
``<wheels-extracted>/<wheel_stem>/`` and a sentinel file marks completion so
subsequent registrations skip the work.

Layer responsibility (see Phase4_AddonRegistration.md):
- This module is the only place that touches sys.path.
- Operators must NEVER call sys.path.append themselves.
"""
from __future__ import annotations

import importlib
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List

_WHEELS_INSERTED: bool = False
_EXTRACTED_SENTINEL = ".thyllore_extracted"


def _wheels_dir() -> Path:
    """Resolve wheels/ relative to this file.

    Works both for the source-tree layout (``blender_addon/wheels/``) and for
    the extension ZIP layout where this module sits at the package root.
    """
    return Path(__file__).resolve().parent / "wheels"


def _extracted_root() -> Path:
    return Path(__file__).resolve().parent / "wheels-extracted"


def _is_already_imported(module_name: str) -> bool:
    prefix = module_name + "."
    return module_name in sys.modules or any(
        m == module_name or m.startswith(prefix) for m in sys.modules
    )


def _extract_wheel_once(wheel_path: Path, target_dir: Path) -> None:
    """Extract a single wheel into ``target_dir`` if not already done.

    Idempotent: re-running with the same wheel + target is a no-op once the
    sentinel file exists.
    """
    sentinel = target_dir / _EXTRACTED_SENTINEL
    if sentinel.is_file():
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(wheel_path) as zf:
        zf.extractall(target_dir)

    sentinel.touch()


def insert_wheels_to_sys_path() -> None:
    """Extract vendored wheels and insert their directories into sys.path.

    Idempotent — re-importing has no effect after the first call.
    """
    global _WHEELS_INSERTED
    if _WHEELS_INSERTED:
        return

    wheels_dir = _wheels_dir()
    if not wheels_dir.is_dir():
        # Development source layout without wheels/ populated yet.
        _WHEELS_INSERTED = True
        return

    conflict_modules: List[str] = [
        m
        for m in ("grpc", "google.protobuf", "thyllore_ml_core")
        if _is_already_imported(m)
    ]
    if conflict_modules:
        print(
            f"[Thyllore] WARNING: modules already imported by another addon — "
            f"vendored wheels may be ignored: {conflict_modules}"
        )

    extracted_root = _extracted_root()
    extracted_root.mkdir(parents=True, exist_ok=True)

    inserted = 0
    for wheel_path in sorted(wheels_dir.glob("*.whl")):
        target_dir = extracted_root / wheel_path.stem
        try:
            _extract_wheel_once(wheel_path, target_dir)
        except (OSError, zipfile.BadZipFile) as e:
            print(
                f"[Thyllore] WARNING: failed to extract {wheel_path.name}: {e}"
            )
            continue

        target_str = str(target_dir)
        if target_str not in sys.path:
            sys.path.insert(0, target_str)
            inserted += 1

    if inserted > 0:
        importlib.invalidate_caches()
        print(
            f"[Thyllore] Inserted {inserted} extracted wheels into sys.path"
        )

    _WHEELS_INSERTED = True


def remove_wheels_from_sys_path() -> None:
    """Remove vendored wheels from sys.path. Optional — Blender unregister
    typically does not need to call this, but it is exposed for tests."""
    global _WHEELS_INSERTED
    if not _WHEELS_INSERTED:
        return

    extracted_dir = str(_extracted_root())
    sys.path[:] = [p for p in sys.path if not p.startswith(extracted_dir)]
    _WHEELS_INSERTED = False


def is_wheel_inserted() -> bool:
    """Test helper."""
    return _WHEELS_INSERTED
