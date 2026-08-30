from __future__ import annotations

import hashlib
import importlib
import sys
import zipfile
from pathlib import Path
from typing import List

_WHEELS_INSERTED: bool = False
_EXTRACTED_SENTINEL = ".thyllore_extracted"


def _wheels_dir() -> Path:
    return Path(__file__).resolve().parent / "wheels"


def _extracted_root() -> Path:
    return Path(__file__).resolve().parent / "wheels-extracted"


def _is_blender_runtime() -> bool:
    try:
        import bpy
    except ImportError:
        return False
    return True


def _is_already_imported(module_name: str) -> bool:
    prefix = module_name + "."
    return module_name in sys.modules or any(
        m == module_name or m.startswith(prefix) for m in sys.modules
    )


def _wheel_sha256(wheel_path: Path) -> str:
    digest = hashlib.sha256()
    with wheel_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_wheel_once(wheel_path: Path, target_dir: Path) -> None:
    import shutil

    wheel_hash = _wheel_sha256(wheel_path)
    sentinel = target_dir / _EXTRACTED_SENTINEL
    if sentinel.is_file() and sentinel.read_text(encoding="utf-8").strip() == wheel_hash:
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(wheel_path) as zf:
        zf.extractall(target_dir)

    sentinel.write_text(wheel_hash, encoding="utf-8")


def insert_wheels_to_sys_path() -> None:
    global _WHEELS_INSERTED
    if _WHEELS_INSERTED:
        return

    if _is_blender_runtime():
        _WHEELS_INSERTED = True
        return

    wheels_dir = _wheels_dir()
    if not wheels_dir.is_dir():
        _WHEELS_INSERTED = True
        return

    conflict_modules: List[str] = [
        m
        for m in ("thyllore_effect_core",)
        if _is_already_imported(m)
    ]
    if conflict_modules:
        print(
            f"[Thyllore Flame] WARNING: modules already imported by another addon — "
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
                f"[Thyllore Flame] WARNING: failed to extract {wheel_path.name}: {e}"
            )
            continue

        target_str = str(target_dir)
        if target_str not in sys.path:
            sys.path.insert(0, target_str)
            inserted += 1

    if inserted > 0:
        importlib.invalidate_caches()
        print(
            f"[Thyllore Flame] Inserted {inserted} extracted wheels into sys.path"
        )

    _WHEELS_INSERTED = True


def remove_wheels_from_sys_path() -> None:
    global _WHEELS_INSERTED
    if not _WHEELS_INSERTED:
        return

    extracted_dir = str(_extracted_root())
    sys.path[:] = [p for p in sys.path if not p.startswith(extracted_dir)]
    _WHEELS_INSERTED = False


def is_wheel_inserted() -> bool:
    return _WHEELS_INSERTED
