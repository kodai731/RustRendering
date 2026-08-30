from __future__ import annotations

import importlib


def _import_shared(name: str):
    try:
        return importlib.import_module(f"{__package__}.common.{name}")
    except ModuleNotFoundError:
        return importlib.import_module(f"blender_addon.common.{name}")


coordinates = _import_shared("coordinates")
