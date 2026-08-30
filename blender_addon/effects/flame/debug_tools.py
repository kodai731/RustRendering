from __future__ import annotations

try:
    from . import debug
except ImportError:
    debug = None


def is_available() -> bool:
    return debug is not None


def register() -> None:
    if debug is not None:
        debug.register()


def unregister() -> None:
    if debug is not None:
        debug.unregister()


def draw_panel(layout) -> None:
    if debug is not None:
        debug.draw_panel(layout)
