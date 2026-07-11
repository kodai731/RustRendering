"""Curve Copilot feedback telemetry (bundled in mode B builds only).

Public surface used by the rest of the addon:
- ``record_prediction`` / ``mark_all_cleared`` -- stage-1 collection
- ``resolve_unlock_token`` / ``should_send`` -- gating helpers
- ``request_token_refresh`` -- opt-in handshake
- ``register`` / ``unregister`` -- operator + save handler lifecycle
"""
from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

from . import operators, records, sender
from .records import (
    complete_and_flush,
    mark_all_cleared,
    pending_count,
    record_prediction,
    request_token_refresh,
)
from .sender import discard_unlock_token, resolve_unlock_token, should_send


@persistent
def _on_save_post(_filepath) -> None:
    from .. import preferences

    complete_and_flush(preferences.get_preferences())


def register() -> None:
    operators.register()
    if _on_save_post not in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.append(_on_save_post)


def unregister() -> None:
    if _on_save_post in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.remove(_on_save_post)
    operators.unregister()
