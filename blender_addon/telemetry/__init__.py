"""Curve Copilot feedback telemetry (bundled in mode B builds only).

Public surface used by the rest of the addon:
- ``record_prediction`` / ``mark_all_cleared`` -- stage-1 collection
- ``resolve_full_token`` / ``should_send`` -- gating helpers
- ``request_token_refresh`` -- opt-in handshake
- ``register`` / ``unregister`` -- save handler lifecycle

Free-text feedback is NOT part of this package: it ships in every build mode
via ``feedback_message``.
"""
from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

from . import records, sender
from .records import (
    complete_and_flush,
    mark_all_cleared,
    pending_count,
    record_prediction,
    request_token_refresh,
)
from .sender import discard_full_token, resolve_full_token, should_send


@persistent
def _on_save_post(_filepath) -> None:
    from .. import preferences

    complete_and_flush(preferences.get_preferences())


def register() -> None:
    if _on_save_post not in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.append(_on_save_post)


def unregister() -> None:
    if _on_save_post in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.remove(_on_save_post)
