"""Curve Copilot license activation (bundled in mode C builds only).

Public surface used by the rest of the addon:
- ``resolve_unlock_token`` / ``discard_unlock_token`` -- gating helpers
- ``refresh_license`` -- explicit online activation
- ``register`` / ``unregister`` -- operator lifecycle + startup seat renewal
"""
from __future__ import annotations

import threading

import bpy

from . import client, operators
from .client import discard_unlock_token, refresh_license, resolve_unlock_token


def _renew_seat_in_background() -> None:
    """Keep an activated seat's ctx64 alive across sessions without UI action.

    Preferences are read on the main thread; only the network call runs in a
    daemon thread. Refusals inside refresh_license discard the cached token.
    """
    from .. import preferences

    if not bpy.app.online_access:
        return
    prefs = preferences.get_preferences()
    license_key = prefs.license_key.strip()
    if not license_key:
        return
    threading.Thread(
        target=client.refresh_license, args=(license_key, prefs.device_id), daemon=True
    ).start()


def register() -> None:
    operators.register()
    _renew_seat_in_background()


def unregister() -> None:
    operators.unregister()
