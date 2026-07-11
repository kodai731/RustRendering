"""Cached unlock-token state shared by telemetry (mode B) and license_client
(mode C).

One JSON file per channel under Blender's user CONFIG directory. The wheel
re-verifies the Ed25519 signature and expiry itself; this store only caches
what the server issued, plus small channel-specific values (e.g. the
telemetry anon_id).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import bpy


class TokenStore:
    def __init__(self, filename: str):
        self._filename = filename
        self._cache: dict | None = None

    def _path(self) -> Path:
        config_dir = Path(bpy.utils.user_resource("CONFIG", create=True))
        return config_dir / self._filename

    def load(self) -> dict:
        if self._cache is not None:
            return self._cache
        try:
            self._cache = json.loads(self._path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._cache = {}
        return self._cache

    def save(self, state: dict) -> None:
        self._cache = state
        try:
            self._path().write_text(json.dumps(state), encoding="utf-8")
        except OSError:
            pass

    def resolve_unlock_token(self) -> str | None:
        """The cached unlock token if the server-issued expiry is still ahead."""
        state = self.load()
        token = state.get("unlock_token")
        exp = state.get("unlock_exp", 0)
        if not token or exp <= time.time():
            return None
        return token

    def store_unlock_token(self, token: str, exp: float) -> None:
        state = self.load()
        state["unlock_token"] = token
        state["unlock_exp"] = exp
        self.save(state)

    def discard_unlock_token(self) -> None:
        state = self.load()
        state.pop("unlock_token", None)
        state.pop("unlock_exp", None)
        self.save(state)
