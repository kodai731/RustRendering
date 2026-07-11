"""HTTPS transport for Curve Copilot feedback (mode B only).

Sends anonymized feedback batches to the Cloudflare Worker with stdlib
``urllib`` (no extra dependencies), and stores the short-lived Ed25519
``unlock_token`` returned by the Worker. The wheel re-verifies the token
signature and expiry itself, so this module only caches what the server
issued.
"""
from __future__ import annotations

import gzip
import json
import time
import uuid
from pathlib import Path

import bpy

from ..build_config import FEEDBACK_ENDPOINT, INGEST_TOKEN

MESSAGE_ENDPOINT = FEEDBACK_ENDPOINT.rsplit("/", 1)[0] + "/message"
SCHEMA_VERSION = "curve_copilot_feedback/v0"
REQUEST_TIMEOUT_SECONDS = 10.0

_state_cache: dict | None = None


def _state_path() -> Path:
    config_dir = Path(bpy.utils.user_resource("CONFIG", create=True))
    return config_dir / "thyllore_curve_copilot_feedback.json"


def _load_state() -> dict:
    global _state_cache
    if _state_cache is not None:
        return _state_cache
    try:
        _state_cache = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _state_cache = {}
    return _state_cache


def _save_state(state: dict) -> None:
    global _state_cache
    _state_cache = state
    try:
        _state_path().write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def anon_id() -> str:
    """Random anonymous client id, unrelated to the license device_id."""
    state = _load_state()
    if "anon_id" not in state:
        state["anon_id"] = str(uuid.uuid4())
        _save_state(state)
    return state["anon_id"]


def resolve_unlock_token() -> str | None:
    """The cached unlock token if the server-issued expiry is still ahead."""
    state = _load_state()
    token = state.get("unlock_token")
    exp = state.get("unlock_exp", 0)
    if not token or exp <= time.time():
        return None
    return token


def discard_unlock_token() -> None:
    """Opt-out immediately reverts to ctx32: drop the cached token."""
    state = _load_state()
    state.pop("unlock_token", None)
    state.pop("unlock_exp", None)
    _save_state(state)


def should_send(prefs) -> bool:
    return bool(getattr(prefs, "telemetry_opt_in", False)) and bpy.app.online_access


def send_feedback_batch(records: list[dict]) -> bool:
    """POST one gzip JSONL batch; caches the refreshed unlock token.

    An empty batch is a valid token-refresh handshake. Returns False on any
    network / server error (callers keep their records and retry later).
    """
    import urllib.error
    import urllib.request

    lines = "\n".join(json.dumps(record) for record in records)
    body = gzip.compress(lines.encode("utf-8"))
    request = urllib.request.Request(
        FEEDBACK_ENDPOINT,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {INGEST_TOKEN}",
            "Content-Type": "application/x-ndjson",
            "Content-Encoding": "gzip",
            "X-Schema-Version": SCHEMA_VERSION,
            "X-Anon-Id": anon_id(),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False

    token = payload.get("unlock_token")
    exp = payload.get("exp")
    if isinstance(token, str) and isinstance(exp, (int, float)):
        state = _load_state()
        state["unlock_token"] = token
        state["unlock_exp"] = exp
        _save_state(state)
    return True


def send_message(text: str, addon_version: str) -> bool:
    """POST one free-text feedback message to the Worker (/v1/message)."""
    import urllib.error
    import urllib.request

    body = json.dumps(
        {
            "text": text,
            "addon_version": addon_version,
            "anon_id": anon_id(),
            "ts": int(time.time()),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        MESSAGE_ENDPOINT,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {INGEST_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
