"""Transport shim for Curve Copilot feedback (mode B only).

The HTTPS transport (gzip JSONL, headers, endpoints) lives in the
``thyllore_ml_core`` wheel — the single source of truth shared with the
engine. This module keeps only the bpy-dependent parts: the opt-in check,
the anonymous client id and the cached Ed25519 ``full_token`` returned by
the Worker (the wheel re-verifies the token signature and expiry itself).
"""
from __future__ import annotations

import uuid

import bpy

from .. import _debuglog
from .._token_store import TokenStore
from ..build_config import FEEDBACK_ENDPOINT, INGEST_TOKEN

_store = TokenStore("thyllore_curve_copilot_feedback.json")


def anon_id() -> str:
    """Random anonymous client id, unrelated to the license device_id."""
    state = _store.load()
    if "anon_id" not in state:
        state["anon_id"] = str(uuid.uuid4())
        _store.save(state)
    return state["anon_id"]


def resolve_full_token() -> str | None:
    """ctx64 is granted in exchange for sending; offline sessions degrade."""
    if not bpy.app.online_access:
        return None
    return _store.resolve_full_token()


def discard_full_token() -> None:
    """Opt-out immediately reverts to ctx32: drop the cached token."""
    _store.discard_full_token()


def should_send(prefs) -> bool:
    return bool(getattr(prefs, "telemetry_opt_in", False)) and bpy.app.online_access


def send_feedback_batch(records: list[dict]) -> bool:
    """POST one gzip JSONL batch via the wheel; caches the refreshed token.

    An empty batch is a valid token-refresh handshake. Returns False on any
    network / server error (callers keep their records and retry later).
    """
    import thyllore_ml_core as tml

    try:
        payload = tml.send_feedback_batch(
            FEEDBACK_ENDPOINT, INGEST_TOKEN, anon_id(), records
        )
    except Exception:  # noqa: BLE001
        _debuglog.log_exception(f"send_feedback_batch to {FEEDBACK_ENDPOINT} failed")
        return False

    token = payload.get("full_token")
    exp = payload.get("exp")
    if isinstance(token, str) and isinstance(exp, (int, float)):
        _store.store_full_token(token, exp)
    return True
