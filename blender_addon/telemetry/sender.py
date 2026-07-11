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

import bpy

from .._token_store import TokenStore
from ..build_config import FEEDBACK_ENDPOINT, INGEST_TOKEN

MESSAGE_ENDPOINT = FEEDBACK_ENDPOINT.rsplit("/", 1)[0] + "/message"
SCHEMA_VERSION = "curve_copilot_feedback/v0"
REQUEST_TIMEOUT_SECONDS = 10.0
USER_AGENT = "ThylloreCurveCopilot/1.0"

_store = TokenStore("thyllore_curve_copilot_feedback.json")


def anon_id() -> str:
    """Random anonymous client id, unrelated to the license device_id."""
    state = _store.load()
    if "anon_id" not in state:
        state["anon_id"] = str(uuid.uuid4())
        _store.save(state)
    return state["anon_id"]


def resolve_unlock_token() -> str | None:
    return _store.resolve_unlock_token()


def discard_unlock_token() -> None:
    """Opt-out immediately reverts to ctx32: drop the cached token."""
    _store.discard_unlock_token()


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
            "User-Agent": USER_AGENT,
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
        _store.store_unlock_token(token, exp)
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
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
