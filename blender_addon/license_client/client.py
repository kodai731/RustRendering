"""Online license activation for Curve Copilot (mode C only).

POSTs ``{license_key, device_id}`` to the Worker's ``/v1/license/refresh``.
While the LicenseSeats Durable Object grants a seat, the response carries a
short-lived Ed25519 unlock_token that the wheel verifies itself, so ctx64
stays unlocked between refreshes (offline grace = token expiry). An explicit
refusal (copied key exhausting seats, revoked license) discards the cached
token immediately; network failures keep it until expiry.
"""
from __future__ import annotations

import json

from .._token_store import TokenStore
from ..build_config import LICENSE_ENDPOINT

REQUEST_TIMEOUT_SECONDS = 10.0
USER_AGENT = "ThylloreCurveCopilot/1.0"

_store = TokenStore("thyllore_curve_copilot_license.json")


def resolve_unlock_token() -> str | None:
    return _store.resolve_unlock_token()


def discard_unlock_token() -> None:
    _store.discard_unlock_token()


def refresh_license(license_key: str, device_id: str) -> tuple[bool, str]:
    """Request a seat and cache the unlock token. Returns (ok, refusal_reason)."""
    import urllib.error
    import urllib.request

    body = json.dumps({"license_key": license_key, "device_id": device_id}).encode("utf-8")
    request = urllib.request.Request(
        LICENSE_ENDPOINT,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 403:
            discard_unlock_token()
            return False, _refusal_reason(error)
        return False, f"http_{error.code}"
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False, "network_error"

    token = payload.get("unlock_token")
    exp = payload.get("exp")
    if not isinstance(token, str) or not isinstance(exp, (int, float)):
        return False, "invalid_response"
    _store.store_unlock_token(token, exp)
    return True, ""


def _refusal_reason(error) -> str:
    try:
        return str(json.loads(error.read().decode("utf-8")).get("error", "refused"))
    except (ValueError, OSError):
        return "refused"
