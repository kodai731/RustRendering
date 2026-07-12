"""Thin shim over the ``thyllore_ml_core`` wheel for license activation
(mode C only). The HTTPS transport and refusal policy live in the wheel —
the single source of truth shared with the engine. This module keeps only
the addon-local token cache.
"""
from __future__ import annotations

from .. import _debuglog
from .._token_store import TokenStore
from ..build_config import LICENSE_ENDPOINT

_store = TokenStore("thyllore_curve_copilot_license.json")


def resolve_full_token() -> str | None:
    return _store.resolve_full_token()


def discard_full_token() -> None:
    _store.discard_full_token()


def refresh_license(license_key: str, device_id: str) -> tuple[bool, str]:
    """Request a seat via the wheel and cache the token. Returns (ok, refusal_reason)."""
    import thyllore_ml_core as tml

    try:
        result = tml.refresh_license(LICENSE_ENDPOINT, license_key, device_id)
    except Exception:  # noqa: BLE001
        _debuglog.log_exception(f"refresh_license to {LICENSE_ENDPOINT} failed")
        return False, "network_error"

    if result.get("ok"):
        _store.store_full_token(result["full_token"], result["exp"])
        return True, ""

    if result.get("discard_cached_token"):
        discard_full_token()
    return False, str(result.get("reason", "refused"))
