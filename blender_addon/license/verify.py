"""Ed25519-based offline license verification.

Token format: ``<json_base64url>.<signature_base64url>`` where the JSON payload
is the structure documented in Phase4_PackagingAndLicense.md (license_id /
user_email / device_id / issued_at / expires_at / tier / features).
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False


@dataclass(frozen=True)
class LicenseStatus:
    """Outcome of a license verification attempt.

    The shape of this dataclass is part of the L4 stability contract: Phase 6
    will add fields (e.g. ``server_validated_at``) but never remove them, and
    Operators that consume only ``is_valid`` / ``error_message`` keep working.
    """

    is_valid: bool
    error_message: str = ""
    user_email: str = ""
    tier: str = ""
    expires_at: Optional[datetime] = None


_PUBLIC_KEY_PATH = Path(__file__).resolve().parent / "public_key.pem"


def verify_license_token(token: str) -> LicenseStatus:
    """Verify a license token against the bundled Ed25519 public key.

    Returns a :class:`LicenseStatus` with ``is_valid=False`` for any failure
    rather than raising, so the addon's ``register()`` can decide UX without
    catching exceptions.
    """
    if not _HAS_CRYPTOGRAPHY:
        return LicenseStatus(
            is_valid=False,
            error_message=(
                "cryptography library not available -- install via Preferences "
                "or rebuild the addon ZIP with the cryptography wheel bundled"
            ),
        )

    if not token or "." not in token:
        return LicenseStatus(is_valid=False, error_message="Empty or malformed token")

    try:
        json_b64, sig_b64 = token.split(".", 1)
    except ValueError:
        return LicenseStatus(
            is_valid=False, error_message="Token format invalid (expected <json>.<sig>)"
        )

    try:
        json_bytes = base64.urlsafe_b64decode(_pad_b64(json_b64))
        signature = base64.urlsafe_b64decode(_pad_b64(sig_b64))
    except Exception as e:  # noqa: BLE001 -- base64 may raise binascii.Error or others
        return LicenseStatus(is_valid=False, error_message=f"Base64 decode failed: {e}")

    if not _PUBLIC_KEY_PATH.exists():
        return LicenseStatus(
            is_valid=False, error_message="Public key not found in addon"
        )

    try:
        with open(_PUBLIC_KEY_PATH, "rb") as f:
            pub_key = serialization.load_pem_public_key(f.read())
    except Exception as e:  # noqa: BLE001
        return LicenseStatus(
            is_valid=False, error_message=f"Public key load failed: {e}"
        )
    if not isinstance(pub_key, Ed25519PublicKey):
        return LicenseStatus(
            is_valid=False, error_message="Public key is not Ed25519"
        )

    try:
        pub_key.verify(signature, json_bytes)
    except InvalidSignature:
        return LicenseStatus(
            is_valid=False, error_message="Signature verification failed"
        )
    except Exception as e:  # noqa: BLE001
        return LicenseStatus(
            is_valid=False, error_message=f"Signature check raised: {e}"
        )

    try:
        payload = json.loads(json_bytes.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return LicenseStatus(is_valid=False, error_message=f"Payload not JSON: {e}")

    expires_at_str = payload.get("expires_at")
    if not expires_at_str:
        return LicenseStatus(
            is_valid=False, error_message="Token missing expires_at"
        )

    try:
        expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
    except Exception as e:  # noqa: BLE001
        return LicenseStatus(
            is_valid=False, error_message=f"expires_at parse error: {e}"
        )

    now = datetime.now(timezone.utc)
    if expires_at < now:
        return LicenseStatus(
            is_valid=False,
            error_message=f"License expired at {expires_at.isoformat()}",
            expires_at=expires_at,
        )

    return LicenseStatus(
        is_valid=True,
        user_email=payload.get("user_email", ""),
        tier=payload.get("tier", "free"),
        expires_at=expires_at,
    )


_HEADLESS_ENV_VAR = "THYLLORE_HEADLESS"
_TEST_BYPASS_ENV_VAR = "THYLLORE_TEST_BYPASS_LICENSE"


def verify_license_from_preferences() -> LicenseStatus:
    """Read ``license_key`` from AddonPreferences and verify it.

    The environment variable ``THYLLORE_LICENSE`` overrides the Preferences
    value, which is convenient for headless smoke tests.

    For automated parity tests that must skip licensing entirely, setting both
    ``THYLLORE_HEADLESS=1`` and ``THYLLORE_TEST_BYPASS_LICENSE=1`` returns a
    synthetic-valid status. Both flags are required so production users
    (who do not run with HEADLESS) can never trigger the bypass.
    """
    if (
        os.environ.get(_HEADLESS_ENV_VAR) == "1"
        and os.environ.get(_TEST_BYPASS_ENV_VAR) == "1"
    ):
        return LicenseStatus(
            is_valid=True,
            user_email="test@bypass.local",
            tier="test",
        )

    license_key = ""
    try:
        import bpy  # type: ignore

        from .. import preferences as prefs_module

        prefs = prefs_module.get_preferences()
        license_key = prefs.license_key or ""
    except Exception:  # noqa: BLE001 -- bpy unavailable / addon not registered
        pass

    license_key = os.environ.get("THYLLORE_LICENSE", license_key)
    return verify_license_token(license_key)


def _pad_b64(s: str) -> str:
    return s + "=" * (-len(s) % 4)
