"""License verification for the Thyllore Animation addon.

Phase 4 ships only offline Ed25519 verification. Phase 6 will add
``auth_backend_validate`` for Polar.sh-issued JWT tokens; the
:class:`verify.LicenseStatus` dataclass and :func:`verify.verify_license_token`
signature are designed to remain stable across that transition.
"""
from . import verify  # noqa: F401

__all__ = ["verify"]
