"""Thyllore Animation Blender add-on entry point.

Layer responsibility (see Phase4_AddonRegistration.md and Phase4_FourLayerStability.md):
- This file performs the L4 -> L3 boundary check (``__abi_marker__``) and
  orchestrates registration of preferences / operators / panels.
- All actual operator logic lives in ``operators/``; all panel UI in ``panels/``.
- Tier A operators (text_to_mesh / auto_rig) are registered only when their
  modules are present in the installed ZIP -- the ``-Variant lite`` MVP build
  excludes them, so ``operators.has_tier_a()`` returns False at runtime and
  panels skip their UI.
- Phase 5.5: license verification removed (MVP is unauthenticated). Phase 6
  will reintroduce it as an Auth Backend round-trip in a fresh module.
"""
from __future__ import annotations

bl_info = {
    "name": "Thyllore Animation",
    "author": "kodai731",
    "version": (0, 0, 1),
    "blender": (4, 2, 0),
    "location": "View3D > N-Panel > Thyllore Animation",
    "description": "AI-driven animation toolkit (Curve Copilot / Text-to-Motion)",
    "category": "Animation",
    "support": "COMMUNITY",
}

# CRITICAL: must come before any grpc / thyllore_ml_core import.
from . import _bootstrap  # noqa: E402

_bootstrap.insert_wheels_to_sys_path()


# Expected ABI marker -- must match crates/thyllore-ml-api/src/lib.rs::ABI_MARKER.
# Bump in lockstep with that constant when the L2 trait has breaking changes.
EXPECTED_ABI_MARKER: int = 2


# Skip importing bpy-dependent modules during pure-package consumption (e.g.,
# pytest running blender_addon/tests/* without Blender). This preserves the
# Phase 3 ability to ``from blender_addon.grpc_client import AutoRigClient``
# from outside Blender.
try:
    import bpy  # type: ignore  # noqa: F401

    _BPY_AVAILABLE = True
except ImportError:
    _BPY_AVAILABLE = False


if _BPY_AVAILABLE:
    import os  # noqa: E402

    from . import operators  # noqa: E402
    from . import panels  # noqa: E402
    from . import preferences  # noqa: E402

    # Telemetry ships only in mode B builds; A/C ZIPs exclude the package.
    try:
        from . import telemetry  # noqa: E402

        _TELEMETRY_AVAILABLE = True
    except ImportError:
        _TELEMETRY_AVAILABLE = False

    _REGISTERED_ABI_FAILURE: bool = False

    FORCE_MOCK_SERVER_ENV_VAR = "THYLLORE_FORCE_MOCK_SERVER"
    MOCK_SERVER_HOST = "127.0.0.1"
    MOCK_SERVER_PORT = 50051

    def _apply_mock_server_pin_if_requested() -> None:
        """Pin server_host / server_port / use_tls to the local mock server.

        Triggered by ``THYLLORE_FORCE_MOCK_SERVER=1`` for the parity test
        orchestrator. The override lives only in the in-memory
        ``AddonPreferences`` instance for the duration of this Blender process.
        Skipped silently when the lite Variant strips the gRPC preference
        fields.
        """
        if os.environ.get(FORCE_MOCK_SERVER_ENV_VAR) != "1":
            return
        prefs = preferences.get_preferences()
        if prefs is None:
            return
        if not hasattr(prefs, "server_host"):
            return
        prefs.server_host = MOCK_SERVER_HOST
        prefs.server_port = MOCK_SERVER_PORT
        prefs.use_tls = False
        print(
            f"[Thyllore] mock server pin active: "
            f"server pinned to {MOCK_SERVER_HOST}:{MOCK_SERVER_PORT}"
        )

    def _check_wheel_abi() -> tuple[bool, str]:
        """Verify the thyllore_ml_core wheel's ``__abi_marker__``.

        Wheel absence is NOT treated as ABI failure -- the curve_copilot
        operator will simply gray-out via tml.capabilities() while Tier A
        operators continue to work in the full Variant.
        """
        try:
            import thyllore_ml_core as tml  # type: ignore
        except ImportError:
            return True, ""

        actual = getattr(tml, "__abi_marker__", 0)
        if actual != EXPECTED_ABI_MARKER:
            return (
                False,
                f"thyllore_ml_core ABI mismatch: addon expects "
                f"{EXPECTED_ABI_MARKER}, wheel provides {actual}. "
                f"Please update the addon.",
            )
        return True, ""

    def register() -> None:
        global _REGISTERED_ABI_FAILURE

        abi_ok, abi_err = _check_wheel_abi()
        if not abi_ok:
            preferences.register()
            panels.register_placeholder(error_message=abi_err)
            _REGISTERED_ABI_FAILURE = True
            print(f"[Thyllore] {abi_err}")
            return

        preferences.register()
        operators.register()
        panels.register()
        if _TELEMETRY_AVAILABLE:
            telemetry.register()
        _apply_mock_server_pin_if_requested()
        _REGISTERED_ABI_FAILURE = False
        print("[Thyllore] Addon registered successfully")

    def unregister() -> None:
        global _REGISTERED_ABI_FAILURE

        if _TELEMETRY_AVAILABLE and not _REGISTERED_ABI_FAILURE:
            telemetry.unregister()
        panels.unregister()
        if not _REGISTERED_ABI_FAILURE:
            operators.unregister()
        preferences.unregister()
        _REGISTERED_ABI_FAILURE = False
