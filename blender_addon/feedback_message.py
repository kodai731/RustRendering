"""Free-text feedback to the developers (/v1/message), available in every
build mode (A/B/C). Separate channel from the mode-B learning records: only
the user-written text is sent, always via an explicit button press, and
``bpy.app.online_access`` is respected. Transport lives in the
``thyllore_ml_core`` wheel.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import bpy
from bpy.types import Operator

from . import _debuglog
from ._token_store import TokenStore
from .capabilities import CAPS

_store = TokenStore("thyllore_curve_copilot_message.json")


def anon_id() -> str:
    """Random anonymous client id, unrelated to the license device_id."""
    state = _store.load()
    if "anon_id" not in state:
        state["anon_id"] = str(uuid.uuid4())
        _store.save(state)
    return state["anon_id"]


def send_message(text: str, addon_version: str) -> bool:
    import thyllore_ml_core as tml

    from .capabilities import FEEDBACK_ENDPOINT, INGEST_TOKEN

    try:
        tml.send_message(FEEDBACK_ENDPOINT, INGEST_TOKEN, anon_id(), text, addon_version)
        return True
    except Exception:  # noqa: BLE001
        _debuglog.log_exception(f"send_message to {FEEDBACK_ENDPOINT} failed")
        return False


class THYLLORE_OT_SendFeedback(Operator):
    bl_idname = "thyllore.send_feedback"
    bl_label = "Send Feedback"
    bl_description = "Send the feedback text to the Thyllore developers (anonymous)"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        try:
            return self._execute_checked()
        except Exception:
            _debuglog.log_exception("send_feedback failed unexpectedly")
            raise

    def _execute_checked(self) -> set:
        from . import preferences

        if not bpy.app.online_access:
            return self._cancel("Blender's 'Allow Online Access' is disabled")

        prefs = preferences.get_preferences()
        text = prefs.feedback_text.strip()
        if not text:
            return self._cancel("Feedback text is empty")

        if not send_message(text, _addon_version()):
            return self._cancel("Failed to send feedback (kept locally, try again later)")

        prefs.feedback_text = ""
        self.report({"INFO"}, "Feedback sent. Thank you!")
        return {"FINISHED"}

    def _cancel(self, reason: str) -> set:
        _debuglog.log_error(f"send_feedback: {reason}")
        self.report({"ERROR"}, reason)
        return {"CANCELLED"}


def _addon_version() -> str:
    """Version from blender_manifest.toml — the extension SSoT. Blender's
    extension loader strips ``bl_info`` from the module, so it cannot be used."""
    import tomllib

    manifest = Path(__file__).resolve().parent / "blender_manifest.toml"
    try:
        with manifest.open("rb") as f:
            return str(tomllib.load(f).get("version", "0.0.0"))
    except (OSError, tomllib.TOMLDecodeError):
        return "0.0.0"


_CLASSES = (THYLLORE_OT_SendFeedback,)


def register() -> None:
    if not CAPS.message_available:
        return
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    if not CAPS.message_available:
        return
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
