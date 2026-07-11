"""Free-text feedback operator (mode B only, /v1/message).

Separate channel from the learning pairs in ``records``: user-written
messages go to a different endpoint and storage prefix. Sending is always an
explicit button press and respects ``bpy.app.online_access``.
"""
from __future__ import annotations

import bpy
from bpy.types import Operator

from . import sender


class THYLLORE_OT_SendFeedback(Operator):
    bl_idname = "thyllore.send_feedback"
    bl_label = "Send Feedback"
    bl_description = "Send the feedback text to the Thyllore developers (anonymous)"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        from .. import preferences

        if not bpy.app.online_access:
            self.report({"ERROR"}, "Blender's 'Allow Online Access' is disabled")
            return {"CANCELLED"}

        prefs = preferences.get_preferences()
        text = prefs.feedback_text.strip()
        if not text:
            self.report({"ERROR"}, "Feedback text is empty")
            return {"CANCELLED"}

        addon_version = ".".join(str(v) for v in _addon_version())
        if not sender.send_message(text, addon_version):
            self.report({"ERROR"}, "Failed to send feedback (kept locally, try again later)")
            return {"CANCELLED"}

        prefs.feedback_text = ""
        self.report({"INFO"}, "Feedback sent. Thank you!")
        return {"FINISHED"}


def _addon_version() -> tuple:
    from .. import bl_info

    return bl_info.get("version", (0, 0, 0))


_CLASSES = (THYLLORE_OT_SendFeedback,)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
