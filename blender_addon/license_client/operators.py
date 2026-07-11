"""License activation operator (mode C only).

Activation is always an explicit button press: no background network calls
are made on preference edits, and ``bpy.app.online_access`` is respected.
"""
from __future__ import annotations

import bpy
from bpy.types import Operator

from . import client

REFUSAL_MESSAGES = {
    "unknown_license": "License key is not recognized",
    "revoked": "This license has been revoked",
    "seat_exhausted": "All seats for this license are in use (copied key?)",
    "network_error": "Could not reach the license server (kept previous state)",
}


class THYLLORE_OT_RefreshLicense(Operator):
    bl_idname = "thyllore.refresh_license"
    bl_label = "Activate License"
    bl_description = "Verify the license key online and unlock ctx64 prediction"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        from .. import preferences

        if not bpy.app.online_access:
            self.report({"ERROR"}, "Blender's 'Allow Online Access' is disabled")
            return {"CANCELLED"}

        prefs = preferences.get_preferences()
        license_key = prefs.license_key.strip()
        if not license_key:
            self.report({"ERROR"}, "License key is empty")
            return {"CANCELLED"}

        ok, reason = client.refresh_license(license_key, prefs.device_id)
        if not ok:
            self.report({"ERROR"}, REFUSAL_MESSAGES.get(reason, f"Activation failed ({reason})"))
            return {"CANCELLED"}

        self.report({"INFO"}, "License activated: ctx64 prediction unlocked")
        return {"FINISHED"}


_CLASSES = (THYLLORE_OT_RefreshLicense,)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
