"""Addon preferences UI and accessor.

Phase 5.5 MVP variant: the lite ZIP excludes Tier A operators and the
``grpc_client`` package, so the gRPC / license fields below are present in the
preference data model but their UI is gated on
``operators.has_tier_a()`` -- they only appear when the full Variant bundles
the Tier A operators that consume them. This keeps the same
``AddonPreferences`` class binary-compatible across Variants and lets Phase 6
(Auth Backend reintroduction) repopulate the existing fields without a
migration step.
"""
from __future__ import annotations

import uuid

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import AddonPreferences, Operator

from . import _full_token
from .capabilities import CAPS

try:
    from . import telemetry
except ImportError:
    telemetry = None

try:
    from . import license_client
except ImportError:
    license_client = None

DISCORD_INVITE_URL = "https://discord.gg/HZNRAENX8"

# Resolve the addon's package name so AddonPreferences.bl_idname matches the
# top-level extension id (legacy install: "blender_addon"; lite extension:
# "thyllore_animation_lite"; full extension: "thyllore_animation").
ADDON_PACKAGE = __package__ or "blender_addon"


class ThylloreAnimationPreferences(AddonPreferences):
    bl_idname = ADDON_PACKAGE

    server_host: StringProperty(  # type: ignore[valid-type]
        name="Server Host",
        default="127.0.0.1",
        description="gRPC server hostname or IP (full Variant only)",
    )
    server_port: IntProperty(  # type: ignore[valid-type]
        name="Server Port",
        default=50051,
        min=1,
        max=65535,
    )
    use_tls: BoolProperty(  # type: ignore[valid-type]
        name="Use TLS",
        default=False,
        description="Enable TLS for gRPC connection (required for Thyllore Cloud)",
    )
    deadline_seconds: FloatProperty(  # type: ignore[valid-type]
        name="Deadline (s)",
        default=120.0,
        min=1.0,
        max=600.0,
    )

    def _on_license_key_changed(self, context):
        if license_client is not None and not self.license_key.strip():
            license_client.discard_full_token()

    license_key: StringProperty(  # type: ignore[valid-type]
        name="License Key",
        default="",
        subtype="PASSWORD",
        update=_on_license_key_changed,
        description="License key from your purchase (activates ctx64 prediction)",
    )
    device_id: StringProperty(  # type: ignore[valid-type]
        name="Device ID",
        default="",
        description="Unique device identifier (auto-generated on first install)",
    )

    enable_curve_copilot: BoolProperty(  # type: ignore[valid-type]
        name="Enable Curve Copilot",
        default=True,
        description="Enable in-process ONNX inference for FCurve suggestions (Tier B)",
    )
    curve_copilot_model_path: StringProperty(  # type: ignore[valid-type]
        name="Curve Copilot Model Path",
        default="",
        subtype="FILE_PATH",
        description="Path to curve_copilot.onnx (leave empty to use bundled model)",
    )

    text_to_motion_model_dir: StringProperty(  # type: ignore[valid-type]
        name="Text-to-Motion Cache Dir",
        default="",
        subtype="DIR_PATH",
        description="Directory for the lazily-downloaded light_t2m.onnx (leave empty for ~/.cache/thyllore/light_t2m/)",
    )

    def _on_telemetry_opt_in_changed(self, context):
        if telemetry is None:
            return
        if self.telemetry_opt_in:
            if bpy.app.online_access:
                telemetry.request_token_refresh()
        else:
            telemetry.discard_full_token()

    telemetry_opt_in: BoolProperty(  # type: ignore[valid-type]
        name="I consent to sending Curve Copilot feedback data",
        default=False,
        update=_on_telemetry_opt_in_changed,
        description=(
            "Enables ctx64 high-accuracy prediction. In exchange, anonymized "
            "feedback (the curve the model predicted and the keyframe values "
            "you actually entered) is sent periodically in the background. "
            "No file names, object names, bone names, scene contents or "
            "personal data are ever collected. Used solely to improve the "
            "Curve Copilot model. Turn off anytime: sending stops immediately "
            "and prediction reverts to ctx32"
        ),
    )
    feedback_text: StringProperty(  # type: ignore[valid-type]
        name="Feedback",
        default="",
        description=(
            "Free-form feedback for the developers. "
            "Please do not include personal or confidential information"
        ),
    )

    def draw(self, context):
        from . import operators

        layout = self.layout

        box = layout.box()
        box.label(text="Curve Copilot (Tier B - PyO3)", icon="FCURVE")
        box.prop(self, "enable_curve_copilot")
        box.prop(self, "curve_copilot_model_path")
        effective_ctx = _effective_context_length()
        if effective_ctx is not None:
            box.label(text=f"Prediction accuracy: ctx {effective_ctx}", icon="INFO")

        if CAPS.telemetry_available:
            self._draw_telemetry_box(layout)
        if CAPS.license_activation:
            self._draw_license_box(layout)
        if CAPS.message_available:
            self._draw_message_box(layout)
        if DISCORD_INVITE_URL:
            layout.operator(
                "wm.url_open", text="Join the community (Discord)", icon="URL"
            ).url = DISCORD_INVITE_URL

        if operators.has_text_to_motion():
            box = layout.box()
            box.label(text="Text to Motion (Tier B - PyO3)", icon="POSE_HLT")
            box.prop(self, "text_to_motion_model_dir")
            box.label(text="Models are downloaded from HuggingFace on first use.")

        if operators.has_tier_a():
            box = layout.box()
            box.label(text="Server (Tier A - gRPC)", icon="URL")
            box.prop(self, "server_host")
            box.prop(self, "server_port")
            box.prop(self, "use_tls")
            box.prop(self, "deadline_seconds")

            box = layout.box()
            box.label(text="License", icon="LOCKED")
            box.prop(self, "license_key")
            row = box.row()
            row.label(text=f"Device: {self.device_id or '(not set)'}")
            row.operator("thyllore.regenerate_device_id", text="Regenerate")
        else:
            layout.label(
                text="Tier A cloud features (mesh / auto-rig) require Phase 6 SaaS upgrade.",
                icon="INFO",
            )

    def _draw_telemetry_box(self, layout):
        box = layout.box()
        box.label(text="Curve Copilot Feedback (enables ctx64)", icon="EXPERIMENTAL")
        box.prop(self, "telemetry_opt_in")

        col = box.column()
        col.scale_y = 0.8
        col.label(text="Sent: only the predicted curve and the keyframe values you enter.")
        col.label(text="Never sent: file / object / bone names, scene contents, personal data.")
        col.label(text="Used solely to improve the Curve Copilot model (license-clean training).")
        col.label(text="Requires 'Allow Online Access'. Turning off reverts to ctx32 immediately.")

    def _draw_message_box(self, layout):
        box = layout.box()
        box.label(text="Feedback to the Developers", icon="OUTLINER_OB_SPEAKER")
        row = box.row(align=True)
        row.prop(self, "feedback_text", text="")
        row.operator("thyllore.send_feedback", text="Send Feedback")

    def _draw_license_box(self, layout):
        box = layout.box()
        box.label(text="Curve Copilot License (enables ctx64)", icon="LOCKED")
        box.prop(self, "license_key")

        row = box.row()
        row.label(text=f"Device: {self.device_id or '(not set)'}")
        row.operator("thyllore.refresh_license", text="Activate License")

        col = box.column()
        col.scale_y = 0.8
        col.label(text="Activation verifies your seat online; no feedback data is sent.")
        col.label(text="Requires 'Allow Online Access'. Re-activation runs weekly per device.")


class THYLLORE_OT_RegenerateDeviceID(Operator):
    bl_idname = "thyllore.regenerate_device_id"
    bl_label = "Regenerate Device ID"
    bl_description = "Generate a new device identifier (forces re-authentication)"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        prefs = get_preferences()
        prefs.device_id = str(uuid.uuid4())
        self.report({"INFO"}, f"Device ID regenerated: {prefs.device_id}")
        return {"FINISHED"}


_CLASSES = (ThylloreAnimationPreferences, THYLLORE_OT_RegenerateDeviceID)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    prefs = get_preferences()
    if not prefs.device_id:
        prefs.device_id = str(uuid.uuid4())


def unregister() -> None:
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


def _effective_context_length() -> int | None:
    """The wheel-decided context window (read-only display; addon holds no 32/64)."""
    try:
        import thyllore_ml_core as tml
    except ImportError:
        return None

    try:
        return int(tml.effective_context_length(_full_token.resolve_full_token()))
    except AttributeError:
        return None


def get_preferences() -> ThylloreAnimationPreferences:
    """Return the addon's AddonPreferences instance.

    Operators use this to read server/license configuration without binding
    to any specific Blender layout.
    """
    return bpy.context.preferences.addons[ADDON_PACKAGE].preferences
