from __future__ import annotations

OPERATOR_IDS = (
    "thyllore.flame_setup_compositor",
    "thyllore.flame_render_sequence",
    "thyllore.flame_dump_viewport",
)


def operator_classes():
    from . import operators

    return (
        operators.THYLLORE_OT_flame_render_sequence,
        operators.THYLLORE_OT_flame_setup_compositor,
        operators.THYLLORE_OT_flame_dump_viewport,
    )


def register():
    import bpy

    for cls in operator_classes():
        bpy.utils.register_class(cls)


def unregister():
    import bpy

    for cls in reversed(operator_classes()):
        bpy.utils.unregister_class(cls)


def draw_panel(layout):
    box = layout.box()
    box.label(text="Debug")
    for operator_id in OPERATOR_IDS:
        box.operator(operator_id)
