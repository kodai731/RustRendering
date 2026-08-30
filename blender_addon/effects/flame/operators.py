from __future__ import annotations

import bpy

from .properties import apply_initial_values


class THYLLORE_OT_flame_add(bpy.types.Operator):

    bl_idname = "thyllore.flame_add"
    bl_label = "Add Flame"

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        cursor_location = context.scene.cursor.location

        obj = bpy.data.objects.new("Flame", None)
        obj.empty_display_type = "CUBE"
        obj.empty_display_size = 0.5
        obj.location = cursor_location

        context.collection.objects.link(obj)

        obj.thyllore_flame.is_flame = True
        obj.thyllore_flame.preset = "campfire"
        apply_initial_values(obj.thyllore_flame)

        context.view_layer.objects.active = obj
        obj.select_set(True)

        return {"FINISHED"}
