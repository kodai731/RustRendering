from __future__ import annotations

import bpy


class THYLLORE_OT_water_add(bpy.types.Operator):

    bl_idname = "thyllore.water_add"
    bl_label = "Add Water"

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        cursor_location = context.scene.cursor.location

        obj = bpy.data.objects.new("Water", None)
        obj.empty_display_type = "CUBE"
        obj.empty_display_size = 0.5
        obj.location = cursor_location

        context.collection.objects.link(obj)

        obj.thyllore_water.is_water = True
        obj.thyllore_water.preset = "default"

        context.view_layer.objects.active = obj
        obj.select_set(True)

        return {"FINISHED"}
