from __future__ import annotations

import bpy


class THYLLORE_OT_wind_add(bpy.types.Operator):

    bl_idname = "thyllore.wind_add"
    bl_label = "Add Wind"

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        import thyllore_effect_core as fx

        cursor_location = context.scene.cursor.location

        obj = bpy.data.objects.new("Wind", None)
        obj.empty_display_type = "CUBE"
        obj.empty_display_size = 0.5
        obj.location = cursor_location

        context.collection.objects.link(obj)

        obj.thyllore_wind.is_wind = True
        obj.thyllore_wind.preset = fx.wind_preset_names()[0]

        context.view_layer.objects.active = obj
        obj.select_set(True)

        return {"FINISHED"}
