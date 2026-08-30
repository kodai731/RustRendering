from __future__ import annotations

import bpy

from . import debug_tools


class VIEW3D_PT_thyllore_water(bpy.types.Panel):

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Thyllore"
    bl_label = "Water"

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        obj = context.view_layer.objects.active
        if obj is None or not obj.thyllore_water.is_water:
            self.layout.operator("thyllore.water_add")
            return

        layout = self.layout
        props = obj.thyllore_water

        layout.prop(props, "preset")
        box = layout.box()
        for name in type(props).PARAM_NAMES:
            box.prop(props, name)

        debug_tools.draw_panel(layout)
