from __future__ import annotations

import bpy

from . import debug_tools
from ._common import effect_properties


class VIEW3D_PT_thyllore_flame(bpy.types.Panel):

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Thyllore"
    bl_label = "Flame"

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        obj = context.view_layer.objects.active
        if obj is None or not obj.thyllore_flame.is_flame:
            self.layout.operator("thyllore.flame_add")
            return

        layout = self.layout
        props = obj.thyllore_flame

        layout.prop(props, "preset")
        effect_properties.draw_param_groups(layout, props)

        debug_tools.draw_panel(layout)
