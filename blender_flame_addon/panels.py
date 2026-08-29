from __future__ import annotations

import bpy


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

        layout.operator("thyllore.flame_setup_compositor")
        layout.operator("thyllore.flame_render_sequence")
        layout.operator("thyllore.flame_dump_viewport")
        layout.prop(props, "preset")

        cls = type(props)
        param_owners = cls.PARAM_OWNERS

        for owner in ("frame", "shape", "style"):
            if owner not in param_owners:
                continue
            box = layout.box()
            box.label(text=owner.title())
            for name in param_owners[owner]:
                box.prop(props, name)
