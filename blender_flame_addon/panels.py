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

        layout.prop(props, "preset")

        import thyllore_effect_core as fx

        ui_params = fx.flame_ui_params()
        param_owners: dict[str, list[dict]] = {}
        for p in ui_params:
            owner = p["owner"]
            if owner not in param_owners:
                param_owners[owner] = []
            param_owners[owner].append(p)

        for owner in ("frame", "shape", "style"):
            if owner not in param_owners:
                continue
            box = layout.box()
            box.label(text=owner.title())
            for p in param_owners[owner]:
                name = p["name"]
                kind = _property_kind(p["default"])
                if kind == "vector":
                    box.prop(props, name)
                else:
                    box.prop(props, name)


def _property_kind(default) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, (list, tuple)):
        return "vector"
    return "float"
