from __future__ import annotations

from . import _bootstrap
from . import properties


def register():
    import bpy

    from . import operators
    from . import panels

    _bootstrap.insert_wheels_to_sys_path()

    cls = properties.build_flame_property_group()
    properties._registered_cls = cls
    bpy.utils.register_class(cls)
    bpy.types.Object.thyllore_flame = bpy.props.PointerProperty(type=cls)

    bpy.utils.register_class(operators.THYLLORE_OT_flame_add)
    bpy.utils.register_class(panels.VIEW3D_PT_thyllore_flame)


def unregister():
    import bpy

    from . import operators
    from . import panels

    bpy.utils.unregister_class(panels.VIEW3D_PT_thyllore_flame)
    bpy.utils.unregister_class(operators.THYLLORE_OT_flame_add)

    del bpy.types.Object.thyllore_flame

    if hasattr(properties, "_registered_cls"):
        bpy.utils.unregister_class(properties._registered_cls)
        del properties._registered_cls

    _bootstrap.remove_wheels_from_sys_path()
