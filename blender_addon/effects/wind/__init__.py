from __future__ import annotations

from . import _bootstrap
from . import properties


def register():
    import bpy

    from . import debug_tools
    from . import draw_handler
    from . import operators
    from . import panels

    _bootstrap.insert_wheels_to_sys_path()

    cls = properties.build_wind_property_group()
    properties._registered_cls = cls
    bpy.utils.register_class(cls)
    bpy.types.Object.thyllore_wind = bpy.props.PointerProperty(type=cls)

    bpy.utils.register_class(operators.THYLLORE_OT_wind_add)
    debug_tools.register()
    bpy.utils.register_class(panels.VIEW3D_PT_thyllore_wind)

    draw_handler.register_draw_handler()


def unregister():
    import bpy

    from . import debug_tools
    from . import draw_handler
    from . import operators
    from . import panels

    draw_handler.unregister_draw_handler()
    bpy.utils.unregister_class(panels.VIEW3D_PT_thyllore_wind)
    debug_tools.unregister()
    bpy.utils.unregister_class(operators.THYLLORE_OT_wind_add)

    del bpy.types.Object.thyllore_wind

    if hasattr(properties, "_registered_cls"):
        bpy.utils.unregister_class(properties._registered_cls)
        del properties._registered_cls

    _bootstrap.remove_wheels_from_sys_path()
