from __future__ import annotations


def precision_from_format(fmt: str) -> int:
    if "d" in fmt:
        return 0
    if "." in fmt and "f" in fmt:
        dot = fmt.index(".")
        rest = fmt[dot + 1 :]
        fpos = rest.index("f")
        digits = rest[:fpos]
        return int(digits) if digits else 0
    return 3


def property_kind(default) -> str:
    if isinstance(default, str):
        return default
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, (list, tuple)):
        return "vector"
    return "float"


def collect_params(props, names: list[str]) -> dict:
    result = {}
    for name in names:
        value = getattr(props, name)
        if isinstance(value, (list, tuple)):
            result[name] = list(value)
        else:
            result[name] = value
    return result


def build_flame_property_group():
    import bpy
    import thyllore_effect_core as fx

    ui_params = fx.flame_ui_params()
    param_names = [p["name"] for p in ui_params]

    def apply_preset(self, context):
        preset_values = fx.flame_preset_params(self.preset)
        for name in param_names:
            if name in preset_values:
                value = preset_values[name]
                kind = property_kind(value)
                if kind == "vector":
                    for i, v in enumerate(value):
                        setattr(self, f"{name}_{i}", v)
                else:
                    setattr(self, name, value)

    annotations: dict[str, object] = {}

    for p in ui_params:
        name = p["name"]
        label = p["label"]
        tooltip = p["tooltip"]
        default = p["default"]
        kind = property_kind(default)

        if kind == "bool":
            annotations[name] = bpy.props.BoolProperty(
                name=label, description=tooltip, default=default
            )
        elif kind == "vector":
            subtype = "COLOR" if name.startswith("color_") else "NONE"
            annotations[name] = bpy.props.FloatVectorProperty(
                name=label,
                description=tooltip,
                default=default,
                size=len(default),
                subtype=subtype,
            )
        else:
            min_val = p.get("min")
            max_val = p.get("max")
            fmt = p.get("format", "%.3f")
            precision = precision_from_format(fmt)
            kwargs = {
                "name": label,
                "description": tooltip,
                "default": default,
                "precision": precision,
            }
            if min_val is not None:
                kwargs["min"] = min_val
            if max_val is not None:
                kwargs["max"] = max_val
            annotations[name] = bpy.props.FloatProperty(**kwargs)

    annotations["is_flame"] = bpy.props.BoolProperty(default=False)
    annotations["preset"] = bpy.props.EnumProperty(
        items=[(n, n.title(), "") for n in fx.flame_preset_names()],
        default="campfire",
        update=apply_preset,
    )

    attrs = {
        "__annotations__": annotations,
        "PARAM_NAMES": param_names,
        "__module__": __name__,
    }

    return type("ThylloreFlameProperties", (bpy.types.PropertyGroup,), attrs)
