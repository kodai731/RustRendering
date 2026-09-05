from __future__ import annotations

import math
from typing import Callable

ABSORPTION_COLOR_FLOOR = 1e-3


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


def absorption_to_color(absorption: list[float], reference_distance: float) -> list[float]:
    return [math.exp(-coefficient * reference_distance) for coefficient in absorption]


def color_to_absorption(color: list[float], reference_distance: float) -> list[float]:
    return [-math.log(max(channel, ABSORPTION_COLOR_FLOOR)) / reference_distance for channel in color]


def absorption_color_property_name(name: str) -> str:
    return f"{name}_color"


def display_property_names(exposed_params: list[dict]) -> dict[str, str]:
    """Picker attribute shown in place of a parameter whose stored value is not the edited colour."""
    return {
        p["name"]: absorption_color_property_name(p["name"])
        for p in exposed_params
        if p.get("kind") == "absorption"
    }


def select_exposed_params(ui_params: list[dict]) -> list[dict]:
    """Mirrors the engine's persisted UI parameters; runtime-only ones are driven by scene playback."""
    return [p for p in ui_params if p["persisted"]]


def group_params_by_owner(exposed_params: list[dict]) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for param in exposed_params:
        groups.setdefault(param.get("owner", "frame"), []).append(param["name"])
    return list(groups.items())


def draw_param_groups(layout, props) -> None:
    groups = type(props).PARAM_GROUPS
    display_names = type(props).PARAM_DISPLAY_NAMES
    for owner, names in groups:
        box = layout.box()
        if len(groups) > 1:
            box.label(text=owner.title())
        for name in names:
            box.prop(props, display_names.get(name, name))


def collect_params(props, names: list[str]) -> dict:
    result = {}
    for name in names:
        value = getattr(props, name)
        if isinstance(value, (str, bool, int, float)):
            result[name] = value
        else:
            result[name] = [float(v) for v in value]
    return result


def merge_preset_params(preset_values: dict, exposed_values: dict) -> dict:
    merged = dict(preset_values)
    merged.update(exposed_values)
    return merged


def render_params(props, preset_params: Callable[[str], dict]) -> dict:
    preset_values = preset_params(props.preset)
    exposed_values = collect_params(props, type(props).PARAM_NAMES)
    return merge_preset_params(preset_values, exposed_values)


def _range_kwargs(param: dict) -> dict:
    kwargs = {}
    if param.get("min") is not None:
        kwargs["min"] = param["min"]
    if param.get("max") is not None:
        kwargs["max"] = param["max"]
    return kwargs


def _build_color_property(param: dict):
    import bpy

    return bpy.props.FloatVectorProperty(
        name=param["label"],
        description=param["tooltip"],
        default=param["default"],
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
    )


def _build_absorption_properties(param: dict) -> dict[str, object]:
    """The stored coefficient stays keyed by the engine name; the picker is a derived colour property."""
    import bpy

    name = param["name"]
    reference_distance = float(param["reference_distance"])

    def get_color(self):
        return absorption_to_color(list(getattr(self, name)), reference_distance)

    def set_color(self, value):
        setattr(self, name, color_to_absorption(list(value), reference_distance))

    coefficient = bpy.props.FloatVectorProperty(
        name=param["label"],
        description=param["tooltip"],
        default=param["default"],
        size=3,
        options={"HIDDEN"},
        **_range_kwargs(param),
    )
    picker = bpy.props.FloatVectorProperty(
        name=param["label"],
        description=f"{param['tooltip']} ({reference_distance:g} m)",
        default=absorption_to_color(list(param["default"]), reference_distance),
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        get=get_color,
        set=set_color,
    )
    return {name: coefficient, absorption_color_property_name(name): picker}


def build_param_properties(param: dict) -> dict[str, object]:
    """Blender property annotations for one engine parameter, keyed by attribute name."""
    import bpy

    name = param["name"]
    label = param["label"]
    tooltip = param["tooltip"]
    default = param["default"]

    ui_kind = param.get("kind", "scalar")
    if ui_kind == "color":
        return {name: _build_color_property(param)}
    if ui_kind == "absorption":
        return _build_absorption_properties(param)

    kind = property_kind(default)
    if kind == "bool":
        return {name: bpy.props.BoolProperty(name=label, description=tooltip, default=default)}

    if kind == "vector":
        return {
            name: bpy.props.FloatVectorProperty(
                name=label,
                description=tooltip,
                default=default,
                size=len(default),
            )
        }

    return {
        name: bpy.props.FloatProperty(
            name=label,
            description=tooltip,
            default=default,
            precision=precision_from_format(param.get("format", "%.3f")),
            **_range_kwargs(param),
        )
    }


def build_effect_property_group(
    *,
    ui_params: Callable[[], list[dict]],
    preset_params: Callable[[str], dict],
    preset_names: Callable[[], list[str]],
    flag_name: str,
    default_preset: str,
    class_name: str,
    module_name: str,
    preset_values_post_process: Callable[[dict], dict] | None = None,
):
    import bpy

    exposed_params = select_exposed_params(ui_params())
    param_names = [p["name"] for p in exposed_params]

    def apply_preset(self, context):
        preset_values = preset_params(self.preset)
        if preset_values_post_process is not None:
            preset_values = preset_values_post_process(preset_values)
        for name in param_names:
            if name in preset_values:
                setattr(self, name, preset_values[name])

    annotations: dict[str, object] = {}
    for param in exposed_params:
        annotations.update(build_param_properties(param))

    annotations[flag_name] = bpy.props.BoolProperty(default=False)
    annotations["preset"] = bpy.props.EnumProperty(
        items=[(n, n.title(), "") for n in preset_names()],
        default=default_preset,
        update=apply_preset,
    )

    attrs = {
        "__annotations__": annotations,
        "PARAM_NAMES": param_names,
        "PARAM_GROUPS": group_params_by_owner(exposed_params),
        "PARAM_DISPLAY_NAMES": display_property_names(exposed_params),
        "__module__": module_name,
    }

    return type(class_name, (bpy.types.PropertyGroup,), attrs)
