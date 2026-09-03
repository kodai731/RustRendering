from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Callable


def load_exposed_param_rules(params_file: Path) -> dict:
    exposed = tomllib.loads(params_file.read_text(encoding="utf-8"))["exposed"]
    return {"names": tuple(exposed["names"]), "prefixes": tuple(exposed["prefixes"])}


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


def is_exposed_param(name: str, rules: dict) -> bool:
    return name in rules["names"] or name.startswith(rules["prefixes"])


def select_exposed_params(ui_params: list[dict], rules: dict) -> list[dict]:
    return [p for p in ui_params if is_exposed_param(p["name"], rules)]


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


def build_param_property(param: dict):
    import bpy

    name = param["name"]
    label = param["label"]
    tooltip = param["tooltip"]
    default = param["default"]
    kind = property_kind(default)

    if kind == "bool":
        return bpy.props.BoolProperty(name=label, description=tooltip, default=default)

    if kind == "vector":
        subtype = "COLOR" if name.startswith("color_") else "NONE"
        return bpy.props.FloatVectorProperty(
            name=label,
            description=tooltip,
            default=default,
            size=len(default),
            subtype=subtype,
        )

    kwargs = {
        "name": label,
        "description": tooltip,
        "default": default,
        "precision": precision_from_format(param.get("format", "%.3f")),
    }
    min_val = param.get("min")
    max_val = param.get("max")
    if min_val is not None:
        kwargs["min"] = min_val
    if max_val is not None:
        kwargs["max"] = max_val
    return bpy.props.FloatProperty(**kwargs)


def build_effect_property_group(
    *,
    params_file: Path,
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

    rules = load_exposed_param_rules(params_file)
    exposed_params = select_exposed_params(ui_params(), rules)
    param_names = [p["name"] for p in exposed_params]

    def apply_preset(self, context):
        preset_values = preset_params(self.preset)
        if preset_values_post_process is not None:
            preset_values = preset_values_post_process(preset_values)
        for name in param_names:
            if name in preset_values:
                setattr(self, name, preset_values[name])

    annotations: dict[str, object] = {p["name"]: build_param_property(p) for p in exposed_params}

    annotations[flag_name] = bpy.props.BoolProperty(default=False)
    annotations["preset"] = bpy.props.EnumProperty(
        items=[(n, n.title(), "") for n in preset_names()],
        default=default_preset,
        update=apply_preset,
    )

    attrs = {
        "__annotations__": annotations,
        "PARAM_NAMES": param_names,
        "__module__": module_name,
    }

    return type(class_name, (bpy.types.PropertyGroup,), attrs)
