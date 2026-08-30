from __future__ import annotations

import tomllib
from pathlib import Path

PARAMS_FILE = Path(__file__).resolve().parent / "flame_params.toml"


def load_exposed_param_rules(params_file: Path = PARAMS_FILE) -> dict:
    exposed = tomllib.loads(params_file.read_text(encoding="utf-8"))["exposed"]
    return {"names": tuple(exposed["names"]), "prefixes": tuple(exposed["prefixes"])}


EXPOSED_PARAM_RULES = load_exposed_param_rules()


def resolve_preset_values(preset_values: dict, effective_optical_depth: float) -> dict:
    resolved = dict(preset_values)
    resolved["optical_depth"] = effective_optical_depth
    return resolved


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


def is_exposed_param(name: str, rules: dict = EXPOSED_PARAM_RULES) -> bool:
    return name in rules["names"] or name.startswith(rules["prefixes"])


def select_exposed_params(ui_params: list[dict], rules: dict = EXPOSED_PARAM_RULES) -> list[dict]:
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


def flame_render_params(props) -> dict:
    import thyllore_effect_core as fx

    preset_values = fx.flame_preset_params(props.preset)
    exposed_values = collect_params(props, type(props).PARAM_NAMES)
    return merge_preset_params(preset_values, exposed_values)


def build_flame_property_group():
    import bpy
    import thyllore_effect_core as fx

    ui_params = select_exposed_params(fx.flame_ui_params())
    param_names = [p["name"] for p in ui_params]

    def apply_preset(self, context):
        preset_values = fx.flame_preset_params(self.preset)
        preset_values = resolve_preset_values(preset_values, fx.flame_effective_optical_depth(preset_values))
        for name in param_names:
            if name in preset_values:
                setattr(self, name, preset_values[name])

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
