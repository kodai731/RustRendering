from __future__ import annotations

from pathlib import Path

from ._common import effect_properties

PARAMS_FILE = Path(__file__).resolve().parent / "water_params.toml"


def load_exposed_param_rules(params_file: Path = PARAMS_FILE) -> dict:
    return effect_properties.load_exposed_param_rules(params_file)


EXPOSED_PARAM_RULES = load_exposed_param_rules()

precision_from_format = effect_properties.precision_from_format
property_kind = effect_properties.property_kind
collect_params = effect_properties.collect_params
merge_preset_params = effect_properties.merge_preset_params


def is_exposed_param(name: str, rules: dict = EXPOSED_PARAM_RULES) -> bool:
    return effect_properties.is_exposed_param(name, rules)


def select_exposed_params(ui_params: list[dict], rules: dict = EXPOSED_PARAM_RULES) -> list[dict]:
    return effect_properties.select_exposed_params(ui_params, rules)


def water_render_params(props) -> dict:
    import thyllore_effect_core as fx

    return effect_properties.render_params(props, fx.water_preset_params)


def build_water_property_group():
    import thyllore_effect_core as fx

    return effect_properties.build_effect_property_group(
        params_file=PARAMS_FILE,
        ui_params=fx.water_ui_params,
        preset_params=fx.water_preset_params,
        preset_names=fx.water_preset_names,
        flag_name="is_water",
        default_preset="pond",
        class_name="ThylloreWaterProperties",
        module_name=__name__,
    )
