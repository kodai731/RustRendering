from __future__ import annotations

from ._common import effect_properties

precision_from_format = effect_properties.precision_from_format
property_kind = effect_properties.property_kind
collect_params = effect_properties.collect_params
merge_preset_params = effect_properties.merge_preset_params
select_exposed_params = effect_properties.select_exposed_params


def resolve_preset_values(preset_values: dict, effective_optical_depth: float) -> dict:
    resolved = dict(preset_values)
    resolved["optical_depth"] = effective_optical_depth
    return resolved


def flame_render_params(props) -> dict:
    import thyllore_effect_core as fx

    return effect_properties.render_params(props, fx.flame_preset_params)


def build_flame_property_group():
    import thyllore_effect_core as fx

    def replace_optical_depth_with_effective(preset_values: dict) -> dict:
        return resolve_preset_values(preset_values, fx.flame_effective_optical_depth(preset_values))

    return effect_properties.build_effect_property_group(
        ui_params=fx.flame_ui_params,
        preset_params=fx.flame_preset_params,
        preset_names=fx.flame_preset_names,
        flag_name="is_flame",
        default_preset="campfire",
        class_name="ThylloreFlameProperties",
        module_name=__name__,
        preset_values_post_process=replace_optical_depth_with_effective,
    )
