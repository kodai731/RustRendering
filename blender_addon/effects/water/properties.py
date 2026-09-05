from __future__ import annotations

from ._common import effect_properties

precision_from_format = effect_properties.precision_from_format
property_kind = effect_properties.property_kind
collect_params = effect_properties.collect_params
merge_preset_params = effect_properties.merge_preset_params
select_exposed_params = effect_properties.select_exposed_params
absorption_to_color = effect_properties.absorption_to_color
color_to_absorption = effect_properties.color_to_absorption
display_property_names = effect_properties.display_property_names


def water_render_params(props) -> dict:
    import thyllore_effect_core as fx

    return effect_properties.render_params(props, fx.water_preset_params)


def build_water_property_group():
    import thyllore_effect_core as fx

    return effect_properties.build_effect_property_group(
        ui_params=fx.water_ui_params,
        preset_params=fx.water_preset_params,
        preset_names=fx.water_preset_names,
        flag_name="is_water",
        default_preset="pond",
        class_name="ThylloreWaterProperties",
        module_name=__name__,
    )
