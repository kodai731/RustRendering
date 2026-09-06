from __future__ import annotations

from ._common import effect_properties

precision_from_format = effect_properties.precision_from_format
property_kind = effect_properties.property_kind
collect_params = effect_properties.collect_params
merge_preset_params = effect_properties.merge_preset_params
select_exposed_params = effect_properties.select_exposed_params


def wind_render_params(props) -> dict:
    import thyllore_effect_core as fx

    return effect_properties.render_params(props, fx.wind_preset_params)


def build_wind_property_group():
    import thyllore_effect_core as fx

    return effect_properties.build_effect_property_group(
        ui_params=fx.wind_ui_params,
        preset_params=fx.wind_preset_params,
        preset_names=fx.wind_preset_names,
        flag_name="is_wind",
        default_preset="funnel",
        class_name="ThylloreWindProperties",
        module_name=__name__,
    )
