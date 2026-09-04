pub const WATER_SHAPE_PARAMS: &[&str] = &["major_radius", "minor_radius"];

pub const WATER_OPTICS_PARAMS: &[&str] = &["ior", "absorption"];

pub const WATER_FLOW_PARAMS: &[&str] = &["flow_longitudinal", "flow_meridional"];

pub const WATER_WAVE_PARAMS: &[&str] = &[
    "wave_amplitude",
    "wave_frequency",
    "wave_speed",
    "wave_dispersion",
    "wave_lb_blend",
];

pub const WATER_LIGHTING_PARAMS: &[&str] = &[
    "light_intensity",
    "highlight_sharpness",
    "sky_brightness",
    "scatter_strength",
    "scatter_anisotropy",
];

pub const WATER_LOOK_PARAMS: &[&str] = &[
    "reflect_strength",
    "refract_strength",
    "caustic_strength",
    "tint",
];
pub const WATER_PARAM_GROUPS: &[&[&str]] = &[
    WATER_SHAPE_PARAMS,
    WATER_OPTICS_PARAMS,
    WATER_FLOW_PARAMS,
    WATER_WAVE_PARAMS,
    WATER_LIGHTING_PARAMS,
    WATER_LOOK_PARAMS,
];

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_effect_core::{WATER_SCALAR_PARAMS, WATER_UI_PARAMS};
    use thyllore_scene_core::{find_scalar_param, find_ui_param, UiKind};

    #[test]
    fn test_every_grouped_name_resolves_to_ui_and_scalar_params() {
        for name in WATER_PARAM_GROUPS.iter().flat_map(|group| group.iter()) {
            let meta = find_ui_param(WATER_UI_PARAMS, name).unwrap_or_else(|| panic!("{name}"));
            let accessor_names = match meta.kind {
                UiKind::Scalar => vec![name.to_string()],
                UiKind::Color | UiKind::Absorption => meta.color_component_names().to_vec(),
            };
            for accessor_name in accessor_names {
                assert!(
                    find_scalar_param(WATER_SCALAR_PARAMS, &accessor_name).is_some(),
                    "{accessor_name}"
                );
            }
        }
    }

    #[test]
    fn test_grouped_names_are_unique() {
        let mut names: Vec<&str> = WATER_PARAM_GROUPS
            .iter()
            .flat_map(|group| group.iter().copied())
            .collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
    }
}
