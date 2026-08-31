pub const WATER_SHAPE_PARAMS: &[&str] = &["major_radius", "minor_radius"];

pub const WATER_OPTICS_PARAMS: &[&str] = &["ior", "absorption_r", "absorption_g", "absorption_b"];

pub const WATER_FLOW_PARAMS: &[&str] = &["flow_longitudinal", "flow_meridional"];

pub const WATER_WAVE_PARAMS: &[&str] = &["wave_amplitude", "wave_frequency", "wave_speed"];

pub const WATER_LOOK_PARAMS: &[&str] = &[
    "reflect_strength",
    "refract_strength",
    "caustic_strength",
    "tint_r",
    "tint_g",
    "tint_b",
];
pub const WATER_PARAM_GROUPS: &[&[&str]] = &[
    WATER_SHAPE_PARAMS,
    WATER_OPTICS_PARAMS,
    WATER_FLOW_PARAMS,
    WATER_WAVE_PARAMS,
    WATER_LOOK_PARAMS,
];

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_effect_core::{WATER_SCALAR_PARAMS, WATER_UI_PARAMS};
    use thyllore_scene_core::{find_scalar_param, find_ui_param};

    #[test]
    fn test_every_grouped_name_resolves_to_ui_and_scalar_param() {
        for name in WATER_PARAM_GROUPS.iter().flat_map(|group| group.iter()) {
            assert!(find_ui_param(WATER_UI_PARAMS, name).is_some(), "{name}");
            assert!(
                find_scalar_param(WATER_SCALAR_PARAMS, name).is_some(),
                "{name}"
            );
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
