pub const WIND_SHAPE_PARAMS: &[&str] = &[
    "column_height",
    "wall_radius_base",
    "wall_radius_top",
    "wall_width_q",
    "core_radius",
    "top_fade",
];

pub const WIND_DENSITY_PARAMS: &[&str] = &["density", "wall_strength", "core_strength"];

pub const WIND_LOOK_PARAMS: &[&str] = &["albedo", "ambient_brightness", "phase_g", "sun_intensity"];

pub const WIND_MOTION_PARAMS: &[&str] = &[
    "rise_initial_height",
    "rise_duration",
    "spread_start",
    "spread_rate",
    "dissipate_start",
    "dissipate_time",
    "ring_height",
    "ring_radius",
    "ring_width_q",
    "ring_strength",
    "ring_spread_rate",
];

pub const WIND_PARAM_GROUPS: &[&[&str]] = &[
    WIND_SHAPE_PARAMS,
    WIND_DENSITY_PARAMS,
    WIND_MOTION_PARAMS,
    WIND_LOOK_PARAMS,
];

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_effect_core::{WIND_SCALAR_PARAMS, WIND_UI_PARAMS};
    use thyllore_scene_core::{find_scalar_param, find_ui_param};

    #[test]
    fn test_every_grouped_name_resolves_to_ui_and_scalar_params() {
        for name in WIND_PARAM_GROUPS.iter().flat_map(|group| group.iter()) {
            let meta = find_ui_param(WIND_UI_PARAMS, name).unwrap_or_else(|| panic!("{name}"));
            for accessor_name in meta.scalar_accessor_names() {
                assert!(
                    find_scalar_param(WIND_SCALAR_PARAMS, &accessor_name).is_some(),
                    "{accessor_name}"
                );
            }
        }
    }

    #[test]
    fn test_grouped_names_are_unique_and_cover_every_persisted_scalar() {
        let mut names: Vec<&str> = WIND_PARAM_GROUPS
            .iter()
            .flat_map(|group| group.iter().copied())
            .collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
        for param in WIND_UI_PARAMS.iter().filter(|p| p.persisted) {
            assert!(names.contains(&param.name), "{} is not grouped", param.name);
        }
    }
}
