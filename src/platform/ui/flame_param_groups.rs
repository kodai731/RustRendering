pub const FLAME_BODY_PARAMS: &[&str] = &[
    "height",
    "radius",
    "optical_depth",
    "intensity",
    "density_exp",
    "temp_exp",
    "wien_c_k",
];

pub const FLAME_COLOR_PARAMS: &[&str] = &["color_base", "color_tip"];

pub const FLAME_NOISE_PARAMS: &[&str] = &[
    "noise_amplitude",
    "noise_contrast",
    "swirl_gain",
    "noise_aniso_y",
];

pub const FLAME_MIX_PARAMS: &[&str] = &[
    "mix_lo",
    "mix_hi",
    "mix_scale",
    "mix_radial_gain",
    "mix_height_gain",
];

pub const FLAME_MOTION_PARAMS: &[&str] = &[
    "twist_gain",
    "twist_speed",
    "burnout_gain",
    "carve_residual",
    "meander_amp",
    "meander_frequency",
    "swirl_speed",
    "spread_gain",
];

pub const FLAME_BRANCH_PARAMS: &[&str] = &[
    "branch_period",
    "branch_life",
    "branch_gain",
    "branch_core_radius",
    "branch_core_offset",
    "branch_reach",
    "branch_spread",
    "branch_spawn_height",
    "branch_spawn_range",
];

pub const FLAME_FOOTER_PARAMS: &[&str] = &["support_margin", "time_scale"];

pub const FLAME_PARAM_GROUPS: &[&[&str]] = &[
    FLAME_BODY_PARAMS,
    FLAME_COLOR_PARAMS,
    FLAME_NOISE_PARAMS,
    FLAME_MIX_PARAMS,
    FLAME_MOTION_PARAMS,
    FLAME_BRANCH_PARAMS,
    FLAME_FOOTER_PARAMS,
];

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_effect_core::{FLAME_SCALAR_PARAMS, FLAME_UI_PARAMS};
    use thyllore_scene_core::{find_scalar_param, find_ui_param};

    #[test]
    fn test_every_grouped_name_resolves_to_ui_and_scalar_params() {
        for name in FLAME_PARAM_GROUPS.iter().flat_map(|group| group.iter()) {
            let meta = find_ui_param(FLAME_UI_PARAMS, name).unwrap_or_else(|| panic!("{name}"));
            for accessor_name in meta.scalar_accessor_names() {
                assert!(
                    find_scalar_param(FLAME_SCALAR_PARAMS, &accessor_name).is_some(),
                    "{accessor_name}"
                );
            }
        }
    }

    #[test]
    fn test_grouped_names_are_unique() {
        let mut names: Vec<&str> = FLAME_PARAM_GROUPS
            .iter()
            .flat_map(|group| group.iter().copied())
            .collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
    }
}
