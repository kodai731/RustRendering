pub const FLAME_BODY_PARAMS: &[&str] = &[
    "height",
    "radius",
    "base_spread",
    "base_spread_height",
    "optical_depth",
    "intensity",
    "density_exp",
    "temp_exp",
    "wien_c_k",
];

pub const FLAME_NOISE_PARAMS: &[&str] = &[
    "noise_amplitude",
    "noise_contrast",
    "swirl_gain",
    "noise_aniso_y",
    "noise_lobe_scale",
    "noise_lobe_aniso",
];

pub const FLAME_MIX_PARAMS: &[&str] = &[
    "mix_lo",
    "mix_hi",
    "mix_scale",
    "mix_radial_gain",
    "mix_core_radius",
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

pub const FLAME_PUFF_PARAMS: &[&str] = &[
    "puff_gain",
    "puff_period",
    "puff_rise",
    "puff_radius",
    "puff_spread",
    "puff_decay",
    "puff_aspect",
    "puff_spawn_height",
];

pub const FLAME_FLOW_PARAMS: &[&str] = &[
    "flow_gain",
    "flow_period",
    "flow_rise",
    "flow_strength",
    "flow_core",
    "flow_gust",
    "flow_gust_frequency",
    "flow_burst",
    "flow_damping",
];

pub const FLAME_LOBE_PARAMS: &[&str] = &[
    "lobe_gain",
    "lobe_period",
    "lobe_life",
    "lobe_rise",
    "lobe_size",
    "lobe_spawn_height",
    "lobe_spawn_range",
    "lobe_accel",
    "lobe_spread",
    "lobe_shift",
];

pub const FLAME_FOOTER_PARAMS: &[&str] = &["support_margin", "time_scale"];

pub const FLAME_PARAM_GROUPS: &[&[&str]] = &[
    FLAME_BODY_PARAMS,
    FLAME_NOISE_PARAMS,
    FLAME_MIX_PARAMS,
    FLAME_MOTION_PARAMS,
    FLAME_BRANCH_PARAMS,
    FLAME_PUFF_PARAMS,
    FLAME_FLOW_PARAMS,
    FLAME_LOBE_PARAMS,
    FLAME_FOOTER_PARAMS,
];

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_effect_core::{FLAME_SCALAR_PARAMS, FLAME_UI_PARAMS};
    use thyllore_scene_core::{find_scalar_param, find_ui_param};

    #[test]
    fn test_every_grouped_name_resolves_to_ui_and_scalar_param() {
        for name in FLAME_PARAM_GROUPS.iter().flat_map(|group| group.iter()) {
            assert!(find_ui_param(FLAME_UI_PARAMS, name).is_some(), "{name}");
            assert!(
                find_scalar_param(FLAME_SCALAR_PARAMS, name).is_some(),
                "{name}"
            );
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
