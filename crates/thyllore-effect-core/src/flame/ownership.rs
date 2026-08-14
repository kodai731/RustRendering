use super::FlameEffect;

/// Parameter ownership split (design SSoT: style_preset.md / parameter_ownership.md):
/// Frame = placed by the scene, Shape = written by the still texture fit,
/// Style = written by the reference-footage style. Writers must stay inside
/// their declared set; the tests below enforce it by diffing snapshots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterOwner {
    Frame,
    Shape,
    Style,
}

use ParameterOwner::{Frame, Shape, Style};

/// Persisted flame parameters (scene serde field names) mapped to their owner.
/// Must stay 1:1 with `FlameEffectData` (coverage test in scene/format.rs)
/// and with `flame_parameter_snapshot` (coverage test below).
pub const PARAMETER_OWNERSHIP: &[(&str, ParameterOwner)] = &[
    ("position", Frame),
    ("rotation", Frame),
    ("height", Frame),
    ("radius", Frame),
    ("sigma_t", Style),
    ("intensity", Style),
    ("color_base", Style),
    ("color_tip", Style),
    ("temperature_base_k", Style),
    ("temperature_tip_k", Style),
    ("use_blackbody", Style),
    ("noise_amplitude", Style),
    ("noise_contrast", Style),
    ("noise_frequency", Style),
    ("noise_scroll_speed", Style),
    ("time_scale", Frame),
    ("time_offset", Frame),
    ("warp_amp", Style),
    ("warp_freq", Style),
    ("rise_speed", Style),
    ("taper_power", Shape),
    ("radius_tip_ratio", Shape),
    ("edge_low", Style),
    ("edge_high", Style),
    ("white_boost", Style),
    ("wind_direction", Frame),
    ("bend_amount", Frame),
    ("bend_power", Frame),
    ("self_shadow_strength", Style),
    ("envelope_peak", Shape),
    ("envelope_base", Shape),
    ("envelope_tail", Shape),
    ("radial_sharpness", Shape),
    ("occlusion_lum_ref", Style),
    ("contour_wiggle_amp", Style),
    ("aniso_axis_advect", Style),
    ("rte_bands", Style),
    ("sigma_dispersion", Style),
    ("edge_temperature_blend", Style),
    ("tip_carve_depth", Style),
    ("tip_carve_reach", Style),
    ("warp_reach", Style),
    ("swirl_gain", Style),
    ("swirl_speed", Style),
    ("spread_gain", Style),
    ("support_margin", Style),
    ("meander_amp", Style),
    ("edge_outer_sharpen", Style),
    ("noise_scale_mode", Style),
    ("erosion_noise_gain", Style),
    ("twist_gain", Style),
    ("twist_speed", Style),
    ("burnout_gain", Style),
    ("noise_shaping_scale", Style),
    ("optical_depth", Style),
];

pub fn parameter_owner(field: &str) -> Option<ParameterOwner> {
    PARAMETER_OWNERSHIP
        .iter()
        .find(|(name, _)| *name == field)
        .map(|(_, owner)| *owner)
}

pub fn parameters_owned_by(owner: ParameterOwner) -> Vec<&'static str> {
    PARAMETER_OWNERSHIP
        .iter()
        .filter(|(_, o)| *o == owner)
        .map(|(name, _)| *name)
        .collect()
}

/// Style parameters the texture fit's turbulence group is declared to write.
pub const TEXTURE_FIT_TURBULENCE_PARAMETERS: &[&str] =
    &["noise_amplitude", "noise_frequency", "contour_wiggle_amp"];

/// Style parameters the texture fit's color group is declared to write.
pub const TEXTURE_FIT_COLOR_PARAMETERS: &[&str] = &[
    "color_base",
    "color_tip",
    "temperature_base_k",
    "temperature_tip_k",
    "use_blackbody",
];

/// Frame parameters the texture fit's tilt group is declared to write — the one
/// documented exception where a fit writes Frame-owned parameters (user opt-in
/// via the tilt group checkbox).
pub const TEXTURE_FIT_TILT_PARAMETERS: &[&str] = &["wind_direction", "bend_amount"];

/// Bit-exact value snapshot of every persisted parameter, keyed by the same field
/// names as `PARAMETER_OWNERSHIP`. Diffing two snapshots yields the exact set of
/// parameters a writer touched.
pub fn flame_parameter_snapshot(effect: &FlameEffect) -> Vec<(&'static str, Vec<f32>)> {
    vec![
        (
            "position",
            vec![effect.position.x, effect.position.y, effect.position.z],
        ),
        (
            "rotation",
            vec![
                effect.rotation.s,
                effect.rotation.v.x,
                effect.rotation.v.y,
                effect.rotation.v.z,
            ],
        ),
        ("height", vec![effect.height]),
        ("radius", vec![effect.radius]),
        ("sigma_t", vec![effect.sigma_t]),
        ("intensity", vec![effect.intensity]),
        ("color_base", effect.color_base.to_vec()),
        ("color_tip", effect.color_tip.to_vec()),
        ("temperature_base_k", vec![effect.temperature_base_k]),
        ("temperature_tip_k", vec![effect.temperature_tip_k]),
        ("use_blackbody", vec![effect.use_blackbody as u8 as f32]),
        ("noise_amplitude", vec![effect.noise_amplitude]),
        ("noise_contrast", vec![effect.noise_contrast]),
        ("noise_frequency", vec![effect.noise_frequency]),
        ("noise_scroll_speed", vec![effect.noise_scroll_speed]),
        ("time_scale", vec![effect.time_scale]),
        ("time_offset", vec![effect.time_offset]),
        ("warp_amp", vec![effect.warp_amp]),
        ("warp_freq", vec![effect.warp_freq]),
        ("rise_speed", vec![effect.rise_speed]),
        ("taper_power", vec![effect.taper_power]),
        ("radius_tip_ratio", vec![effect.radius_tip_ratio]),
        ("edge_low", vec![effect.edge_low]),
        ("edge_high", vec![effect.edge_high]),
        ("white_boost", vec![effect.white_boost]),
        (
            "wind_direction",
            vec![effect.wind_direction.x, effect.wind_direction.y],
        ),
        ("bend_amount", vec![effect.bend_amount]),
        ("bend_power", vec![effect.bend_power]),
        ("self_shadow_strength", vec![effect.self_shadow_strength]),
        ("envelope_peak", vec![effect.envelope_peak]),
        ("envelope_base", vec![effect.envelope_base]),
        ("envelope_tail", vec![effect.envelope_tail]),
        ("radial_sharpness", vec![effect.radial_sharpness]),
        ("occlusion_lum_ref", vec![effect.occlusion_lum_ref]),
        ("contour_wiggle_amp", vec![effect.contour_wiggle_amp]),
        ("aniso_axis_advect", vec![effect.aniso_axis_advect]),
        ("rte_bands", vec![effect.rte_bands]),
        ("sigma_dispersion", vec![effect.sigma_dispersion]),
        (
            "edge_temperature_blend",
            vec![effect.edge_temperature_blend],
        ),
        ("tip_carve_depth", vec![effect.tip_carve_depth]),
        ("tip_carve_reach", vec![effect.tip_carve_reach]),
        ("warp_reach", vec![effect.warp_reach]),
        ("swirl_gain", vec![effect.swirl_gain]),
        ("swirl_speed", vec![effect.swirl_speed]),
        ("spread_gain", vec![effect.spread_gain]),
        ("support_margin", vec![effect.support_margin]),
        ("meander_amp", vec![effect.meander_amp]),
        ("edge_outer_sharpen", vec![effect.edge_outer_sharpen]),
        ("noise_scale_mode", vec![effect.noise_scale_mode]),
        ("erosion_noise_gain", vec![effect.erosion_noise_gain]),
        ("twist_gain", vec![effect.twist_gain]),
        ("twist_speed", vec![effect.twist_speed]),
        ("burnout_gain", vec![effect.burnout_gain]),
        ("noise_shaping_scale", vec![effect.noise_shaping_scale]),
        ("optical_depth", vec![effect.optical_depth]),
    ]
}

pub fn changed_parameters(
    before: &[(&'static str, Vec<f32>)],
    after: &[(&'static str, Vec<f32>)],
) -> Vec<&'static str> {
    assert_eq!(before.len(), after.len());
    before
        .iter()
        .zip(after)
        .filter(|((name_b, values_b), (name_a, values_a))| {
            assert_eq!(name_b, name_a);
            values_b
                .iter()
                .zip(values_a)
                .any(|(b, a)| b.to_bits() != a.to_bits())
        })
        .map(|((name, _), _)| *name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flame::bake::texture_fit::{apply_texture_fit, FlameTextureFit, TextureFitGroups};
    use crate::flame::FlameBaked;
    use std::collections::HashSet;

    fn extreme_fit() -> FlameTextureFit {
        FlameTextureFit {
            envelope_peak: 2.0,
            envelope_base: 0.5,
            envelope_tail: 0.1,
            radius: 1.0,
            radius_tip_ratio: 0.3,
            taper_power: 4.0,
            temperature_base_k: 3000.0,
            temperature_tip_k: 1500.0,
            noise_amplitude: 0.8,
            contour_wiggle_amp: 0.6,
            noise_frequency: 5.0,
            wind_x: 1.0,
            wind_z: 0.5,
            bend_amount: 0.9,
            use_blackbody: true,
            color_bands: [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            suggested_instances: 1,
            envelope_profile: [0.5; 33],
            radius_profile: [0.4; 33],
            color_ramp: [[1.0, 0.5, 0.2]; 8],
        }
    }

    fn diff_from_texture_fit(profile: bool) -> Vec<&'static str> {
        let mut effect = FlameEffect::default();
        let mut baked = FlameBaked::default();
        let before = flame_parameter_snapshot(&effect);
        apply_texture_fit(
            &mut effect,
            &mut baked,
            &extreme_fit(),
            TextureFitGroups::default(),
            1.0,
            profile,
        );
        changed_parameters(&before, &flame_parameter_snapshot(&effect))
    }

    fn allowed_shape_color_tilt() -> HashSet<&'static str> {
        parameters_owned_by(ParameterOwner::Shape)
            .into_iter()
            .chain(TEXTURE_FIT_COLOR_PARAMETERS.iter().copied())
            .chain(TEXTURE_FIT_TILT_PARAMETERS.iter().copied())
            .collect()
    }

    #[test]
    fn test_ownership_table_has_no_duplicates() {
        let mut names: Vec<&str> = PARAMETER_OWNERSHIP.iter().map(|(name, _)| *name).collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
    }

    #[test]
    fn test_snapshot_covers_ownership_table_exactly() {
        let snapshot = flame_parameter_snapshot(&FlameEffect::default());
        let snapshot_names: Vec<&str> = snapshot.iter().map(|(name, _)| *name).collect();
        let table_names: Vec<&str> = PARAMETER_OWNERSHIP.iter().map(|(name, _)| *name).collect();
        assert_eq!(snapshot_names, table_names);
    }

    #[test]
    fn test_declared_fit_parameters_exist_in_table() {
        for name in TEXTURE_FIT_TURBULENCE_PARAMETERS
            .iter()
            .chain(TEXTURE_FIT_COLOR_PARAMETERS)
            .chain(TEXTURE_FIT_TILT_PARAMETERS)
        {
            assert!(parameter_owner(name).is_some(), "{name}");
        }
    }

    #[test]
    fn test_profile_fit_writes_only_shape_color_tilt() {
        let changed = diff_from_texture_fit(true);
        let allowed = allowed_shape_color_tilt();
        for name in &changed {
            assert!(allowed.contains(name), "profile fit wrote {name}");
        }
        for name in TEXTURE_FIT_TURBULENCE_PARAMETERS {
            assert!(
                !changed.contains(name),
                "profile fit wrote turbulence parameter {name}"
            );
        }
    }

    #[test]
    fn test_statistics_fit_stays_inside_declared_set() {
        let changed = diff_from_texture_fit(false);
        let mut allowed = allowed_shape_color_tilt();
        allowed.extend(TEXTURE_FIT_TURBULENCE_PARAMETERS);
        for name in &changed {
            assert!(allowed.contains(name), "statistics fit wrote {name}");
        }
    }
}
