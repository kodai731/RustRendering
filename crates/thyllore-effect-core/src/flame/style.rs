use serde::{Deserialize, Serialize};

use super::FlameEffect;

/// A named, dimensionless look extracted from reference footage (design SSoT:
/// style_preset.md). Every field is optional: `None` means "this reference did
/// not determine the parameter" and the target keeps its current value.
/// Only Style-owned parameters appear here (parameter_ownership.md); lengths
/// are stored over r0 and opacity as tau0 so one style transfers across sizes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FlameStyle {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub motion: FlameStyleMotion,
    #[serde(default)]
    pub texture: FlameStyleTexture,
    #[serde(default)]
    pub optics: FlameStyleOptics,
}

pub const FLAME_STYLE_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlameStyleMotion {
    pub noise_scroll_speed: Option<f32>,
    pub rise_speed: Option<f32>,
    pub warp_amp: Option<f32>,
    pub warp_freq: Option<f32>,
    pub swirl_gain: Option<f32>,
    pub swirl_speed: Option<f32>,
    pub twist_gain: Option<f32>,
    pub twist_speed: Option<f32>,
    pub spread_gain: Option<f32>,
    pub meander_amp_over_r0: Option<f32>,
    pub burnout_gain: Option<f32>,
    pub aniso_axis_advect: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlameStyleTexture {
    pub noise_amplitude: Option<f32>,
    pub noise_contrast: Option<f32>,
    pub noise_frequency: Option<f32>,
    pub noise_shaping_scale: Option<f32>,
    pub erosion_noise_gain: Option<f32>,
    pub support_margin: Option<f32>,
    pub contour_wiggle_amp: Option<f32>,
    pub edge_outer_sharpen: Option<f32>,
    pub noise_scale_mode: Option<f32>,
    pub edge_low: Option<f32>,
    pub edge_high: Option<f32>,
    pub tip_carve_depth: Option<f32>,
    pub tip_carve_reach: Option<f32>,
    pub warp_reach: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlameStyleOptics {
    pub tau0: Option<f32>,
    pub intensity: Option<f32>,
    pub temperature_base_k: Option<f32>,
    pub temperature_tip_k: Option<f32>,
    pub use_blackbody: Option<bool>,
    pub white_boost: Option<f32>,
    pub self_shadow_strength: Option<f32>,
    pub sigma_dispersion: Option<f32>,
    pub rte_bands: Option<f32>,
    pub edge_temperature_blend: Option<f32>,
    pub occlusion_lum_ref: Option<f32>,
    pub color_base: Option<[f32; 3]>,
    pub color_tip: Option<[f32; 3]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StyleGroups {
    pub motion: bool,
    pub texture: bool,
    pub optics: bool,
}

impl Default for StyleGroups {
    fn default() -> Self {
        Self {
            motion: true,
            texture: true,
            optics: true,
        }
    }
}

macro_rules! apply_style_param {
    ($applied:ident, $effect:ident, $source:expr, $field:ident) => {
        if let Some(value) = $source.$field {
            $effect.$field = value;
            $applied.push(stringify!($field));
        }
    };
}

/// Apply the style's `Some` fields onto the effect, returning the parameter
/// names that were written. Idempotent, order-free against Frame/Shape
/// writers (the written set is Style-owned only).
pub fn apply_flame_style(
    effect: &mut FlameEffect,
    style: &FlameStyle,
    groups: StyleGroups,
) -> Vec<&'static str> {
    let mut applied = Vec::new();

    if groups.motion {
        let motion = &style.motion;
        apply_style_param!(applied, effect, motion, noise_scroll_speed);
        apply_style_param!(applied, effect, motion, rise_speed);
        apply_style_param!(applied, effect, motion, warp_amp);
        apply_style_param!(applied, effect, motion, warp_freq);
        apply_style_param!(applied, effect, motion, swirl_gain);
        apply_style_param!(applied, effect, motion, swirl_speed);
        apply_style_param!(applied, effect, motion, twist_gain);
        apply_style_param!(applied, effect, motion, twist_speed);
        apply_style_param!(applied, effect, motion, spread_gain);
        apply_style_param!(applied, effect, motion, burnout_gain);
        apply_style_param!(applied, effect, motion, aniso_axis_advect);
        if let Some(amp_over_r0) = motion.meander_amp_over_r0 {
            effect.meander_amp = amp_over_r0 * effect.radius;
            applied.push("meander_amp");
        }
    }

    if groups.texture {
        let texture = &style.texture;
        apply_style_param!(applied, effect, texture, noise_amplitude);
        apply_style_param!(applied, effect, texture, noise_contrast);
        apply_style_param!(applied, effect, texture, noise_frequency);
        apply_style_param!(applied, effect, texture, noise_shaping_scale);
        apply_style_param!(applied, effect, texture, erosion_noise_gain);
        apply_style_param!(applied, effect, texture, support_margin);
        apply_style_param!(applied, effect, texture, contour_wiggle_amp);
        apply_style_param!(applied, effect, texture, edge_outer_sharpen);
        apply_style_param!(applied, effect, texture, noise_scale_mode);
        apply_style_param!(applied, effect, texture, edge_low);
        apply_style_param!(applied, effect, texture, edge_high);
        apply_style_param!(applied, effect, texture, tip_carve_depth);
        apply_style_param!(applied, effect, texture, tip_carve_reach);
        apply_style_param!(applied, effect, texture, warp_reach);
    }

    if groups.optics {
        let optics = &style.optics;
        apply_style_param!(applied, effect, optics, intensity);
        apply_style_param!(applied, effect, optics, temperature_base_k);
        apply_style_param!(applied, effect, optics, temperature_tip_k);
        apply_style_param!(applied, effect, optics, use_blackbody);
        apply_style_param!(applied, effect, optics, white_boost);
        apply_style_param!(applied, effect, optics, self_shadow_strength);
        apply_style_param!(applied, effect, optics, sigma_dispersion);
        apply_style_param!(applied, effect, optics, rte_bands);
        apply_style_param!(applied, effect, optics, edge_temperature_blend);
        apply_style_param!(applied, effect, optics, occlusion_lum_ref);
        apply_style_param!(applied, effect, optics, color_base);
        apply_style_param!(applied, effect, optics, color_tip);
        if let Some(tau0) = optics.tau0 {
            effect.optical_depth = tau0;
            applied.push("optical_depth");
        }
    }

    applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flame::ownership::{
        changed_parameters, flame_parameter_snapshot, parameter_owner, ParameterOwner,
    };
    use std::collections::HashSet;

    fn full_style() -> FlameStyle {
        FlameStyle {
            version: FLAME_STYLE_VERSION,
            name: "test".to_string(),
            motion: FlameStyleMotion {
                noise_scroll_speed: Some(2.0),
                rise_speed: Some(1.1),
                warp_amp: Some(0.9),
                warp_freq: Some(4.5),
                swirl_gain: Some(0.7),
                swirl_speed: Some(1.3),
                twist_gain: Some(6.0),
                twist_speed: Some(1.0),
                spread_gain: Some(0.4),
                meander_amp_over_r0: Some(0.6),
                burnout_gain: Some(2.0),
                aniso_axis_advect: Some(1.0),
            },
            texture: FlameStyleTexture {
                noise_amplitude: Some(6.0),
                noise_contrast: Some(4.0),
                noise_frequency: Some(7.0),
                noise_shaping_scale: Some(0.45),
                erosion_noise_gain: Some(1.0),
                support_margin: Some(2.0),
                contour_wiggle_amp: Some(0.5),
                edge_outer_sharpen: Some(0.2),
                noise_scale_mode: Some(1.0),
                edge_low: Some(0.25),
                edge_high: Some(0.75),
                tip_carve_depth: Some(1.5),
                tip_carve_reach: Some(0.3),
                warp_reach: Some(0.4),
            },
            optics: FlameStyleOptics {
                tau0: Some(4.0),
                intensity: Some(1.5),
                temperature_base_k: Some(1900.0),
                temperature_tip_k: Some(1350.0),
                use_blackbody: Some(true),
                white_boost: Some(0.5),
                self_shadow_strength: Some(0.3),
                sigma_dispersion: Some(0.8),
                rte_bands: Some(4.0),
                edge_temperature_blend: Some(0.1),
                occlusion_lum_ref: Some(0.9),
                color_base: Some([1.0, 0.4, 0.1]),
                color_tip: Some([1.0, 0.1, 0.0]),
            },
        }
    }

    #[test]
    fn test_empty_style_is_a_noop() {
        let mut effect = FlameEffect::default();
        let before = flame_parameter_snapshot(&effect);
        let applied =
            apply_flame_style(&mut effect, &FlameStyle::default(), StyleGroups::default());
        assert!(applied.is_empty());
        assert!(changed_parameters(&before, &flame_parameter_snapshot(&effect)).is_empty());
    }

    #[test]
    fn test_apply_is_idempotent() {
        let mut once = FlameEffect::default();
        apply_flame_style(&mut once, &full_style(), StyleGroups::default());
        let mut twice = once.clone();
        apply_flame_style(&mut twice, &full_style(), StyleGroups::default());
        assert_eq!(once, twice);
    }

    #[test]
    fn test_style_writes_only_style_owned_parameters() {
        let mut effect = FlameEffect::default();
        let before = flame_parameter_snapshot(&effect);
        let applied = apply_flame_style(&mut effect, &full_style(), StyleGroups::default());
        let changed = changed_parameters(&before, &flame_parameter_snapshot(&effect));
        for name in &changed {
            assert_eq!(
                parameter_owner(name),
                Some(ParameterOwner::Style),
                "style wrote non-Style parameter {name}"
            );
        }
        let applied_set: HashSet<&str> = applied.into_iter().collect();
        for name in &changed {
            assert!(applied_set.contains(name), "unreported write {name}");
        }
    }

    #[test]
    fn test_full_style_covers_every_style_parameter_except_sigma_t() {
        let mut effect = FlameEffect::default();
        let applied: HashSet<&str> =
            apply_flame_style(&mut effect, &full_style(), StyleGroups::default())
                .into_iter()
                .collect();
        let expected: HashSet<&str> =
            crate::flame::ownership::parameters_owned_by(ParameterOwner::Style)
                .into_iter()
                .filter(|name| *name != "sigma_t")
                .collect();
        assert_eq!(applied, expected);
    }

    #[test]
    fn test_groups_gate_their_parameters() {
        let mut effect = FlameEffect::default();
        let applied = apply_flame_style(
            &mut effect,
            &full_style(),
            StyleGroups {
                motion: true,
                texture: false,
                optics: false,
            },
        );
        assert!(applied.contains(&"twist_gain"));
        assert!(!applied.contains(&"noise_amplitude"));
        assert!(!applied.contains(&"optical_depth"));
    }

    #[test]
    fn test_style_transfers_across_scale() {
        let style = full_style();
        for radius in [0.5_f32, 2.0] {
            let mut effect = FlameEffect::default();
            effect.radius = radius;
            apply_flame_style(&mut effect, &style, StyleGroups::default());
            assert!((effect.meander_amp / radius - 0.6).abs() < 1e-6);
            let tau = crate::flame::effective_sigma_t(&effect) * radius;
            assert!((tau - 4.0).abs() < 1e-5);
        }
    }
}
