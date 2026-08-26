use serde::{Deserialize, Serialize};

use super::FlameEffect;

pub const FLAME_STYLE_VERSION: u32 = 1;

/// One declaration per style group: the serde struct, its extractor, and its
/// applier are all generated from the same field list, so the three cannot
/// drift. `direct` fields map 1:1 onto an effect field path; `custom` fields
/// need a conversion and are handled by the callers in
/// `flame_style_from_effect` / `apply_flame_style`.
macro_rules! declare_style_group {
    (
        $(#[$doc:meta])*
        $struct_name:ident, $extract_fn:ident, $apply_fn:ident, $effect:ident {
            direct {
                $( $field:ident : $ty:ty => $($path:ident).+ ),* $(,)?
            }
            custom {
                $( $custom_field:ident : $custom_ty:ty ),* $(,)?
            }
        }
    ) => {
        $(#[$doc])*
        #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
        #[serde(default)]
        pub struct $struct_name {
            $( pub $field: Option<$ty>, )*
            $( pub $custom_field: Option<$custom_ty>, )*
        }

        fn $extract_fn($effect: &FlameEffect) -> $struct_name {
            $struct_name {
                $( $field: Some($effect.$($path).+), )*
                ..Default::default()
            }
        }

        fn $apply_fn(
            $effect: &mut FlameEffect,
            group: &$struct_name,
            applied: &mut Vec<&'static str>,
        ) {
            $(
                if let Some(value) = group.$field {
                    $effect.$($path).+ = value;
                    applied.push(stringify!($field));
                }
            )*
        }
    };
}

declare_style_group! {
    FlameStyleMotion, extract_motion_style, apply_motion_style, effect {
        direct {
            noise_scroll_speed: f32 => noise.scroll_speed,
            rise_speed: f32 => warp.rise_speed,
            rise_accel: f32 => warp.rise_accel,
            warp_amp: f32 => warp.amp,
            warp_freq: f32 => warp.freq,
            swirl_gain: f32 => swirl.gain,
            swirl_speed: f32 => swirl.speed,
            twist_gain: f32 => twist.gain,
            twist_speed: f32 => twist.speed,
            spread_gain: f32 => spread_gain,
            burnout_gain: f32 => carve.burnout_gain,
            meander_frequency: f32 => meander.frequency,
            aniso_axis_advect: f32 => contour.aniso_axis_advect,
            branch_period: f32 => branch.period,
            branch_life: f32 => branch.life,
            branch_gain: f32 => branch.gain,
            branch_core_radius: f32 => branch.core_radius,
            branch_core_offset: f32 => branch.core_offset,
            branch_reach: f32 => branch.reach,
            branch_spread: f32 => branch.spread,
            branch_spawn_height: f32 => branch.spawn_height,
            branch_spawn_range: f32 => branch.spawn_range,
            puff_gain: f32 => puff.gain,
            puff_period: f32 => puff.period,
            puff_rise: f32 => puff.rise,
            puff_radius: f32 => puff.radius,
            puff_spread: f32 => puff.spread,
            puff_decay: f32 => puff.decay,
            puff_aspect: f32 => puff.aspect,
            puff_spawn_height: f32 => puff.spawn_height,
            puff_root_gain: f32 => puff.root_gain,
            puff_root_height: f32 => puff.root_height,
            flow_gain: f32 => flow.gain,
            flow_period: f32 => flow.period,
            flow_rise: f32 => flow.rise,
            flow_strength: f32 => flow.strength,
            flow_core: f32 => flow.core,
            flow_gust: f32 => flow.gust,
            flow_gust_frequency: f32 => flow.gust_frequency,
            flow_burst: f32 => flow.burst,
            flow_damping: f32 => flow.damping,
            flow_damping_slope: f32 => flow.damping_slope,
           flow_transport_speed: f32 => flow.transport_speed,
            flow_transport_accel: f32 => flow.transport_accel,
            flow_inject_height: f32 => flow.inject_height,
            lobe_gain: f32 => lobe.gain,
            lobe_period: f32 => lobe.period,
            lobe_life: f32 => lobe.life,
            lobe_rise: f32 => lobe.rise,
            lobe_size: f32 => lobe.size,
            lobe_spawn_height: f32 => lobe.spawn_height,
            lobe_spawn_range: f32 => lobe.spawn_range,
            lobe_accel: f32 => lobe.accel,
            lobe_spread: f32 => lobe.spread,
            lobe_shift: f32 => lobe.shift,
            lobe_transport: f32 => lobe.transport,
        }
        custom {
            meander_amp_over_r0: f32,
        }
    }
}

declare_style_group! {
    FlameStyleTexture, extract_texture_style, apply_texture_style, effect {
        direct {
            noise_amplitude: f32 => noise.amplitude,
            noise_contrast: f32 => noise.contrast,
            noise_frequency: f32 => noise.frequency,
            noise_aniso_y: f32 => noise.aniso_y,
            noise_lobe_scale: f32 => noise.lobe_scale,
            noise_lobe_aniso: f32 => noise.lobe_aniso,
            mix_lo: f32 => mix.lo,
            mix_hi: f32 => mix.hi,
            mix_height_gain: f32 => mix.height_gain,
            mix_scale: f32 => mix.scale,
            mix_radial_gain: f32 => mix.radial_gain,
            mix_core_radius: f32 => mix.core_radius,
            noise_shaping_scale: f32 => noise.shaping_scale,
            erosion_noise_gain: f32 => noise.erosion_gain,
            support_margin: f32 => support_margin,
            contour_wiggle_amp: f32 => contour.wiggle_amp,
            edge_outer_sharpen: f32 => edge.outer_sharpen,
            base_spread: f32 => edge.base_spread,
            base_spread_height: f32 => edge.base_spread_height,
            noise_scale_mode: f32 => noise.scale_mode,
            edge_low: f32 => edge.low,
            edge_high: f32 => edge.high,
            tip_carve_depth: f32 => carve.tip.depth,
            tip_carve_reach: f32 => carve.tip.reach,
            warp_reach: f32 => warp.reach,
        }
        custom {}
    }
}

declare_style_group! {
    FlameStyleOptics, extract_optics_style, apply_optics_style, effect {
        direct {
            intensity: f32 => intensity,
            temperature_base_k: f32 => color.temperature_base_k,
            temperature_tip_k: f32 => color.temperature_tip_k,
            use_blackbody: bool => color.use_blackbody,
            white_boost: f32 => edge.white_boost,
            density_exp: f32 => thermal.density_exp,
            temp_exp: f32 => thermal.temp_exp,
            wien_c_k: f32 => thermal.wien_c_k,
            self_shadow_strength: f32 => self_shadow_strength,
            sigma_dispersion: f32 => contour.sigma_dispersion,
            rte_bands: f32 => contour.rte_bands,
            occlusion_lum_ref: f32 => color.occlusion_lum_ref,
            color_base: [f32; 3] => color.base,
            color_tip: [f32; 3] => color.tip,
        }
        custom {
            tau0: f32,
        }
    }
}

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

/// Extract the effect's current look as a fully-populated style: the inverse
/// of `apply_flame_style`, so hand-tuned parameters can be saved and reapplied
/// to a flame of any size. tau0 falls back to sigma_t * radius when the
/// optical_depth parameter is not in use.
pub fn flame_style_from_effect(effect: &FlameEffect, name: &str) -> FlameStyle {
    let radius = effect.radius.max(1e-4);
    FlameStyle {
        version: FLAME_STYLE_VERSION,
        name: name.to_string(),
        motion: FlameStyleMotion {
            meander_amp_over_r0: Some(effect.meander.amp / radius),
            ..extract_motion_style(effect)
        },
        texture: extract_texture_style(effect),
        optics: FlameStyleOptics {
            tau0: Some(if effect.optical_depth > 0.0 {
                effect.optical_depth
            } else {
                effect.sigma_t * radius
            }),
            ..extract_optics_style(effect)
        },
    }
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
        apply_motion_style(effect, &style.motion, &mut applied);
        if let Some(amp_over_r0) = style.motion.meander_amp_over_r0 {
            effect.meander.amp = amp_over_r0 * effect.radius;
            applied.push("meander_amp");
        }
    }

    if groups.texture {
        apply_texture_style(effect, &style.texture, &mut applied);
    }

    if groups.optics {
        apply_optics_style(effect, &style.optics, &mut applied);
        if let Some(tau0) = style.optics.tau0 {
            effect.optical_depth = tau0;
            applied.push("optical_depth");
        }
    }

    applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flame::flame_parameter_snapshot;
    use crate::flame::ownership::{changed_parameters, parameter_owner, ParameterOwner};
    use std::collections::HashSet;

    fn full_style() -> FlameStyle {
        FlameStyle {
            version: FLAME_STYLE_VERSION,
            name: "test".to_string(),
            motion: FlameStyleMotion {
                noise_scroll_speed: Some(2.0),
                rise_speed: Some(1.1),
                rise_accel: Some(0.0),
                warp_amp: Some(0.9),
                warp_freq: Some(4.5),
                swirl_gain: Some(0.7),
                swirl_speed: Some(1.3),
                twist_gain: Some(6.0),
                twist_speed: Some(1.0),
                spread_gain: Some(0.4),
                meander_amp_over_r0: Some(0.6),
                burnout_gain: Some(2.0),
                meander_frequency: Some(1.0),
                aniso_axis_advect: Some(1.0),
                branch_period: Some(0.5),
                branch_life: Some(2.0),
                branch_gain: Some(1.5),
                branch_core_radius: Some(0.8),
                branch_core_offset: Some(1.0),
                branch_reach: Some(2.0),
                branch_spread: Some(0.4),
                branch_spawn_height: Some(0.5),
                branch_spawn_range: Some(1.0),
                puff_gain: Some(1.5),
                puff_period: Some(0.4),
                puff_rise: Some(0.25),
                puff_radius: Some(0.7),
                puff_spread: Some(0.6),
                puff_decay: Some(1.0),
                puff_aspect: Some(0.5),
                puff_spawn_height: Some(0.0),
                puff_root_gain: Some(0.0),
                puff_root_height: Some(0.0),
                flow_gain: Some(1.0),
                flow_period: Some(1.0),
                flow_rise: Some(0.3),
                flow_strength: Some(1.0),
                flow_core: Some(0.6),
                flow_gust: Some(0.3),
                flow_gust_frequency: Some(0.4),
                flow_burst: Some(0.5),
                flow_damping: Some(0.5),
                flow_damping_slope: Some(0.0),
                flow_transport_speed: Some(0.0),
                flow_transport_accel: Some(0.0),
                flow_inject_height: Some(0.0),
                lobe_gain: Some(0.5),
                lobe_period: Some(0.4),
                lobe_life: Some(2.0),
                lobe_rise: Some(0.1),
                lobe_size: Some(0.08),
                lobe_spawn_height: Some(0.2),
                lobe_spawn_range: Some(0.0),
                lobe_accel: Some(0.0),
                lobe_shift: Some(1.0),
                lobe_spread: Some(0.5),
                lobe_transport: Some(0.0),
            },
            texture: FlameStyleTexture {
                noise_amplitude: Some(6.0),
                noise_contrast: Some(4.0),
                noise_frequency: Some(7.0),
                noise_aniso_y: Some(0.5),
                mix_lo: Some(0.2),
                mix_hi: Some(1.8),
                mix_height_gain: Some(0.3),
                mix_scale: Some(0.5),
                mix_radial_gain: Some(0.5),
                mix_core_radius: Some(0.6),
                noise_lobe_scale: Some(0.4),
                noise_lobe_aniso: Some(0.8),
                noise_shaping_scale: Some(0.45),
                erosion_noise_gain: Some(1.0),
                support_margin: Some(2.0),
                contour_wiggle_amp: Some(0.5),
                edge_outer_sharpen: Some(0.2),
                base_spread: Some(0.8),
                base_spread_height: Some(0.3),
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
                density_exp: Some(1.2),
                temp_exp: Some(1.5),
                wien_c_k: Some(10000.0),
                self_shadow_strength: Some(0.3),
                sigma_dispersion: Some(0.8),
                rte_bands: Some(4.0),
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
    fn test_extract_apply_roundtrip_preserves_the_look() {
        let mut source = FlameEffect::default();
        apply_flame_style(&mut source, &full_style(), StyleGroups::default());
        let extracted = flame_style_from_effect(&source, "roundtrip");

        let mut target = FlameEffect::default();
        target.radius = source.radius;
        apply_flame_style(&mut target, &extracted, StyleGroups::default());

        let source_snapshot = flame_parameter_snapshot(&source);
        let target_snapshot = flame_parameter_snapshot(&target);
        for ((name, s), (_, t)) in source_snapshot.iter().zip(&target_snapshot) {
            if parameter_owner(name) == Some(ParameterOwner::Style) && *name != "sigma_t" {
                assert_eq!(s, t, "{name}");
            }
        }
    }

    #[test]
    fn test_apply_extract_roundtrip_returns_the_style() {
        let mut effect = FlameEffect::default();
        apply_flame_style(&mut effect, &full_style(), StyleGroups::default());
        let mut extracted = flame_style_from_effect(&effect, "test");
        extracted.version = full_style().version;
        assert_eq!(extracted, full_style());
    }

    #[test]
    fn test_style_transfers_across_scale() {
        let style = full_style();
        for radius in [0.5_f32, 2.0] {
            let mut effect = FlameEffect::default();
            effect.radius = radius;
            apply_flame_style(&mut effect, &style, StyleGroups::default());
            assert!((effect.meander.amp / radius - 0.6).abs() < 1e-6);
            let tau = crate::flame::effective_sigma_t(&effect) * radius;
            assert!((tau - 4.0).abs() < 1e-5);
        }
    }
}
