use crate::flame::{refresh_flame_coefficients, FlameBaked, FlameEffect};
use cgmath::{Quaternion, Vector2, Vector3};

pub const FLAME_PRESET_NAMES: &[&str] = &["campfire", "candle", "torch", "inferno", "blue", "ring"];

fn runtime_state(effect: &FlameEffect) -> (Vector3<f32>, Quaternion<f32>, f32) {
    (effect.position, effect.rotation, effect.time)
}

fn apply_runtime_state(effect: &mut FlameEffect, state: (Vector3<f32>, Quaternion<f32>, f32)) {
    let (position, rotation, time) = state;
    effect.position = position;
    effect.rotation = rotation;
    effect.time = time;
}

pub fn apply_flame_preset(effect: &mut FlameEffect, name: &str) -> bool {
    let state = runtime_state(effect);
    let mut preset = FlameEffect::default();

    match name {
        "campfire" => {
            // no changes from default
        }
        "candle" => {
            preset.height = 0.28;
            preset.radius = 0.07;
            preset.intensity = 2.0;
            preset.sigma_t = 1.2;
            preset.color.temperature_base_k = 2900.0;
            preset.color.temperature_tip_k = 1600.0;
            preset.noise.amplitude = 0.5;
            preset.noise.frequency = 8.0;
            preset.warp.rise_speed = 0.6;
            preset.warp.amp = 0.4;
            preset.time_scale = 0.6;
            preset.edge.white_boost = 3.0;
            preset.wind.bend_amount = 0.05;
            preset.radial_sharpness = 8.0;
        }
        "torch" => {
            preset.height = 1.2;
            preset.radius = 0.3;
            preset.intensity = 2.4;
            preset.wind.direction = Vector2::new(0.2, 0.0);
            preset.wind.bend_amount = 0.1;
            preset.noise.amplitude = 1.2;
            preset.warp.amp = 1.2;
            preset.radial_sharpness = 4.0;
        }
        "inferno" => {
            preset.height = 2.2;
            preset.radius = 0.8;
            preset.intensity = 2.2;
            preset.sigma_t = 1.0;
            preset.color.temperature_base_k = 3600.0;
            preset.color.temperature_tip_k = 1400.0;
            preset.noise.amplitude = 2.2;
            preset.noise.frequency = 7.0;
            preset.warp.rise_speed = 2.8;
            preset.warp.amp = 2.4;
            preset.time_scale = 1.6;
            preset.edge.white_boost = 3.5;
            preset.wind.bend_amount = 0.25;
            preset.wind.bend_power = 2.0;
            preset.wind.direction = Vector2::new(0.4, 0.2);
            preset.radial_sharpness = 3.0;
        }
        "blue" => {
            preset.height = 0.5;
            preset.radius = 0.25;
            preset.intensity = 4.0;
            preset.sigma_t = 2.0;
            preset.color.use_blackbody = false;
            // Deep-blue tip: a whiter tip washes the whole flame out after
            // tonemapping (calibrated against flame_blue_ref.png, 2026-08-01).
            preset.color.base = [0.15, 0.35, 1.0];
            preset.color.tip = [0.45, 0.65, 1.0];
            preset.noise.amplitude = 0.7;
            preset.noise.frequency = 7.0;
            preset.warp.rise_speed = 1.0;
            preset.warp.amp = 0.6;
            preset.time_scale = 0.8;
            preset.edge.white_boost = 2.0;
            preset.radial_sharpness = 6.0;
        }
        "ring" => {
            preset.height = 1.2;
            preset.radius = 0.5;
            preset.intensity = 2.4;
            preset.emitter.kind = 1;
            preset.emitter.ring_major_radius = 1.5;
            preset.emitter.ring_angular_speed = 0.6;
            preset.noise.amplitude = 1.6;
        }
        _ => {
            apply_runtime_state(effect, state);
            return false;
        }
    }

    refresh_flame_coefficients(&mut preset, &FlameBaked::default());

    // Copy all fields except runtime state
    effect.height = preset.height;
    effect.radius = preset.radius;
    effect.sigma_t = preset.sigma_t;
    effect.intensity = preset.intensity;
    effect.color.base = preset.color.base;
    effect.color.tip = preset.color.tip;
    effect.color.temperature_base_k = preset.color.temperature_base_k;
    effect.color.temperature_tip_k = preset.color.temperature_tip_k;
    effect.color.use_blackbody = preset.color.use_blackbody;
    effect.noise.amplitude = preset.noise.amplitude;
    effect.noise.frequency = preset.noise.frequency;
    effect.noise.scroll_speed = preset.noise.scroll_speed;
    effect.time_scale = preset.time_scale;
    effect.time_offset = preset.time_offset;
    effect.coefficients = preset.coefficients;
    effect.light_position_world = preset.light_position_world;
    effect.self_shadow_strength = preset.self_shadow_strength;
    effect.warp.amp = preset.warp.amp;
    effect.warp.freq = preset.warp.freq;
    effect.warp.rise_speed = preset.warp.rise_speed;
    effect.warp.taper_power = preset.warp.taper_power;
    effect.edge.radius_tip_ratio = preset.edge.radius_tip_ratio;
    effect.edge.low = preset.edge.low;
    effect.edge.high = preset.edge.high;
    effect.edge.white_boost = preset.edge.white_boost;
    effect.wind.direction = preset.wind.direction;
    effect.wind.bend_amount = preset.wind.bend_amount;
    effect.wind.bend_power = preset.wind.bend_power;
    effect.envelope.peak = preset.envelope.peak;
    effect.envelope.base = preset.envelope.base;
    effect.envelope.tail = preset.envelope.tail;
    effect.radial_sharpness = preset.radial_sharpness;
    effect.noise.aniso_y = preset.noise.aniso_y;
    effect.warp.y_scale = preset.warp.y_scale;
    effect.emitter.kind = preset.emitter.kind;
    effect.emitter.ring_major_radius = preset.emitter.ring_major_radius;
    effect.emitter.ring_angular_speed = preset.emitter.ring_angular_speed;
    effect.color.occlusion_lum_ref = preset.color.occlusion_lum_ref;
    effect.contour.wiggle_amp = preset.contour.wiggle_amp;
    effect.boundary = preset.boundary;

    apply_runtime_state(effect, state);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_presets_return_true_and_sanity_ranges() {
        for name in FLAME_PRESET_NAMES {
            let mut effect = FlameEffect::default();
            let result = apply_flame_preset(&mut effect, *name);
            assert!(result, "preset {} should return true", name);
            assert!(
                (0.05..=4.0).contains(&effect.height),
                "{}: height {} out of range [0.05, 4.0]",
                name,
                effect.height
            );
            assert!(
                (0.0..=8.0).contains(&effect.intensity),
                "{}: intensity {} out of range [0.0, 8.0]",
                name,
                effect.intensity
            );
            assert!(
                (0.02..=2.0).contains(&effect.radius),
                "{}: radius {} out of range [0.02, 2.0]",
                name,
                effect.radius
            );
        }
    }

    #[test]
    fn test_idempotency() {
        for name in FLAME_PRESET_NAMES {
            let mut effect1 = FlameEffect::default();
            apply_flame_preset(&mut effect1, name);

            let mut effect2 = FlameEffect::default();
            apply_flame_preset(&mut effect2, name);
            apply_flame_preset(&mut effect2, name);

            assert_eq!(effect1, effect2, "preset {} is not idempotent", name);
        }
    }

    #[test]
    fn test_campfire_matches_default() {
        let mut effect = FlameEffect::default();
        let default_effect = FlameEffect::default();
        apply_flame_preset(&mut effect, "campfire");

        assert_eq!(effect.height, default_effect.height);
        assert_eq!(effect.radius, default_effect.radius);
        assert_eq!(effect.sigma_t, default_effect.sigma_t);
        assert_eq!(effect.intensity, default_effect.intensity);
        assert_eq!(effect.color.base, default_effect.color.base);
        assert_eq!(effect.color.tip, default_effect.color.tip);
        assert_eq!(
            effect.color.temperature_base_k,
            default_effect.color.temperature_base_k
        );
        assert_eq!(
            effect.color.temperature_tip_k,
            default_effect.color.temperature_tip_k
        );
        assert_eq!(
            effect.color.use_blackbody,
            default_effect.color.use_blackbody
        );
        assert_eq!(effect.noise.amplitude, default_effect.noise.amplitude);
        assert_eq!(effect.noise.frequency, default_effect.noise.frequency);
        assert_eq!(effect.noise.scroll_speed, default_effect.noise.scroll_speed);
        assert_eq!(effect.time_scale, default_effect.time_scale);
        assert_eq!(effect.time_offset, default_effect.time_offset);
        assert_eq!(effect.coefficients, default_effect.coefficients);
        assert_eq!(
            effect.light_position_world,
            default_effect.light_position_world
        );
        assert_eq!(
            effect.self_shadow_strength,
            default_effect.self_shadow_strength
        );
        assert_eq!(effect.warp.amp, default_effect.warp.amp);
        assert_eq!(effect.warp.freq, default_effect.warp.freq);
        assert_eq!(effect.warp.rise_speed, default_effect.warp.rise_speed);
        assert_eq!(effect.warp.taper_power, default_effect.warp.taper_power);
        assert_eq!(
            effect.edge.radius_tip_ratio,
            default_effect.edge.radius_tip_ratio
        );
        assert_eq!(effect.edge.low, default_effect.edge.low);
        assert_eq!(effect.edge.high, default_effect.edge.high);
        assert_eq!(effect.edge.white_boost, default_effect.edge.white_boost);
        assert_eq!(effect.wind.direction, default_effect.wind.direction);
        assert_eq!(effect.wind.bend_amount, default_effect.wind.bend_amount);
        assert_eq!(effect.wind.bend_power, default_effect.wind.bend_power);
        assert_eq!(effect.envelope.peak, default_effect.envelope.peak);
        assert_eq!(effect.envelope.base, default_effect.envelope.base);
        assert_eq!(effect.envelope.tail, default_effect.envelope.tail);
        assert_eq!(effect.radial_sharpness, default_effect.radial_sharpness);
        assert_eq!(effect.noise.aniso_y, default_effect.noise.aniso_y);
        assert_eq!(effect.warp.y_scale, default_effect.warp.y_scale);
        assert_eq!(effect.emitter.kind, default_effect.emitter.kind);
        assert_eq!(
            effect.emitter.ring_major_radius,
            default_effect.emitter.ring_major_radius
        );
        assert_eq!(
            effect.emitter.ring_angular_speed,
            default_effect.emitter.ring_angular_speed
        );
        assert_eq!(
            effect.color.occlusion_lum_ref,
            default_effect.color.occlusion_lum_ref
        );
        assert_eq!(effect.contour.wiggle_amp, default_effect.contour.wiggle_amp);
    }
}
