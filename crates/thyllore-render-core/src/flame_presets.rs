use crate::flame::{refresh_flame_coefficients, FlameEffect};
use cgmath::{Quaternion, Vector2, Vector3};

pub const FLAME_PRESET_NAMES: &[&str] = &["campfire", "candle", "torch", "inferno", "blue", "ring"];

fn runtime_state(effect: &FlameEffect) -> (Vector3<f32>, Quaternion<f32>, f32, u64) {
    (
        effect.position,
        effect.rotation,
        effect.time,
        effect.frame_index,
    )
}

fn apply_runtime_state(effect: &mut FlameEffect, state: (Vector3<f32>, Quaternion<f32>, f32, u64)) {
    let (position, rotation, time, frame_index) = state;
    effect.position = position;
    effect.rotation = rotation;
    effect.time = time;
    effect.frame_index = frame_index;
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
            preset.temperature_base_k = 2900.0;
            preset.temperature_tip_k = 1600.0;
            preset.noise_amplitude = 0.5;
            preset.noise_frequency = 8.0;
            preset.rise_speed = 0.6;
            preset.warp_amp = 0.4;
            preset.time_scale = 0.6;
            preset.white_boost = 3.0;
            preset.bend_amount = 0.05;
            preset.radial_sharpness = 8.0;
        }
        "torch" => {
            preset.height = 1.2;
            preset.radius = 0.3;
            preset.intensity = 2.4;
            preset.wind_direction = Vector2::new(0.2, 0.0);
            preset.bend_amount = 0.1;
            preset.noise_amplitude = 1.2;
            preset.warp_amp = 1.2;
            preset.radial_sharpness = 4.0;
        }
        "inferno" => {
            preset.height = 2.2;
            preset.radius = 0.8;
            preset.intensity = 2.2;
            preset.sigma_t = 1.0;
            preset.temperature_base_k = 3600.0;
            preset.temperature_tip_k = 1400.0;
            preset.noise_amplitude = 2.2;
            preset.noise_frequency = 7.0;
            preset.rise_speed = 2.8;
            preset.warp_amp = 2.4;
            preset.time_scale = 1.6;
            preset.white_boost = 3.5;
            preset.bend_amount = 0.25;
            preset.bend_power = 2.0;
            preset.wind_direction = Vector2::new(0.4, 0.2);
            preset.radial_sharpness = 3.0;
        }
        "blue" => {
            preset.height = 0.5;
            preset.radius = 0.25;
            preset.intensity = 4.0;
            preset.sigma_t = 2.0;
            preset.use_blackbody = false;
            preset.color_base = [0.25, 0.5, 1.0];
            preset.color_tip = [0.7, 0.85, 1.0];
            preset.noise_amplitude = 0.7;
            preset.noise_frequency = 7.0;
            preset.rise_speed = 1.0;
            preset.warp_amp = 0.6;
            preset.time_scale = 0.8;
            preset.white_boost = 2.0;
            preset.radial_sharpness = 6.0;
        }
        "ring" => {
            preset.height = 1.2;
            preset.radius = 0.5;
            preset.intensity = 2.4;
            preset.emitter_kind = 1;
            preset.ring_major_radius = 1.5;
            preset.ring_angular_speed = 0.6;
            preset.noise_amplitude = 1.6;
        }
        _ => {
            apply_runtime_state(effect, state);
            return false;
        }
    }

    refresh_flame_coefficients(&mut preset);

    // Copy all fields except runtime state
    effect.height = preset.height;
    effect.radius = preset.radius;
    effect.sigma_t = preset.sigma_t;
    effect.intensity = preset.intensity;
    effect.color_base = preset.color_base;
    effect.color_tip = preset.color_tip;
    effect.temperature_base_k = preset.temperature_base_k;
    effect.temperature_tip_k = preset.temperature_tip_k;
    effect.use_blackbody = preset.use_blackbody;
    effect.noise_amplitude = preset.noise_amplitude;
    effect.noise_frequency = preset.noise_frequency;
    effect.noise_scroll_speed = preset.noise_scroll_speed;
    effect.time_scale = preset.time_scale;
    effect.time_offset = preset.time_offset;
    effect.coefficients = preset.coefficients;
    effect.temporal_weight = preset.temporal_weight;
    effect.light_position_world = preset.light_position_world;
    effect.self_shadow_strength = preset.self_shadow_strength;
    effect.warp_amp = preset.warp_amp;
    effect.warp_freq = preset.warp_freq;
    effect.rise_speed = preset.rise_speed;
    effect.taper_power = preset.taper_power;
    effect.radius_tip_ratio = preset.radius_tip_ratio;
    effect.edge_low = preset.edge_low;
    effect.edge_high = preset.edge_high;
    effect.white_boost = preset.white_boost;
    effect.wind_direction = preset.wind_direction;
    effect.bend_amount = preset.bend_amount;
    effect.bend_power = preset.bend_power;
    effect.envelope_peak = preset.envelope_peak;
    effect.envelope_base = preset.envelope_base;
    effect.envelope_tail = preset.envelope_tail;
    effect.radial_sharpness = preset.radial_sharpness;
    effect.noise_aniso_y = preset.noise_aniso_y;
    effect.warp_y_scale = preset.warp_y_scale;
    effect.emitter_kind = preset.emitter_kind;
    effect.ring_major_radius = preset.ring_major_radius;
    effect.ring_angular_speed = preset.ring_angular_speed;
    effect.occlusion_lum_ref = preset.occlusion_lum_ref;
    effect.contour_wiggle_amp = preset.contour_wiggle_amp;

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
        assert_eq!(effect.color_base, default_effect.color_base);
        assert_eq!(effect.color_tip, default_effect.color_tip);
        assert_eq!(effect.temperature_base_k, default_effect.temperature_base_k);
        assert_eq!(effect.temperature_tip_k, default_effect.temperature_tip_k);
        assert_eq!(effect.use_blackbody, default_effect.use_blackbody);
        assert_eq!(effect.noise_amplitude, default_effect.noise_amplitude);
        assert_eq!(effect.noise_frequency, default_effect.noise_frequency);
        assert_eq!(effect.noise_scroll_speed, default_effect.noise_scroll_speed);
        assert_eq!(effect.time_scale, default_effect.time_scale);
        assert_eq!(effect.time_offset, default_effect.time_offset);
        assert_eq!(effect.coefficients, default_effect.coefficients);
        assert_eq!(effect.temporal_weight, default_effect.temporal_weight);
        assert_eq!(
            effect.light_position_world,
            default_effect.light_position_world
        );
        assert_eq!(
            effect.self_shadow_strength,
            default_effect.self_shadow_strength
        );
        assert_eq!(effect.warp_amp, default_effect.warp_amp);
        assert_eq!(effect.warp_freq, default_effect.warp_freq);
        assert_eq!(effect.rise_speed, default_effect.rise_speed);
        assert_eq!(effect.taper_power, default_effect.taper_power);
        assert_eq!(effect.radius_tip_ratio, default_effect.radius_tip_ratio);
        assert_eq!(effect.edge_low, default_effect.edge_low);
        assert_eq!(effect.edge_high, default_effect.edge_high);
        assert_eq!(effect.white_boost, default_effect.white_boost);
        assert_eq!(effect.wind_direction, default_effect.wind_direction);
        assert_eq!(effect.bend_amount, default_effect.bend_amount);
        assert_eq!(effect.bend_power, default_effect.bend_power);
        assert_eq!(effect.envelope_peak, default_effect.envelope_peak);
        assert_eq!(effect.envelope_base, default_effect.envelope_base);
        assert_eq!(effect.envelope_tail, default_effect.envelope_tail);
        assert_eq!(effect.radial_sharpness, default_effect.radial_sharpness);
        assert_eq!(effect.noise_aniso_y, default_effect.noise_aniso_y);
        assert_eq!(effect.warp_y_scale, default_effect.warp_y_scale);
        assert_eq!(effect.emitter_kind, default_effect.emitter_kind);
        assert_eq!(effect.ring_major_radius, default_effect.ring_major_radius);
        assert_eq!(effect.ring_angular_speed, default_effect.ring_angular_speed);
        assert_eq!(effect.occlusion_lum_ref, default_effect.occlusion_lum_ref);
        assert_eq!(effect.contour_wiggle_amp, default_effect.contour_wiggle_amp);
    }
}
