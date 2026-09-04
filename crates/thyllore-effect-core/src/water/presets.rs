use crate::water::WaterTorusEffect;
use cgmath::{Quaternion, Vector3};

pub const WATER_PRESET_NAMES: &[&str] = &["pond", "sea", "glass"];

fn runtime_state(effect: &WaterTorusEffect) -> (Vector3<f32>, Quaternion<f32>, f32) {
    (effect.position, effect.rotation, effect.time)
}

fn apply_runtime_state(effect: &mut WaterTorusEffect, state: (Vector3<f32>, Quaternion<f32>, f32)) {
    let (position, rotation, time) = state;
    effect.position = position;
    effect.rotation = rotation;
    effect.time = time;
}

pub fn apply_water_preset(effect: &mut WaterTorusEffect, name: &str) -> bool {
    let state = runtime_state(effect);
    let mut preset = WaterTorusEffect::default();

    match name {
        "pond" => {}
        "sea" => {
            preset.major_radius = 2.0;
            preset.minor_radius = 0.5;
            preset.absorption = [0.1, 0.3, 0.4];
            preset.flow_longitudinal = 0.5;
            preset.wave_amplitude = 0.05;
            preset.wave_frequency = 4.0;
            preset.tint = [0.0, 0.1, 0.2];
        }
        "glass" => {
            preset.major_radius = 0.5;
            preset.minor_radius = 0.1;
            preset.ior = 1.5;
            preset.absorption = [0.0, 0.0, 0.0];
            preset.reflect_strength = 0.8;
            preset.refract_strength = 1.0;
            preset.tint = [0.9, 0.95, 1.0];
        }
        _ => {
            apply_runtime_state(effect, state);
            return false;
        }
    }

    effect.major_radius = preset.major_radius;
    effect.minor_radius = preset.minor_radius;
    effect.ior = preset.ior;
    effect.absorption = preset.absorption;
    effect.flow_longitudinal = preset.flow_longitudinal;
    effect.flow_meridional = preset.flow_meridional;
    effect.wave_amplitude = preset.wave_amplitude;
    effect.wave_frequency = preset.wave_frequency;
    effect.wave_speed = preset.wave_speed;
    effect.wave_dispersion = preset.wave_dispersion;
    effect.wave_lb_blend = preset.wave_lb_blend;
    effect.reflect_strength = preset.reflect_strength;
    effect.refract_strength = preset.refract_strength;
    effect.caustic_strength = preset.caustic_strength;
    effect.light_intensity = preset.light_intensity;
    effect.highlight_sharpness = preset.highlight_sharpness;
    effect.sky_brightness = preset.sky_brightness;
    effect.scatter_strength = preset.scatter_strength;
    effect.scatter_anisotropy = preset.scatter_anisotropy;
    effect.tint = preset.tint;
    apply_runtime_state(effect, state);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_presets_return_true_and_sanity_ranges() {
        for name in WATER_PRESET_NAMES {
            let mut effect = WaterTorusEffect::default();
            let result = apply_water_preset(&mut effect, *name);
            assert!(result, "preset {} should return true", name);
            assert!(
                (0.01..=10.0).contains(&effect.major_radius),
                "{}: major_radius {} out of range [0.01, 10.0]",
                name,
                effect.major_radius
            );
            assert!(
                (0.0..=1.0).contains(&effect.reflect_strength),
                "{}: reflect_strength {} out of range [0.0, 1.0]",
                name,
                effect.reflect_strength
            );
        }
    }

    #[test]
    fn test_idempotency() {
        for name in WATER_PRESET_NAMES {
            let mut effect1 = WaterTorusEffect::default();
            apply_water_preset(&mut effect1, name);

            let mut effect2 = WaterTorusEffect::default();
            apply_water_preset(&mut effect2, name);
            apply_water_preset(&mut effect2, name);

            assert_eq!(effect1, effect2, "preset {} is not idempotent", name);
        }
    }

    #[test]
    fn test_pond_matches_default() {
        let mut effect = WaterTorusEffect::default();
        let default_effect = WaterTorusEffect::default();
        apply_water_preset(&mut effect, "pond");

        assert_eq!(effect.major_radius, default_effect.major_radius);
        assert_eq!(effect.minor_radius, default_effect.minor_radius);
        assert_eq!(effect.ior, default_effect.ior);
        assert_eq!(effect.absorption, default_effect.absorption);
        assert_eq!(effect.flow_longitudinal, default_effect.flow_longitudinal);
        assert_eq!(effect.flow_meridional, default_effect.flow_meridional);
        assert_eq!(effect.wave_amplitude, default_effect.wave_amplitude);
        assert_eq!(effect.wave_frequency, default_effect.wave_frequency);
        assert_eq!(effect.wave_speed, default_effect.wave_speed);
        assert_eq!(effect.reflect_strength, default_effect.reflect_strength);
        assert_eq!(effect.refract_strength, default_effect.refract_strength);
        assert_eq!(effect.caustic_strength, default_effect.caustic_strength);
        assert_eq!(effect.tint, default_effect.tint);
    }

    #[test]
    fn test_preset_preserves_runtime_state() {
        let mut effect = WaterTorusEffect::default();
        effect.position = Vector3::new(1.0, 2.0, 3.0);
        effect.rotation = Quaternion::new(0.99, 0.1, 0.0, 0.0);
        effect.time = 5.0;

        apply_water_preset(&mut effect, "sea");

        assert_eq!(effect.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(effect.rotation, Quaternion::new(0.99, 0.1, 0.0, 0.0));
        assert_eq!(effect.time, 5.0);
    }

    #[test]
    fn test_unknown_preset_returns_false() {
        let mut effect = WaterTorusEffect::default();
        effect.position = Vector3::new(1.0, 2.0, 3.0);
        effect.time = 5.0;

        let result = apply_water_preset(&mut effect, "unknown");
        assert!(!result);
        assert_eq!(effect.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(effect.time, 5.0);
    }
}
