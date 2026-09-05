use crate::wind::{overwrite_wind_persisted_fields, WindTornadoEffect};

pub const WIND_PRESET_NAMES: &[&str] = &["column", "funnel"];

pub fn apply_wind_preset(effect: &mut WindTornadoEffect, name: &str) -> bool {
    let mut preset = WindTornadoEffect::default();
    match name {
        "column" => {}
        "funnel" => {
            preset.column_height = 3.0;
            preset.wall_radius_base = 0.25;
            preset.wall_radius_top = 1.2;
            preset.wall_width_q = 0.05;
            preset.core_strength = 0.2;
            preset.top_fade = 0.4;
        }
        _ => return false,
    }

    preset.position = effect.position;
    preset.rotation = effect.rotation;
    overwrite_wind_persisted_fields(effect, &preset);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Quaternion, Vector3};

    #[test]
    fn test_all_presets_apply_and_are_idempotent() {
        for name in WIND_PRESET_NAMES {
            let mut once = WindTornadoEffect::default();
            assert!(apply_wind_preset(&mut once, name), "{name}");
            let mut twice = WindTornadoEffect::default();
            apply_wind_preset(&mut twice, name);
            apply_wind_preset(&mut twice, name);
            assert_eq!(once, twice, "{name}");
        }
    }

    #[test]
    fn test_preset_preserves_runtime_state() {
        let mut effect = WindTornadoEffect::default();
        effect.position = Vector3::new(1.0, 2.0, 3.0);
        effect.rotation = Quaternion::new(0.99, 0.1, 0.0, 0.0);
        effect.time = 5.0;

        apply_wind_preset(&mut effect, "funnel");

        assert_eq!(effect.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(effect.rotation, Quaternion::new(0.99, 0.1, 0.0, 0.0));
        assert_eq!(effect.time, 5.0);
        assert_eq!(effect.wall_radius_top, 1.2);
    }

    #[test]
    fn test_unknown_preset_leaves_effect_untouched() {
        let mut effect = WindTornadoEffect::default();
        effect.column_height = 7.0;
        assert!(!apply_wind_preset(&mut effect, "unknown"));
        assert_eq!(effect.column_height, 7.0);
    }
}
