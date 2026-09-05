/// Distance in meters the absorption colour picker shows the transmitted colour over.
pub const ABSORPTION_REFERENCE_DISTANCE: f32 = 1.0;

/// Darkest picker colour still mapped back to a finite coefficient.
pub const ABSORPTION_COLOR_FLOOR: f32 = 1e-3;

/// Beer-Lambert transmittance `exp(-absorption * distance)` per channel.
pub fn absorption_to_transmitted_color(absorption: [f32; 3], distance: f32) -> [f32; 3] {
    absorption.map(|coefficient| (-coefficient * distance).exp())
}

pub fn transmitted_color_to_absorption(color: [f32; 3], distance: f32) -> [f32; 3] {
    color.map(|channel| -channel.max(ABSORPTION_COLOR_FLOOR).ln() / distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_to_absorption_roundtrip_within_tolerance() {
        let absorption = [0.35, 0.08, 0.02];
        let color = absorption_to_transmitted_color(absorption, ABSORPTION_REFERENCE_DISTANCE);
        let restored = transmitted_color_to_absorption(color, ABSORPTION_REFERENCE_DISTANCE);
        for channel in 0..3 {
            assert!((restored[channel] - absorption[channel]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_zero_absorption_is_white_and_black_is_clamped() {
        assert_eq!(absorption_to_transmitted_color([0.0; 3], 1.0), [1.0; 3]);
        let clamped = transmitted_color_to_absorption([0.0; 3], 1.0);
        assert!(clamped.iter().all(|coefficient| coefficient.is_finite()));
    }
}
