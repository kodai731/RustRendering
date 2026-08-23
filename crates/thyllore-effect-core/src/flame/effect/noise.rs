use crate::flame::*;
use crate::flame_wave::{WaveLobeShape, WAVE_LOBE_SCALE_DEFAULT};

/// Erosion noise of the medium.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameNoise {
    pub amplitude: f32,
    /// Scales the edge smoothstep window half-width as hw0 / contrast; 1.0 keeps the authored window.
    pub contrast: f32,
    pub frequency: f32,
    pub scroll_speed: f32,
    pub aniso_y: f32,
    /// < 0.5 uses `aniso_y` as-is, >= 0.5 multiplies it by height / radius.
    pub scale_mode: f32,
    /// tanh shaping scale override for the wave noise; 0 = built-in default.
    pub shaping_scale: f32,
    pub erosion_gain: f32,
    /// Low-octave high-pass knee (normalized |k|): smaller passes larger, rounder lobes.
    pub lobe_scale: f32,
    /// Vertical wavenumber multiplier of the low octaves: below 1 = taller lobes.
    pub lobe_aniso: f32,
}

impl Default for FlameNoise {
    fn default() -> Self {
        Self {
            amplitude: 1.5,
            contrast: 1.0,
            frequency: 6.0,
            scroll_speed: 1.0,
            aniso_y: 0.35,
            scale_mode: 0.0,
            shaping_scale: 0.0,
            erosion_gain: 1.0,
            lobe_scale: WAVE_LOBE_SCALE_DEFAULT,
            lobe_aniso: 1.0,
        }
    }
}

pub fn noise_lobe_shape(noise: &FlameNoise) -> WaveLobeShape {
    WaveLobeShape {
        scale: noise.lobe_scale,
        aniso_y: noise.lobe_aniso,
    }
}

/// Effective noise aniso y: `scale_mode` >= 0.5 compensates the world-to-local
/// scaling by height / radius.
pub fn effective_noise_aniso_y(noise: &FlameNoise, height: f32, radius: f32) -> f32 {
    if noise.scale_mode < 0.5 {
        noise.aniso_y
    } else {
        noise.aniso_y * (height / radius.max(1e-4))
    }
}

/// Noise Sharpness knob in [0, 1] to tanh shaping scale, log curve, crisper to
/// the right; `shaping_scale` stays the single source of truth.
pub fn noise_sharpness_to_shaping_scale(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    NOISE_SHARPNESS_SCALE_SOFT * (NOISE_SHARPNESS_SCALE_SHARP / NOISE_SHARPNESS_SCALE_SOFT).powf(v)
}

/// Inverse of [`noise_sharpness_to_shaping_scale`]. A non-positive scale means
/// "delegate to the built-in default" and derives the knob position from
/// [`crate::flame_wave::WAVE_TANH_SCALE`].
pub fn shaping_scale_to_noise_sharpness(scale: f32) -> f32 {
    let scale = if scale > 0.0 {
        scale
    } else {
        crate::flame_wave::WAVE_TANH_SCALE
    };
    let sharpness = (scale / NOISE_SHARPNESS_SCALE_SOFT).ln()
        / (NOISE_SHARPNESS_SCALE_SHARP / NOISE_SHARPNESS_SCALE_SOFT).ln();
    sharpness.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_noise_aniso_y_mode_zero() {
        let noise = FlameNoise {
            scale_mode: 0.0,
            aniso_y: 0.5,
            ..FlameNoise::default()
        };
        assert!((effective_noise_aniso_y(&noise, 1.6, 0.6) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_effective_noise_aniso_y_mode_one() {
        let noise = FlameNoise {
            scale_mode: 1.0,
            aniso_y: 0.5,
            ..FlameNoise::default()
        };
        assert!((effective_noise_aniso_y(&noise, 8.0, 1.0) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_noise_sharpness_endpoints_and_monotone() {
        assert!((noise_sharpness_to_shaping_scale(0.0) - NOISE_SHARPNESS_SCALE_SOFT).abs() < 1e-5);
        assert!((noise_sharpness_to_shaping_scale(1.0) - NOISE_SHARPNESS_SCALE_SHARP).abs() < 1e-5);
        assert_eq!(
            noise_sharpness_to_shaping_scale(2.0),
            noise_sharpness_to_shaping_scale(1.0)
        );
        assert_eq!(
            noise_sharpness_to_shaping_scale(-1.0),
            noise_sharpness_to_shaping_scale(0.0)
        );
        let mut previous = noise_sharpness_to_shaping_scale(0.0);
        for step in 1..=10 {
            let current = noise_sharpness_to_shaping_scale(step as f32 / 10.0);
            assert!(current < previous);
            previous = current;
        }
    }

    #[test]
    fn test_noise_sharpness_round_trip() {
        for scale in [0.1_f32, 0.25, 0.6, 1.0, 3.0, 6.0] {
            let recovered =
                noise_sharpness_to_shaping_scale(shaping_scale_to_noise_sharpness(scale));
            assert!(
                (recovered - scale).abs() / scale < 1e-3,
                "scale {scale} round-tripped to {recovered}"
            );
        }
    }

    #[test]
    fn test_noise_sharpness_delegation_matches_builtin_default() {
        assert_eq!(
            shaping_scale_to_noise_sharpness(0.0),
            shaping_scale_to_noise_sharpness(crate::flame_wave::WAVE_TANH_SCALE)
        );
    }
}
