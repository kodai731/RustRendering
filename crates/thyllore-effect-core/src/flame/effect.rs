use super::*;
use cgmath::{Matrix4, Quaternion, Vector2, Vector3};

/// Azimuthal swirl-shear of the RTE medium (differential rotation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameSwirl {
    /// Share of the fixed strain budget spent on the swirl modes; 0 = off.
    pub gain: f32,
    /// Phase-drift rate multiplier of the counter-rotating shear layers.
    pub speed: f32,
}

/// Node-frozen azimuthal rotation of the noise coordinate (V design).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameTwist {
    /// Rotation amplitude in radians at the axis tip; rotation never folds, so no strain cap.
    pub gain: f32,
    /// Own phase rate scale; 0 delegates the rate to swirl speed.
    pub speed: f32,
}

/// Discrete branch element layer: a deterministic spawner of vortex transport
/// elements that wrap the RTE medium around a rising core; period 0 or gain 0 = off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameBranch {
    /// Spawn period in seconds; 0 spawns nothing.
    pub period: f32,
    /// Element lifetime in seconds.
    pub life: f32,
    /// Peak rotation angle of the vortex core in radians at full envelope; 0 = off.
    pub gain: f32,
    /// Lamb-Oseen core radius as a ratio of the local trunk radius; near 1 the
    /// whole disc turns coherently (fat tongues), small values shear thin spirals.
    pub core_radius: f32,
    /// Lateral position of the vortex core at spawn as a ratio of the local trunk
    /// radius: 0 sits on the axis (the whole slab tilts), 1 sits on the shear
    /// layer so trunk material rolls outward as a billow.
    pub core_offset: f32,
    /// Compact reach of one element at the end of its life as a ratio of the local
    /// trunk radius; nothing beyond it moves, so it bounds the lateral extent.
    pub reach: f32,
    /// Scatter of azimuth, timing jitter and side alternation in [0, 1].
    pub spread: f32,
    /// Center of the spawn height band in local height units.
    pub spawn_height: f32,
    /// Full width of the spawn height band; 1 with center 0.5 covers the trunk.
    pub spawn_range: f32,
    pub seed: u32,
}

/// Sinusoidal displacement of the density boundary; amp 0 = off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameBoundary {
    pub amp: f32,
    pub freq: f32,
    pub speed: f32,
    pub radius_ratio: f32,
}

/// Carve deepening toward the flame's own luminous tip:
/// relative carve scales as 1 + depth * exp(-mu / reach).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameTipCarve {
    /// Asymptotic extra depth kappa; 0 = uniform depth.
    pub depth: f32,
    /// Reach mu0 in remaining-luminous-fraction units (scale-free).
    pub reach: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlameEffect {
    pub position: Vector3<f32>,
    pub rotation: cgmath::Quaternion<f32>,
    pub height: f32,
    pub radius: f32,
    pub sigma_t: f32,
    /// Line-of-sight optical thickness tau0 = sigma_t * radius; > 0 derives
    /// sigma_t as optical_depth / radius, 0 = use sigma_t directly.
    pub optical_depth: f32,
    pub intensity: f32,
    pub color_base: [f32; 3],
    pub color_tip: [f32; 3],
    pub temperature_base_k: f32,
    pub temperature_tip_k: f32,
    pub use_blackbody: bool,
    pub noise_amplitude: f32,
    /// Scales the edge smoothstep window half-width as hw0 / contrast; 1.0 keeps the authored window.
    pub noise_contrast: f32,
    pub noise_frequency: f32,
    pub noise_scroll_speed: f32,
    pub time: f32,
    pub time_scale: f32,
    pub time_offset: f32,
    pub coefficients: FlameCoefficients,
    pub light_position_world: Vector3<f32>,
    pub self_shadow_strength: f32,
    pub warp_amp: f32,
    pub warp_freq: f32,
    pub rise_speed: f32,
    pub taper_power: f32,
    pub radius_tip_ratio: f32,
    pub edge_low: f32,
    pub edge_high: f32,
    pub white_boost: f32,
    pub wind_direction: Vector2<f32>,
    pub bend_amount: f32,
    pub bend_power: f32,
    pub envelope_peak: f32,
    pub envelope_base: f32,
    pub envelope_tail: f32,
    pub radial_sharpness: f32,
    pub noise_aniso_y: f32,
    pub warp_y_scale: f32,
    pub emitter_kind: u32,
    pub ring_major_radius: f32,
    pub ring_angular_speed: f32,
    pub occlusion_lum_ref: f32,
    pub contour_wiggle_amp: f32,
    /// 0 = world-Y anisotropy axis, 1 = advect direction axis.
    pub aniso_axis_advect: f32,
    /// RTE band count in mode 0: <= 1 legacy linear path, >= 2 per-band Beer-Lambert.
    pub rte_bands: f32,
    /// RTE absorption wavelength dispersion: 0 = grey body, 1 = Rayleigh 1/lambda.
    pub sigma_dispersion: f32,
    /// Blend of the outer-rim RTE band color toward the tip color; 0 = off.
    pub edge_temperature_blend: f32,
    pub boundary: FlameBoundary,
    pub near_fade_radius: f32,
    /// Residual medium fraction left where turbulence carves the soot away; 0 = no floor.
    pub carve_residual: f32,
    pub tip_carve: FlameTipCarve,
    /// Warp strain reach: penetration depth of the tip-asymptotic warp strain,
    /// in the same remaining-luminous-fraction units as tip carve reach.
    pub warp_reach: f32,
    pub swirl: FlameSwirl,
    /// Age-coordinate radial opening of the medium toward the tip; 0 = off.
    pub spread_gain: f32,
    /// Multiplier on the shell support radius, kept matched across density
    /// support, proxy shape, and analytic integration; 1.0 = default.
    pub support_margin: f32,
    /// Animated horizontal displacement amplitude of the centerline; 0 = off.
    pub meander_amp: f32,
    pub edge_outer_sharpen: f32,
    pub noise_scale_mode: f32,
    pub erosion_noise_gain: f32,
    pub twist: FlameTwist,
    /// Deepening of the erosion mean shrink toward the luminous top, sharing
    /// the tip carve reach as mu0; 0 = off.
    pub burnout_gain: f32,
    /// tanh shaping scale override for the wave noise; 0 = built-in default.
    pub noise_shaping_scale: f32,
    pub branch: FlameBranch,
}

impl Default for FlameEffect {
    fn default() -> Self {
        let mut effect = Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            height: 1.6,
            radius: 0.6,
            sigma_t: 1.0,
            optical_depth: 0.0,
            intensity: 2.2,
            color_base: [1.0, 0.45, 0.1],
            color_tip: [1.0, 0.1, 0.02],
            temperature_base_k: 3200.0,
            temperature_tip_k: 1500.0,
            use_blackbody: true,
            noise_amplitude: 1.5,
            noise_contrast: 1.0,
            noise_frequency: 6.0,
            noise_scroll_speed: 1.0,
            time: 0.0,
            time_scale: 1.0,
            time_offset: 0.0,
            coefficients: FlameCoefficients::default(),
            light_position_world: Vector3::new(2.0, 3.0, 2.0),
            self_shadow_strength: 0.5,
            warp_amp: 1.4,
            warp_freq: 5.0,
            rise_speed: 1.5,
            taper_power: 1.4,
            radius_tip_ratio: 0.10,
            edge_low: 0.27,
            edge_high: 0.33,
            white_boost: 4.0,
            wind_direction: Vector2::new(0.0, 0.0),
            bend_amount: 0.0,
            bend_power: 1.7,
            envelope_peak: 0.25,
            envelope_base: 0.05,
            envelope_tail: 1.25,
            radial_sharpness: 4.0,
            noise_aniso_y: 0.35,
            warp_y_scale: 0.6,
            emitter_kind: 0,
            ring_major_radius: 1.0,
            ring_angular_speed: 0.6,
            occlusion_lum_ref: 1.0,
            contour_wiggle_amp: 0.3,
            aniso_axis_advect: 0.0,
            rte_bands: 4.0,
            sigma_dispersion: 1.0,
            edge_temperature_blend: 0.0,
            boundary: FlameBoundary {
                amp: 0.2,
                freq: 1.6,
                speed: 0.8,
                radius_ratio: 0.2,
            },
            near_fade_radius: 0.0,
            carve_residual: 0.12,
            tip_carve: FlameTipCarve {
                depth: 1.0,
                reach: 0.2,
            },
            warp_reach: crate::flame_wave::WARP_REACH_DEFAULT,
            swirl: FlameSwirl {
                gain: 0.0,
                speed: 1.0,
            },
            spread_gain: 0.0,
            support_margin: 1.0,
            meander_amp: 0.0,
            edge_outer_sharpen: 0.0,
            noise_scale_mode: 0.0,
            erosion_noise_gain: 1.0,
            twist: FlameTwist {
                gain: 0.0,
                speed: 0.0,
            },
            burnout_gain: 0.0,
            noise_shaping_scale: 0.0,
            branch: FlameBranch {
                period: 0.0,
                life: 2.5,
                gain: 0.0,
                core_radius: 0.35,
                core_offset: 0.0,
                reach: 1.5,
                spread: 0.3,
                spawn_height: 0.35,
                spawn_range: 0.4,
                seed: 0,
            },
        };
        refresh_flame_coefficients(&mut effect, &FlameBaked::default());
        effect
    }
}

/// Stateless write-through of the Vortex macro knob onto (twist gain, twist
/// speed); the two parameters stay the single source of truth.
pub fn vortex_macro_parameters(v: f32) -> (f32, f32) {
    let v = v.clamp(0.0, 1.0);
    (VORTEX_MACRO_MAX_GAIN * v, VORTEX_MACRO_MAX_SPEED * v)
}

/// Noise Sharpness knob in [0, 1] to tanh shaping scale, log curve, crisper to
/// the right; `noise_shaping_scale` stays the single source of truth.
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

pub fn advance_flame_time(effect: &mut FlameEffect, delta_time: f32) {
    effect.time += delta_time.max(0.0);
}

pub fn effective_sigma_t(effect: &FlameEffect) -> f32 {
    if effect.optical_depth > 0.0 {
        effect.optical_depth / effect.radius.max(MIN_FLAME_EXTENT)
    } else {
        effect.sigma_t
    }
}

pub fn flame_bounding_radius(effect: &FlameEffect) -> f32 {
    if effect.emitter_kind == 1 {
        effect.ring_major_radius + effect.radius
    } else {
        effect.radius
    }
}

pub fn build_flame_model_matrix(effect: &FlameEffect) -> Matrix4<f32> {
    let radius = flame_bounding_radius(effect).max(MIN_FLAME_EXTENT);
    let height = effect.height.max(MIN_FLAME_EXTENT);
    Matrix4::from_translation(effect.position)
        * Matrix4::from(effect.rotation)
        * Matrix4::from_nonuniform_scale(radius, height, radius)
}

pub fn build_flame_inverse_model_matrix(effect: &FlameEffect) -> Matrix4<f32> {
    let radius = flame_bounding_radius(effect).max(MIN_FLAME_EXTENT);
    let height = effect.height.max(MIN_FLAME_EXTENT);
    Matrix4::from_nonuniform_scale(1.0 / radius, 1.0 / height, 1.0 / radius)
        * Matrix4::from(effect.rotation.conjugate())
        * Matrix4::from_translation(-effect.position)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vortex_macro_is_monotone_and_off_at_zero() {
        assert_eq!(vortex_macro_parameters(0.0), (0.0, 0.0));
        assert_eq!(
            vortex_macro_parameters(1.0),
            (VORTEX_MACRO_MAX_GAIN, VORTEX_MACRO_MAX_SPEED)
        );
        assert_eq!(vortex_macro_parameters(2.0), vortex_macro_parameters(1.0));
        assert_eq!(vortex_macro_parameters(-1.0), vortex_macro_parameters(0.0));
        let mut previous = vortex_macro_parameters(0.0);
        for step in 1..=10 {
            let current = vortex_macro_parameters(step as f32 / 10.0);
            assert!(current.0 > previous.0 && current.1 > previous.1);
            previous = current;
        }
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
