use super::*;
use cgmath::{Matrix4, Quaternion, Vector2, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub struct FlameEffect {
    pub position: Vector3<f32>,
    pub rotation: cgmath::Quaternion<f32>,
    pub height: f32,
    pub radius: f32,
    pub sigma_t: f32,
    pub intensity: f32,
    pub color_base: [f32; 3],
    pub color_tip: [f32; 3],
    pub temperature_base_k: f32,
    pub temperature_tip_k: f32,
    pub use_blackbody: bool,
    pub noise_amplitude: f32,
    /// Relative contrast of the noise carving: scales the edge smoothstep
    /// window half-width as hw0 / noise_contrast around a fixed center.
    /// 1.0 keeps the authored edge_low/edge_high window untouched.
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
    /// 0=world-Y axis (current) / 1=advect direction axis
    pub aniso_axis_advect: f32,
    /// mode0 の RTE 離散化帯数。1 以下 = legacy 線形放射経路、2 以上 = 帯ごと Beer-Lambert 合成
    pub rte_bands: f32,
    /// RTE 吸収の波長分散。0=グレー体 (旧一致)、1=Rayleigh 1/λ 比
    pub sigma_dispersion: f32,
    /// RTE 帯色を外縁 (r̂≈1) で tip 色へ寄せる薄い項。0=現行
    pub edge_temperature_blend: f32,
    /// 境界変位 amp。0 = off (変位前と bit 一致)
    pub boundary_amp: f32,
    pub boundary_freq: f32,
    pub boundary_speed: f32,
    pub boundary_radius_ratio: f32,
    pub near_fade_radius: f32,
    /// Residual (un-carved) medium fraction left where turbulence carves the
    /// emitting soot away — the floor that makes a fully carved ray span
    /// translucent instead of a hard hole. 0 restores the pre-floor field.
    pub carve_residual: f32,
    /// Tip-carve asymptotic depth kappa: relative carve deepens toward the
    /// flame's own luminous tip as 1 + kappa * exp(-mu/mu0). 0 = uniform depth.
    pub tip_carve_depth: f32,
    /// Tip-carve reach mu0 in remaining-luminous-fraction units (scale-free).
    pub tip_carve_reach: f32,
    /// Warp strain reach mu_w: how deep the tip-asymptotic warp strain
    /// penetrates, in remaining-luminous-fraction units (same scale as
    /// tip_carve_reach).
    pub warp_reach: f32,
    /// Medium swirl share: fraction weight of the strain budget spent on the
    /// azimuthal swirl-shear modes (differential rotation of the RTE medium
    /// density). 0 = off, bit-identical to the pre-swirl field; raising it
    /// thins the carve warp (total strain budget is fixed).
    pub swirl_gain: f32,
    /// Multiplier on the swirl phase-drift rate: how fast the shear layers
    /// counter-rotate relative to the rising material. Time-only, so it costs
    /// no strain budget and cannot fold the field — the lever for "how alive"
    /// the vortices look, independent of their strength.
    pub swirl_speed: f32,
    /// Medium spread (motion_design L3): age-coordinate radial opening of the
    /// RTE medium toward the luminous tip. The noise sampling contracts toward
    /// the axis as the material rises, so carved features enlarge, drift
    /// outward and dissolve instead of scrolling up unchanged. 0 = off,
    /// bit-identical to the unspread field. Reach shares tip_carve_reach.
    pub spread_gain: f32,
    /// Multiplier on the support radius (FLAME_SHELL_SUPPORT_HEADROOM = 1.5): how wide the
    /// flame's density support extends relative to the shell proxy. Scales the support radius
    /// across all three systems (density support, proxy shape, and analytic integration) so
    /// they stay matched. 1.0 = default (unchanged behavior).
    pub support_margin: f32,
    /// Animated meander amplitude: time-varying horizontal displacement of the centerline.
    /// 0 = off (bit-identical to pre-meander field).
    pub meander_amp: f32,
    pub edge_outer_sharpen: f32,
    pub noise_scale_mode: f32,
    pub erosion_noise_gain: f32,
    /// Medium twist gain (V design): amplitude (radians at the axis, tip) of the
    /// node-frozen azimuthal rotation of the noise coordinate. A rotation never
    /// folds, so unlike the shear warp modes this gain has no strain cap.
    /// 0 = off (bit-identical baseline).
    pub twist_gain: f32,
    /// Medium twist rate scale (V design, plan D): omega_j = twist_speed *
    /// (kappa_j / 2pi)^(2/3). 0 delegates to swirl_speed (the pre-plan-D
    /// behavior), > 0 gives the twist its own rate so amplitude (twist_gain)
    /// and rate stay independently fittable from reference footage (F9c / F9).
    pub twist_speed: f32,
    /// Burnout gain (D design): deepening of the deterministic erosion mean
    /// shrink toward the flame's own luminous top, mean_shrink'(h) =
    /// mean_shrink(h) * (1 + burnout_gain * exp(-mu(h) / mu0)) — the same
    /// remaining-luminous-fraction asymptote as the tip carve, sharing
    /// tip_carve_reach as mu0. The boost pushes the upper column toward the
    /// cut threshold so the stochastic noise troughs actually sever the
    /// support (base shedding, F7) and detached parcels rising into smaller
    /// mu burn away (dissipation). 0 = off (bit-identical baseline).
    pub burnout_gain: f32,
    /// tanh shaping scale override for the wave noise (0 = keep the built-in
    /// WAVE_TANH_SCALE / env default). The default 0.6 clips shaped noise at
    /// +-0.279 around the mean, crushing interior pattern contrast; raising
    /// the scale softens the clip while the variance-preserving amplitude
    /// normalization keeps the silhouette calibration. The sigmaEff chain
    /// reads the same shaping params, so the closed form stays in sync.
    pub noise_shaping_scale: f32,
}

impl Default for FlameEffect {
    fn default() -> Self {
        let mut effect = Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            height: 1.6,
            radius: 0.6,
            sigma_t: 1.0,
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
            boundary_amp: 0.2,
            boundary_freq: 1.6,
            boundary_speed: 0.8,
            boundary_radius_ratio: 0.2,
            near_fade_radius: 0.0,
            carve_residual: 0.12,
            tip_carve_depth: 1.0,
            tip_carve_reach: 0.2,
            warp_reach: crate::flame_wave::WARP_REACH_DEFAULT,
            swirl_gain: 0.0,
            swirl_speed: 1.0,
            spread_gain: 0.0,
            support_margin: 1.0,
            meander_amp: 0.0,
            edge_outer_sharpen: 0.0,
            noise_scale_mode: 0.0,
            erosion_noise_gain: 1.0,
            twist_gain: 0.0,
            twist_speed: 0.0,
            burnout_gain: 0.0,
            noise_shaping_scale: 0.0,
        };
        refresh_flame_coefficients(&mut effect, &FlameBaked::default());
        effect
    }
}

pub(crate) const MIN_FLAME_EXTENT: f32 = 1e-3;

/// Vortex macro (plan D): one UI knob v in [0, 1] mapped onto both twist
/// levers along a monotone "faster and deeper" curve. The macro is a
/// stateless write-through — the two levers stay the single source of truth
/// (a fit can still write them independently) and the knob position shown in
/// the UI is derived back from twist_gain alone. The curve constants are
/// provisional until the F9 gate calibration lands.
pub const VORTEX_MACRO_MAX_GAIN: f32 = 6.0;
pub const VORTEX_MACRO_MAX_SPEED: f32 = 2.0;

pub fn vortex_macro_levers(v: f32) -> (f32, f32) {
    let v = v.clamp(0.0, 1.0);
    (VORTEX_MACRO_MAX_GAIN * v, VORTEX_MACRO_MAX_SPEED * v)
}

/// Noise Sharpness macro: one UI knob v in [0, 1] mapped onto the tanh
/// shaping scale along a log curve, crisper to the right. Perceived crispness
/// comes from tanh saturation (small scale binarizes the pattern into hard
/// step edges), so the knob inverts and log-remaps the scale: the raw lever
/// packs its whole perceptual range into [~0.1, 1] of a 0-6 span and grows
/// softer as the value rises. Stateless write-through like the Vortex macro —
/// `noise_shaping_scale` stays the single source of truth (fit/serde keep
/// writing the raw scale) and the knob position is derived back from it.
pub const NOISE_SHARPNESS_SCALE_SOFT: f32 = 6.0;
pub const NOISE_SHARPNESS_SCALE_SHARP: f32 = 0.1;

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
        assert_eq!(vortex_macro_levers(0.0), (0.0, 0.0));
        assert_eq!(
            vortex_macro_levers(1.0),
            (VORTEX_MACRO_MAX_GAIN, VORTEX_MACRO_MAX_SPEED)
        );
        assert_eq!(vortex_macro_levers(2.0), vortex_macro_levers(1.0));
        assert_eq!(vortex_macro_levers(-1.0), vortex_macro_levers(0.0));
        let mut previous = vortex_macro_levers(0.0);
        for step in 1..=10 {
            let current = vortex_macro_levers(step as f32 / 10.0);
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
