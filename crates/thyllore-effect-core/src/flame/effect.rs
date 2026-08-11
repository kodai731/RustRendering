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
        };
        refresh_flame_coefficients(&mut effect, &FlameBaked::default());
        effect
    }
}

pub(crate) const MIN_FLAME_EXTENT: f32 = 1e-3;

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
