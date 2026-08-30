use crate::flame::*;
use cgmath::{Matrix4, Quaternion, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub struct FlameEffect {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub height: f32,
    pub radius: f32,
    pub sigma_t: f32,
    /// Line-of-sight optical thickness tau0 = sigma_t * radius; > 0 derives
    /// sigma_t as optical_depth / radius, 0 = use sigma_t directly.
    pub optical_depth: f32,
    pub intensity: f32,
    pub time: f32,
    pub time_scale: f32,
    pub time_offset: f32,
    pub coefficients: FlameCoefficients,
    pub light_position_world: Vector3<f32>,
    pub self_shadow_strength: f32,
    pub radial_sharpness: f32,
    /// Age-coordinate radial opening of the medium toward the tip; 0 = off.
    pub spread_gain: f32,
    /// Multiplier on the shell support radius, kept matched across density
    /// support, proxy shape, and analytic integration; 1.0 = default.
    pub support_margin: f32,
    /// Closed-form segments per ray of the wave walk: finer noise needs more
    /// (the segment grid aliases at noise frequency > ~2 with 64); 64 = default.
    pub wave_segments: u32,
    pub color: FlameColor,
    pub noise: FlameNoise,
    pub warp: FlameWarp,
    pub wind: FlameWind,
    pub edge: FlameEdge,
    pub envelope: FlameEnvelope,
    pub emitter: FlameEmitter,
    pub contour: FlameContour,
    pub boundary: FlameBoundary,
    pub carve: FlameCarve,
    pub mix: FlameMix,
    pub thermal: FlameThermal,
    pub swirl: FlameSwirl,
    pub twist: FlameTwist,
    pub meander: FlameMeander,
    pub branch: FlameBranch,
    pub puff: FlamePuff,
    pub flow: FlameFlow,
    pub lobe: FlameLobe,
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
            time: 0.0,
            time_scale: 1.0,
            time_offset: 0.0,
            coefficients: FlameCoefficients::default(),
            light_position_world: Vector3::new(2.0, 3.0, 2.0),
            self_shadow_strength: 0.5,
            radial_sharpness: 4.0,
            spread_gain: 0.0,
            support_margin: 1.0,
            wave_segments: crate::flame_wave::FLAME_WAVE_SEGMENTS as u32,
            color: FlameColor::default(),
            noise: FlameNoise::default(),
            warp: FlameWarp::default(),
            wind: FlameWind::default(),
            edge: FlameEdge::default(),
            envelope: FlameEnvelope::default(),
            emitter: FlameEmitter::default(),
            contour: FlameContour::default(),
            boundary: FlameBoundary::default(),
            carve: FlameCarve::default(),
            mix: FlameMix::default(),
            thermal: FlameThermal::default(),
            swirl: FlameSwirl::default(),
            twist: FlameTwist::default(),
            meander: FlameMeander::default(),
            branch: FlameBranch::default(),
            puff: FlamePuff::default(),
            flow: FlameFlow::default(),
            lobe: FlameLobe::default(),
        };
        refresh_flame_coefficients(&mut effect, &FlameBaked::default());
        effect
    }
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
    emitter_bounding_radius(&effect.emitter, effect.radius)
}

pub fn wave_segment_count(effect: &FlameEffect) -> u32 {
    effect
        .wave_segments
        .clamp(WAVE_SEGMENTS_MIN, WAVE_SEGMENTS_MAX)
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
