use crate::flame_trail::{flame_trail_fade_weight, FlameTrailSample, FlameTrailState};
use cgmath::{Deg, InnerSpace, Matrix3, Matrix4, Quaternion, Vector2, Vector3, Vector4};
use thyllore_math_core::{
    evaluate_chebyshev, fit_chebyshev, fit_erf_response, integrate_chebyshev,
    pack_coefficients_vec4,
};

pub const HEIGHT_PRIMITIVE_COEFFICIENT_COUNT: usize = 12;
pub const HEIGHT_COEFFICIENT_COUNT: usize = 8;
pub const RADIAL_COEFFICIENT_COUNT: usize = 8;

pub struct FlameProfile {
    pub sigma_t: f32,
    pub height_falloff: Box<dyn Fn(f64) -> f64>,
    pub radial_falloff: Box<dyn Fn(f64) -> f64>,
}

impl Default for FlameProfile {
    fn default() -> Self {
        Self {
            sigma_t: 1.0,
            height_falloff: Box::new(default_height_falloff),
            radial_falloff: Box::new(default_radial_falloff),
        }
    }
}

pub fn default_height_falloff(height01: f64) -> f64 {
    parametric_height_falloff(height01, 0.25, 0.05, 1.25)
}

pub fn default_radial_falloff(radius01: f64) -> f64 {
    biweight_radial_falloff(radius01, 4.0)
}

/// Compact-support biweight radial falloff with the support radius derived from sharpness.
fn biweight_radial_falloff(radius01: f64, radial_sharpness: f32) -> f64 {
    let support = crate::flame_radial::flame_radial_support_radius(radial_sharpness) as f64;
    let inside = (1.0 - (radius01 / support) * (radius01 / support)).max(0.0);
    inside * inside
}

/// Smooth step function S(x) = x*x*(3-2x).
fn smooth_step(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Parametric height falloff using envelope parameters.
///
/// p = peak.clamp(0.05, 0.8), v0 = base.clamp(0.0, 0.95), q = tail.clamp(0.5, 4.0)
/// if h < p: v0 + (1.0 - v0) * S(h/p)
/// else: (1.0 - S((h-p)/(1.0-p))).powf(q)
/// Guard: when p >= 1-epsilon and h >= p, return 0 (denominator tiny).
pub fn parametric_height_falloff(h: f64, peak: f64, base: f64, tail: f64) -> f64 {
    let p = peak.clamp(0.05, 0.8);
    let v0 = base.clamp(0.0, 0.95);
    let q = tail.clamp(0.5, 4.0);

    let result = if h < p {
        v0 + (1.0 - v0) * smooth_step(h / p)
    } else {
        let denom = 1.0 - p;
        if denom < 1e-9 {
            0.0
        } else {
            (1.0 - smooth_step((h - p) / denom)).powf(q)
        }
    };

    result.clamp(0.0, 1.0)
}

/// Approximate blackbody (Planckian locus) color for a given temperature in Kelvin.
/// Uses a polynomial approximation valid for 800K-3000K, returning clamped linear RGB [0,1].
pub fn blackbody_rgb(kelvin: f32) -> [f32; 3] {
    let t = (kelvin - 800.0) / (3000.0 - 800.0); // normalize to [0, 1]
    let t2 = t * t;
    let t3 = t2 * t;

    // Polynomial approximation of Planckian locus for 800K-3000K
    // R: starts near 1.0 (hot), stays high
    // G: increases from ~0.1 to ~0.7
    // B: increases from ~0.0 to ~0.4
    let r = 1.0 - 0.3 * t + 0.2 * t2;
    let g = 0.1 + 0.6 * t - 0.15 * t2 + 0.1 * t3;
    let b = 0.0 + 0.4 * t - 0.2 * t2 + 0.15 * t3;

    [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameCoefficients {
    pub height_primitive: [[f32; 4]; 3],
    pub radial: [[f32; 4]; 2],
    pub height: [[f32; 4]; 2],
}

impl Default for FlameCoefficients {
    fn default() -> Self {
        fit_flame_coefficients(&FlameProfile::default())
    }
}

pub fn fit_flame_coefficients(profile: &FlameProfile) -> FlameCoefficients {
    let height_primitive_source = fit_chebyshev(
        &profile.height_falloff,
        (0.0, 1.0),
        HEIGHT_PRIMITIVE_COEFFICIENT_COUNT - 1,
    );
    let height_primitive = integrate_chebyshev(&height_primitive_source);
    let height_series = fit_chebyshev(
        &profile.height_falloff,
        (0.0, 1.0),
        HEIGHT_COEFFICIENT_COUNT,
    );
    let radial_series = fit_chebyshev(
        &profile.radial_falloff,
        (0.0, 1.0),
        RADIAL_COEFFICIENT_COUNT,
    );

    let height_primitive_slots = pack_coefficients_vec4(&height_primitive, 3);
    let height_slots = pack_coefficients_vec4(&height_series, 2);
    let radial_slots = pack_coefficients_vec4(&radial_series, 2);
    FlameCoefficients {
        height_primitive: [
            height_primitive_slots[0],
            height_primitive_slots[1],
            height_primitive_slots[2],
        ],
        radial: [radial_slots[0], radial_slots[1]],
        height: [height_slots[0], height_slots[1]],
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlameShadingMode {
    #[default]
    Analytic,
    ReferenceRaymarch,
    DebugThickness,
    NoiseRaymarch,
    DebugDepthClamp,
}

impl FlameShadingMode {
    pub const ALL: [FlameShadingMode; 5] = [
        FlameShadingMode::Analytic,
        FlameShadingMode::ReferenceRaymarch,
        FlameShadingMode::NoiseRaymarch,
        FlameShadingMode::DebugThickness,
        FlameShadingMode::DebugDepthClamp,
    ];

    pub fn label(self) -> &'static str {
        match self {
            FlameShadingMode::Analytic => "Analytic",
            FlameShadingMode::ReferenceRaymarch => "Reference Raymarch",
            FlameShadingMode::DebugThickness => "Debug Thickness",
            FlameShadingMode::NoiseRaymarch => "Noise Raymarch",
            FlameShadingMode::DebugDepthClamp => "Debug Depth Clamp",
        }
    }

    pub fn as_shader_value(self) -> i32 {
        match self {
            FlameShadingMode::Analytic => 0,
            FlameShadingMode::ReferenceRaymarch => 1,
            FlameShadingMode::DebugThickness => 2,
            FlameShadingMode::NoiseRaymarch => 3,
            FlameShadingMode::DebugDepthClamp => 4,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "analytic" => Some(FlameShadingMode::Analytic),
            "raymarch" => Some(FlameShadingMode::ReferenceRaymarch),
            "thickness" => Some(FlameShadingMode::DebugThickness),
            "noise" => Some(FlameShadingMode::NoiseRaymarch),
            "depthclamp" => Some(FlameShadingMode::DebugDepthClamp),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameRenderSettings {
    pub shading_mode: FlameShadingMode,
    pub reference_step_count: u32,
    pub noise_step_count: u32,
}

impl Default for FlameRenderSettings {
    fn default() -> Self {
        Self {
            shading_mode: FlameShadingMode::Analytic,
            reference_step_count: 128,
            noise_step_count: 8,
        }
    }
}

impl FlameRenderSettings {
    pub fn resolved_step_count(&self) -> u32 {
        match self.shading_mode {
            FlameShadingMode::Analytic | FlameShadingMode::DebugThickness => 1,
            FlameShadingMode::ReferenceRaymarch => self.reference_step_count.max(1),
            FlameShadingMode::NoiseRaymarch => self.noise_step_count.max(1),
            FlameShadingMode::DebugDepthClamp => 1,
        }
    }
}

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
    pub noise_frequency: f32,
    pub noise_scroll_speed: f32,
    pub time: f32,
    pub time_scale: f32,
    pub time_offset: f32,
    pub coefficients: FlameCoefficients,
    pub temporal_weight: f32,
    pub frame_index: u64,
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
    /// erosion のノイズ基底。0=fbm (既定)、1=kernel (blob 和)
    pub turbulence_model: f32,
    pub kernel_blob_size: f32,
    pub kernel_blob_amp: f32,
    /// 境界変位 amp。0 = off (変位前と bit 一致)
    pub boundary_amp: f32,
    pub boundary_freq: f32,
    pub boundary_speed: f32,
    pub boundary_radius_ratio: f32,
    pub near_fade_radius: f32,
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
            noise_frequency: 6.0,
            noise_scroll_speed: 1.0,
            time: 0.0,
            time_scale: 1.0,
            time_offset: 0.0,
            coefficients: FlameCoefficients::default(),
            temporal_weight: 0.0,
            frame_index: 0,
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
            turbulence_model: 0.0,
            kernel_blob_size: 0.15,
            kernel_blob_amp: 1.0,
            boundary_amp: 0.2,
            boundary_freq: 1.6,
            boundary_speed: 0.8,
            boundary_radius_ratio: 0.2,
            near_fade_radius: 0.0,
        };
        refresh_flame_coefficients(&mut effect);
        effect
    }
}

pub fn profile_from_effect(effect: &FlameEffect) -> FlameProfile {
    let peak = effect.envelope_peak as f64;
    let base = effect.envelope_base as f64;
    let tail = effect.envelope_tail as f64;
    let radial_sharpness = effect.radial_sharpness;
    FlameProfile {
        sigma_t: effect.sigma_t,
        height_falloff: Box::new(move |h: f64| parametric_height_falloff(h, peak, base, tail)),
        radial_falloff: Box::new(move |r: f64| biweight_radial_falloff(r, radial_sharpness)),
    }
}

pub fn refresh_flame_coefficients(effect: &mut FlameEffect) {
    effect.coefficients = fit_flame_coefficients(&profile_from_effect(effect));
}

const MIN_FLAME_EXTENT: f32 = 1e-3;

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

type KernelUboFields = (
    [f32; 4],
    [[f32; 4]; 2 * crate::flame_kernel::KERNEL_BLOB_COUNT],
);

fn build_kernel_ubo_fields(effect: &FlameEffect) -> KernelUboFields {
    use crate::flame_kernel::{generate_kernel_blobs, KernelBlobParams, KERNEL_BLOB_COUNT};

    let params = [effect.turbulence_model, 0.0, 0.0, 0.0];
    let mut packed = [[0.0f32; 4]; 2 * KERNEL_BLOB_COUNT];
    if effect.turbulence_model >= 0.5 {
        let ring_major_norm = if effect.emitter_kind == 1 {
            effect.ring_major_radius / flame_bounding_radius(effect)
        } else {
            0.0
        };
        let blobs = generate_kernel_blobs(&KernelBlobParams {
            emitter_kind: effect.emitter_kind,
            ring_major_norm,
            blob_size: effect.kernel_blob_size,
            blob_amp: effect.kernel_blob_amp,
            rise_speed: effect.rise_speed,
            time: effect.time,
        });
        for (i, blob) in blobs.iter().enumerate() {
            packed[2 * i] = [blob.center[0], blob.center[1], blob.center[2], blob.radius];
            packed[2 * i + 1] = [blob.amplitude, 0.0, 0.0, 0.0];
        }
    }
    (params, packed)
}

pub fn build_flame_ubo(effect: &FlameEffect) -> FlameUBO {
    let (color_base, color_mid, color_tip) = if effect.use_blackbody {
        let base = blackbody_rgb(effect.temperature_base_k);
        let tip = blackbody_rgb(effect.temperature_tip_k);
        let mid_temp = (effect.temperature_base_k + effect.temperature_tip_k) / 2.0;
        let mid = blackbody_rgb(mid_temp);
        (base, mid, tip)
    } else {
        let base = effect.color_base;
        let tip = effect.color_tip;
        let mid = [
            (base[0] + tip[0]) / 2.0,
            (base[1] + tip[1]) / 2.0,
            (base[2] + tip[2]) / 2.0,
        ];
        (base, mid, tip)
    };
    let radius = flame_bounding_radius(effect).max(MIN_FLAME_EXTENT);
    let height = effect.height.max(MIN_FLAME_EXTENT);
    let rel = effect.light_position_world - effect.position;
    let dir = Vector3::new(rel.x / radius, rel.y / height, rel.z / radius);
    let norm_dir = if dir.dot(dir) < 1e-6 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        dir.normalize()
    };
    let kernel_fields = build_kernel_ubo_fields(effect);
    FlameUBO {
        model: build_flame_model_matrix(effect),
        inverse_model: build_flame_inverse_model_matrix(effect),
        height_primitive_coefficients: effect.coefficients.height_primitive,
        radial_coefficients: effect.coefficients.radial,
        height_coefficients: effect.coefficients.height,
        time: effect.time,
        sigma_t: effect.sigma_t,
        intensity: effect.intensity,
        height_axis_scale: 1.0,
        noise_amplitude: effect.noise_amplitude,
        noise_frequency: effect.noise_frequency,
        noise_scroll_speed: effect.noise_scroll_speed,
        radialSharpness: effect.radial_sharpness,
        color_base: Vector4::new(
            color_base[0],
            color_base[1],
            color_base[2],
            effect.occlusion_lum_ref,
        ),
        color_mid: Vector4::new(color_mid[0], color_mid[1], color_mid[2], 1.0),
        color_tip: Vector4::new(
            color_tip[0],
            color_tip[1],
            color_tip[2],
            effect.edge_temperature_blend,
        ),
        temporal_data: Vector4::new(
            effect.temporal_weight,
            (effect.frame_index % 16384) as f32,
            effect.noise_aniso_y,
            effect.warp_y_scale,
        ),
        light_data: Vector4::new(
            norm_dir.x,
            norm_dir.y,
            norm_dir.z,
            effect.self_shadow_strength,
        ),
        style_params0: [
            effect.warp_amp,
            effect.warp_freq,
            effect.rise_speed,
            effect.taper_power,
        ],
        style_params1: [
            effect.radius_tip_ratio,
            effect.edge_low,
            effect.edge_high,
            effect.white_boost,
        ],
        style_params2: [
            effect.wind_direction.x,
            effect.wind_direction.y,
            effect.bend_amount,
            effect.bend_power,
        ],
        trail_unit_inverse: Matrix4::<f32>::from_scale(1.0),
        trail_meta: Vector4::new(0.0, 0.0, 0.0, 0.0),
        trail_samples: [[0.0; 4]; 16],
        emitter_params: Vector4::new(
            effect.emitter_kind as f32,
            if effect.emitter_kind == 1 {
                effect.ring_major_radius / flame_bounding_radius(effect)
            } else {
                0.0
            },
            effect.ring_angular_speed,
            if effect.emitter_kind == 2 { 0.15 } else { 0.0 },
        ),
        contour_params: [
            effect.contour_wiggle_amp,
            effect.aniso_axis_advect,
            effect.rte_bands,
            effect.sigma_dispersion,
        ],
        erosion_response: fit_erf_response(effect.edge_low, effect.edge_high).pack(),
        kernel_params: kernel_fields.0,
        kernel_blobs: kernel_fields.1,
        boundary_params: [
            effect.boundary_amp,
            effect.boundary_freq,
            effect.boundary_speed,
            effect.boundary_radius_ratio,
        ],
        near_fade_params: [effect.near_fade_radius, 0.0, 0.0, 0.0],
    }
}

/// Build the expanded model matrix for flame trail rendering.
/// Computes world AABB of all trail samples + effect position, then builds a rotation-free
/// expansion matrix using the same construction rules as build_flame_model_matrix.
pub fn build_flame_trail_expanded_matrix(
    effect: &FlameEffect,
    samples: &[FlameTrailSample],
) -> Matrix4<f32> {
    assert!(
        !samples.is_empty(),
        "build_flame_trail_expanded_matrix requires at least one sample"
    );

    // Compute world AABB of all samples + effect position
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut min_z = f32::MAX;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut max_z = f32::NEG_INFINITY;

    // Include effect position
    let ep = &effect.position;
    min_x = min_x.min(ep.x);
    max_x = max_x.max(ep.x);
    min_y = min_y.min(ep.y);
    max_y = max_y.max(ep.y);
    min_z = min_z.min(ep.z);
    max_z = max_z.max(ep.z);

    // Include all sample positions
    for s in samples {
        let p = &s.position;
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
        min_z = min_z.min(p[2]);
        max_z = max_z.max(p[2]);
    }
    // XZ center = AABB center
    let cx = (min_x + max_x) * 0.5;
    let cz = (min_z + max_z) * 0.5;

    // Extension radius = effect.radius + hypot(half_extent_x, half_extent_z)
    let half_extent_x = (max_x - min_x) * 0.5;
    let half_extent_z = (max_z - min_z) * 0.5;
    let extension_radius = flame_bounding_radius(effect)
        + (half_extent_x * half_extent_x + half_extent_z * half_extent_z).sqrt();

    // Extension height = effect.height + (max_y - min_y)
    let extension_height = effect.height + (max_y - min_y);

    // Base y = min_y
    let base_y = min_y;

    // Build rotation-free expansion matrix using same construction as build_flame_model_matrix
    Matrix4::from_translation(Vector3::new(cx, base_y, cz))
        * Matrix4::from_nonuniform_scale(extension_radius, extension_height, extension_radius)
}

/// Build trail UBO fields: (trailUnitInverse, trailMeta, trailSamples).
/// trailUnitInverse = inverse of the unit model matrix (without expansion).
/// For each sample i: localDelta_i = trailUnitInverse.linear_part * (sample.position - effect.position), w = fade weight.
/// meta = (count as f32, 0.0, 0.0, 0.0).
/// If count is 0, trailUnitInverse = identity matrix.
pub fn build_flame_trail_ubo_fields(
    effect: &FlameEffect,
    trail: &FlameTrailState,
) -> (Matrix4<f32>, Vector4<f32>, [[f32; 4]; 16]) {
    let count = trail.samples.len();

    if count == 0 {
        return (
            Matrix4::<f32>::from_scale(1.0),
            Vector4::new(0.0, 0.0, 0.0, 0.0),
            [[0.0; 4]; 16],
        );
    }

    // trailUnitInverse = inverse of unit model matrix (analytical: translation*scale -> inv_scale*inv_translation)
    let radius = flame_bounding_radius(effect);
    let trail_unit_inverse = Matrix4::from_translation(-effect.position)
        * Matrix4::from_nonuniform_scale(1.0 / radius, 1.0 / effect.height, 1.0 / radius);

    // Build trail samples array
    let mut trail_samples: [[f32; 4]; 16] = [[0.0; 4]; 16];
    for (i, sample) in trail.samples.iter().enumerate() {
        if i >= 16 {
            break;
        }
        // localDelta_i = trailUnitInverse.linear_part * (sample.position - effect.position)
        let diff = Vector3::new(
            sample.position[0] - effect.position.x,
            sample.position[1] - effect.position.y,
            sample.position[2] - effect.position.z,
        );
        let linear = Matrix3::<f32>::from_cols(
            Vector3::new(
                trail_unit_inverse[0][0],
                trail_unit_inverse[1][0],
                trail_unit_inverse[2][0],
            ),
            Vector3::new(
                trail_unit_inverse[0][1],
                trail_unit_inverse[1][1],
                trail_unit_inverse[2][1],
            ),
            Vector3::new(
                trail_unit_inverse[0][2],
                trail_unit_inverse[1][2],
                trail_unit_inverse[2][2],
            ),
        );
        let local_delta = linear * diff;

        // w = flame_trail_fade_weight(sample, trail.fade_seconds)
        let w = flame_trail_fade_weight(sample, trail.fade_seconds);

        trail_samples[i] = [local_delta.x, local_delta.y, local_delta.z, w];
    }

    let meta = Vector4::new(count as f32, 0.0, 0.0, 0.0);

    (trail_unit_inverse, meta, trail_samples)
}

/// Build FlameUBO with optional trail support.
/// If trail is Some AND trail_render_active AND trail.enabled AND samples not empty:
///   - Replace model/inverse_model with the expanded matrix and its inverse.
///   - Fill trail fields using build_flame_trail_ubo_fields.
/// Otherwise, return same as build_flame_ubo.
pub fn build_flame_ubo_with_trail(
    effect: &FlameEffect,
    trail: Option<&FlameTrailState>,
    trail_render_active: bool,
) -> FlameUBO {
    let trail = match trail {
        Some(t) if trail_render_active && t.enabled && !t.samples.is_empty() => t,
        _ => return build_flame_ubo(effect),
    };

    // Build expanded model matrix
    let model = build_flame_trail_expanded_matrix(effect, &trail.samples);
    // Expanded matrix is T*S (translation*scale), so inverse is S^-1 * T^-1
    let inverse_model =
        Matrix4::from_nonuniform_scale(1.0 / model[0][0], 1.0 / model[1][1], 1.0 / model[2][2])
            * Matrix4::from_translation(-Vector3::new(model[3][0], model[3][1], model[3][2]));

    // Build trail fields
    let (trail_unit_inverse, trail_meta, trail_samples) =
        build_flame_trail_ubo_fields(effect, trail);

    // Color computation same as build_flame_ubo
    let (color_base, color_mid, color_tip) = if effect.use_blackbody {
        let base = blackbody_rgb(effect.temperature_base_k);
        let tip = blackbody_rgb(effect.temperature_tip_k);
        let mid_temp = (effect.temperature_base_k + effect.temperature_tip_k) / 2.0;
        let mid = blackbody_rgb(mid_temp);
        (base, mid, tip)
    } else {
        let base = effect.color_base;
        let tip = effect.color_tip;
        let mid = [
            (base[0] + tip[0]) / 2.0,
            (base[1] + tip[1]) / 2.0,
            (base[2] + tip[2]) / 2.0,
        ];
        (base, mid, tip)
    };

    let radius = flame_bounding_radius(effect).max(MIN_FLAME_EXTENT);
    let height = effect.height.max(MIN_FLAME_EXTENT);
    let rel = effect.light_position_world - effect.position;
    let dir = Vector3::new(rel.x / radius, rel.y / height, rel.z / radius);
    let norm_dir = if dir.dot(dir) < 1e-6 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        dir.normalize()
    };

    let kernel_fields = build_kernel_ubo_fields(effect);
    FlameUBO {
        model,
        inverse_model,
        height_primitive_coefficients: effect.coefficients.height_primitive,
        radial_coefficients: effect.coefficients.radial,
        height_coefficients: effect.coefficients.height,
        time: effect.time,
        sigma_t: effect.sigma_t,
        intensity: effect.intensity,
        height_axis_scale: 1.0,
        noise_amplitude: effect.noise_amplitude,
        noise_frequency: effect.noise_frequency,
        noise_scroll_speed: effect.noise_scroll_speed,
        radialSharpness: effect.radial_sharpness,
        color_base: Vector4::new(
            color_base[0],
            color_base[1],
            color_base[2],
            effect.occlusion_lum_ref,
        ),
        color_mid: Vector4::new(color_mid[0], color_mid[1], color_mid[2], 1.0),
        color_tip: Vector4::new(
            color_tip[0],
            color_tip[1],
            color_tip[2],
            effect.edge_temperature_blend,
        ),
        temporal_data: Vector4::new(
            effect.temporal_weight,
            (effect.frame_index % 16384) as f32,
            effect.noise_aniso_y,
            effect.warp_y_scale,
        ),
        light_data: Vector4::new(
            norm_dir.x,
            norm_dir.y,
            norm_dir.z,
            effect.self_shadow_strength,
        ),
        style_params0: [
            effect.warp_amp,
            effect.warp_freq,
            effect.rise_speed,
            effect.taper_power,
        ],
        style_params1: [
            effect.radius_tip_ratio,
            effect.edge_low,
            effect.edge_high,
            effect.white_boost,
        ],
        style_params2: [
            effect.wind_direction.x,
            effect.wind_direction.y,
            effect.bend_amount,
            effect.bend_power,
        ],
        trail_unit_inverse,
        trail_meta,
        trail_samples,
        emitter_params: Vector4::new(
            effect.emitter_kind as f32,
            if effect.emitter_kind == 1 {
                effect.ring_major_radius / flame_bounding_radius(effect)
            } else {
                0.0
            },
            effect.ring_angular_speed,
            if effect.emitter_kind == 2 { 0.15 } else { 0.0 },
        ),
        contour_params: [
            effect.contour_wiggle_amp,
            effect.aniso_axis_advect,
            effect.rte_bands,
            effect.sigma_dispersion,
        ],
        erosion_response: fit_erf_response(effect.edge_low, effect.edge_high).pack(),
        kernel_params: kernel_fields.0,
        kernel_blobs: kernel_fields.1,
        boundary_params: [
            effect.boundary_amp,
            effect.boundary_freq,
            effect.boundary_speed,
            effect.boundary_radius_ratio,
        ],
        near_fade_params: [effect.near_fade_radius, 0.0, 0.0, 0.0],
    }
}

pub fn integrate_emission_segment(source: f32, sigma_t: f32, dt: f32) -> f32 {
    let x = sigma_t * dt;
    if x < 1e-3 {
        source * dt * (1.0 - 0.5 * x + x * x / 6.0)
    } else {
        source * (1.0 - (-x).exp()) / sigma_t
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameUBO {
    pub model: Matrix4<f32>,
    pub inverse_model: Matrix4<f32>,
    pub height_primitive_coefficients: [[f32; 4]; 3],
    pub radial_coefficients: [[f32; 4]; 2],
    pub height_coefficients: [[f32; 4]; 2],
    pub time: f32,
    pub sigma_t: f32,
    pub intensity: f32,
    pub height_axis_scale: f32,
    pub noise_amplitude: f32,
    pub noise_frequency: f32,
    pub noise_scroll_speed: f32,
    pub radialSharpness: f32,
    pub color_base: Vector4<f32>,
    pub color_mid: Vector4<f32>,
    pub color_tip: Vector4<f32>,
    pub temporal_data: Vector4<f32>,
    pub light_data: Vector4<f32>,
    pub style_params0: [f32; 4],
    pub style_params1: [f32; 4],
    pub style_params2: [f32; 4],
    pub trail_unit_inverse: Matrix4<f32>,
    pub trail_meta: Vector4<f32>,
    pub trail_samples: [[f32; 4]; 16],
    pub emitter_params: Vector4<f32>,
    pub contour_params: [f32; 4],
    pub erosion_response: [f32; 4],
    pub kernel_params: [f32; 4],
    pub kernel_blobs: [[f32; 4]; 2 * crate::flame_kernel::KERNEL_BLOB_COUNT],
    pub boundary_params: [f32; 4],
    pub near_fade_params: [f32; 4],
}

impl Default for FlameUBO {
    fn default() -> Self {
        build_flame_ubo(&FlameEffect::default())
    }
}

/// Evaluate self-shadow optical depth for a point in flame-local space.
/// Uses layered concentric cylinders (3 layers) with Chebyshev-evaluated density.
pub fn evaluate_self_shadow_optical_depth(
    p_local: [f32; 3],
    light_dir_local: [f32; 3],
    coefficients: &FlameCoefficients,
    sigma_t: f32,
) -> f32 {
    // Layer radii S = [1/3, 2/3, 1], midpoints m = [1/6, 0.5, 5/6]
    let s: [f32; 3] = [1.0 / 3.0, 2.0 / 3.0, 1.0];
    let m: [f32; 3] = [1.0 / 6.0, 0.5, 5.0 / 6.0];

    // Evaluate density at each layer midpoint using Chebyshev coefficients
    let radial_series = thyllore_math_core::ChebyshevSeries::new(
        coefficients.radial.iter().flatten().copied().collect(),
        (0.0, 1.0),
    );
    let mut dens = [0.0f32; 4];
    for k in 0..3 {
        dens[k] = evaluate_chebyshev(&radial_series, m[k]);
    }
    dens[3] = 0.0;

    // Compute weights w_k = dens_k - dens_{k+1}
    let w: [f32; 3] = [dens[0] - dens[1], dens[1] - dens[2], dens[2] - dens[3]];

    let px = p_local[0];
    let py = p_local[1];
    let pz = p_local[2];
    let lx = light_dir_local[0];
    let ly = light_dir_local[1];
    let lz = light_dir_local[2];

    // For each layer, compute the integral I_k
    let mut total: f32 = 0.0;

    for k in 0..3 {
        let sk = s[k];
        let a = lx * lx + lz * lz;

        // Find intersection of cylinder (x^2 + z^2 = S_k^2) and ray p + s*L
        let (s0, s1) = if a < 1e-6 {
            // Ray is parallel to cylinder axis
            if px * px + pz * pz <= sk * sk {
                (0.0, 1e4)
            } else {
                continue;
            }
        } else {
            // Solve quadratic: a*s^2 + 2*(px*lx + pz*lz)*s + (px^2 + pz^2 - sk^2) = 0
            let b = 2.0 * (px * lx + pz * lz);
            let c = px * px + pz * pz - sk * sk;
            let disc = b * b - 4.0 * a * c;

            if disc <= 0.0 {
                continue;
            }

            let sqrt_disc = disc.sqrt();
            let mut s_start = (-b - sqrt_disc) / (2.0 * a);
            let mut s_end = (-b + sqrt_disc) / (2.0 * a);

            // Clip to s >= 0
            if s_end < 0.0 {
                continue;
            }
            if s_start < 0.0 {
                s_start = 0.0;
            }

            (s_start, s_end)
        };

        // Clip interval by height h(s) = p.y + s*L.y in [0, 1]
        let mut lo = s0;
        let mut hi = s1;

        if ly.abs() < 1e-4 {
            // h is approximately constant
            if py < 0.0 || py > 1.0 {
                continue;
            }
            // F is coefficients.height evaluated at p.y
            let height_series = thyllore_math_core::ChebyshevSeries::new(
                coefficients.height.iter().flatten().copied().collect(),
                (0.0, 1.0),
            );
            let f_val = evaluate_chebyshev(&height_series, py);
            total += w[k] * f_val * (hi - lo);
        } else {
            // h(s) = py + s*ly, find where h in [0, 1]
            // s_lo = (0 - py) / ly, s_hi = (1 - py) / ly
            let mut s_lo = (0.0 - py) / ly;
            let mut s_hi = (1.0 - py) / ly;

            if s_lo > s_hi {
                std::mem::swap(&mut s_lo, &mut s_hi);
            }

            // Clip [lo, hi] by [s_lo, s_hi]
            lo = lo.max(s_lo);
            hi = hi.min(s_hi);

            if lo >= hi {
                continue;
            }

            // I_k = (H1(h(s1)) - H1(h(s0))) / L.y
            let h_s0 = py + lo * ly;
            let h_s1 = py + hi * ly;

            let height_prim_series = thyllore_math_core::ChebyshevSeries::new(
                coefficients
                    .height_primitive
                    .iter()
                    .flatten()
                    .copied()
                    .collect(),
                (0.0, 1.0),
            );
            let h1_s0 = evaluate_chebyshev(&height_prim_series, h_s0);
            let h1_s1 = evaluate_chebyshev(&height_prim_series, h_s1);

            total += w[k] * (h1_s1 - h1_s0) / ly;
        }
    }

    (sigma_t * total).max(0.0)
}
#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_math_core::evaluate_chebyshev;

    fn evaluate_chebyshev12_unrolled(slots: &[[f32; 4]; 3], x01: f32) -> f32 {
        let c: Vec<f32> = slots.iter().flatten().copied().collect();
        let u = 2.0 * x01 - 1.0;
        let t = 2.0 * u;
        let b11 = c[11];
        let b10 = t * b11 + c[10];
        let b9 = t * b10 - b11 + c[9];
        let b8 = t * b9 - b10 + c[8];
        let b7 = t * b8 - b9 + c[7];
        let b6 = t * b7 - b8 + c[6];
        let b5 = t * b6 - b7 + c[5];
        let b4 = t * b5 - b6 + c[4];
        let b3 = t * b4 - b5 + c[3];
        let b2 = t * b3 - b4 + c[2];
        let b1 = t * b2 - b3 + c[1];
        u * b1 - b2 + c[0]
    }

    #[test]
    fn test_fit_flame_coefficients_height_primitive_matches_series() {
        let coefficients = fit_flame_coefficients(&FlameProfile::default());
        let series = fit_chebyshev(
            default_height_falloff,
            (0.0, 1.0),
            HEIGHT_PRIMITIVE_COEFFICIENT_COUNT - 1,
        );
        let primitive = integrate_chebyshev(&series);

        for i in 0..=32 {
            let x01 = i as f32 / 32.0;
            let unrolled = evaluate_chebyshev12_unrolled(&coefficients.height_primitive, x01);
            let reference = evaluate_chebyshev(&primitive, x01);
            assert!(
                (unrolled - reference).abs() < 1e-5,
                "x01 = {x01}: unrolled = {unrolled}, reference = {reference}"
            );
        }
    }

    #[test]
    fn test_fit_flame_coefficients_height_primitive_is_zero_at_base() {
        let coefficients = fit_flame_coefficients(&FlameProfile::default());
        let at_base = evaluate_chebyshev12_unrolled(&coefficients.height_primitive, 0.0);
        assert!(at_base.abs() < 1e-5);
    }

    #[test]
    fn test_fit_flame_coefficients_is_deterministic() {
        let profile = FlameProfile::default();
        let first = fit_flame_coefficients(&profile);
        let second = fit_flame_coefficients(&profile);
        assert_eq!(first, second);
    }

    #[test]
    fn test_integrate_emission_segment_continuous_at_taylor_switch() {
        let sigma_t = 1.0;
        let below = integrate_emission_segment(1.0, sigma_t, 1e-3 - 1e-7);
        let above = integrate_emission_segment(1.0, sigma_t, 1e-3 + 1e-7);
        assert!((below - above).abs() < 1e-6);
    }

    #[test]
    fn test_integrate_emission_segment_matches_exact_form() {
        for &(sigma_t, dt) in &[(0.5f32, 2.0f32), (2.0, 0.1), (4.0, 1.5)] {
            let exact = (1.0 - (-(sigma_t as f64) * dt as f64).exp()) / sigma_t as f64;
            let actual = integrate_emission_segment(1.0, sigma_t, dt) as f64;
            assert!((actual - exact).abs() < 1e-6);
        }
    }

    #[test]
    fn test_flame_shading_mode_parse_matches_shader_values() {
        let cases = [
            ("analytic", FlameShadingMode::Analytic, 0),
            ("raymarch", FlameShadingMode::ReferenceRaymarch, 1),
            ("thickness", FlameShadingMode::DebugThickness, 2),
            ("noise", FlameShadingMode::NoiseRaymarch, 3),
            ("depthclamp", FlameShadingMode::DebugDepthClamp, 4),
        ];
        for (name, mode, shader_value) in cases {
            assert_eq!(FlameShadingMode::parse(name), Some(mode));
            assert_eq!(mode.as_shader_value(), shader_value);
        }
        assert_eq!(FlameShadingMode::parse("unknown"), None);
    }

    #[test]
    fn test_resolved_step_count_selects_per_mode_count() {
        let mut settings = FlameRenderSettings::default();
        assert_eq!(settings.resolved_step_count(), 1);

        settings.shading_mode = FlameShadingMode::ReferenceRaymarch;
        assert_eq!(settings.resolved_step_count(), 128);

        settings.shading_mode = FlameShadingMode::NoiseRaymarch;
        assert_eq!(settings.resolved_step_count(), 8);

        settings.noise_step_count = 0;
        assert_eq!(settings.resolved_step_count(), 1);
    }

    #[test]
    fn test_build_flame_model_and_inverse_are_consistent() {
        let effect = FlameEffect {
            position: Vector3::new(1.5, -0.25, 3.0),
            height: 2.0,
            radius: 0.5,
            ..FlameEffect::default()
        };
        let product = build_flame_model_matrix(&effect) * build_flame_inverse_model_matrix(&effect);
        let identity = Matrix4::<f32>::from_scale(1.0);
        for column in 0..4 {
            for row in 0..4 {
                assert!(
                    (product[column][row] - identity[column][row]).abs() < 1e-5,
                    "model * inverse_model differs from identity at [{column}][{row}]"
                );
            }
        }
    }

    #[test]
    fn test_build_flame_ubo_clamps_degenerate_extent() {
        let effect = FlameEffect {
            height: 0.0,
            radius: -1.0,
            ..FlameEffect::default()
        };
        let ubo = build_flame_ubo(&effect);
        assert!(ubo.model[0][0] > 0.0);
        assert!(ubo.inverse_model[1][1].is_finite());
    }

    #[test]
    fn test_advance_flame_time_accumulates_and_ignores_negative() {
        let mut effect = FlameEffect::default();
        advance_flame_time(&mut effect, 0.5);
        advance_flame_time(&mut effect, 0.25);
        advance_flame_time(&mut effect, -1.0);
        assert!((effect.time - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_flame_ubo_default_matches_effect_default() {
        let ubo = FlameUBO::default();
        let effect = FlameEffect::default();
        assert_eq!(ubo.sigma_t, effect.sigma_t);
        assert_eq!(ubo.noise_amplitude, effect.noise_amplitude);
        assert_eq!(
            ubo.height_primitive_coefficients,
            effect.coefficients.height_primitive
        );
    }

    #[test]
    fn test_flame_ubo_layout_is_std140_compatible() {
        assert_eq!(std::mem::size_of::<FlameUBO>(), 784 + 16 + 3072 + 16 + 16);
        assert_eq!(std::mem::align_of::<FlameUBO>() % 4, 0);
    }

    #[test]
    fn test_blackbody_rgb_clamped_to_unit() {
        for kelvin in [800.0, 1100.0, 1500.0, 2000.0, 2500.0, 3000.0] {
            let rgb = blackbody_rgb(kelvin);
            for &c in &rgb {
                assert!(c >= 0.0 && c <= 1.0, "kelvin={}, channel={}", kelvin, c);
            }
        }
    }

    #[test]
    fn test_blackbody_rgb_1100k_is_red_dominant() {
        let rgb = blackbody_rgb(1100.0);
        assert!(rgb[0] > rgb[1], "R > G at 1100K: {} > {}", rgb[0], rgb[1]);
        assert!(rgb[1] > rgb[2], "G > B at 1100K: {} > {}", rgb[1], rgb[2]);
    }

    #[test]
    fn test_blackbody_rgb_2500k_is_whiter_than_1100k() {
        let cold = blackbody_rgb(1100.0);
        let hot = blackbody_rgb(2500.0);
        assert!(
            hot[1] > cold[1],
            "G at 2500K > G at 1100K: {} > {}",
            hot[1],
            cold[1]
        );
        assert!(
            hot[2] > cold[2],
            "B at 2500K > B at 1100K: {} > {}",
            hot[2],
            cold[2]
        );
    }

    #[test]
    fn test_build_flame_ubo_large_frame_index_precision() {
        // Use a frame_index larger than 2^24 to verify that modular arithmetic
        // prevents f32 precision loss (f32 can only represent integers exactly up to 2^24).
        let frame_index: u64 = (1u64 << 24) + 16385;
        let effect = FlameEffect {
            temporal_weight: 0.5,
            frame_index,
            ..Default::default()
        };
        let ubo = build_flame_ubo(&effect);
        let expected_y = (frame_index % 16384) as f32;
        assert_eq!(
            ubo.temporal_data.y, expected_y,
            "temporal_data.y should be (frame_index %% 16384) as f32 to avoid precision loss"
        );
    }

    #[test]
    fn test_evaluate_self_shadow_optical_depth_layered_density() {
        // Test numerical integration of layered constant density vs evaluate_self_shadow_optical_depth
        let coefficients = fit_flame_coefficients(&FlameProfile::default());
        let sigma_t = 1.0;

        // Build the same density model as evaluate_self_shadow_optical_depth:
        // piecewise-constant in radius (3 layers), Chebyshev height profile
        let radial_series = thyllore_math_core::ChebyshevSeries::new(
            coefficients.radial.iter().flatten().copied().collect(),
            (0.0, 1.0),
        );
        let height_series = thyllore_math_core::ChebyshevSeries::new(
            coefficients.height.iter().flatten().copied().collect(),
            (0.0, 1.0),
        );

        // Layer radii and midpoints
        let s: [f32; 3] = [1.0 / 3.0, 2.0 / 3.0, 1.0];
        let m: [f32; 3] = [1.0 / 6.0, 0.5, 5.0 / 6.0];

        // Evaluate density at each layer midpoint
        let mut dens = [0.0f32; 4];
        for k in 0..3 {
            dens[k] = evaluate_chebyshev(&radial_series, m[k]);
        }
        dens[3] = 0.0;

        // Compute weights w_k = dens_k - dens_{k+1}
        let w: [f32; 3] = [dens[0] - dens[1], dens[1] - dens[2], dens[2] - dens[3]];

        // Define density function matching the layered model
        fn layered_density(
            r: f32,
            h: f32,
            w: &[f32; 3],
            s: &[f32; 3],
            height_series: &thyllore_math_core::ChebyshevSeries,
        ) -> f32 {
            if r >= s[2] {
                return 0.0;
            }
            // Find which layer r falls in
            for k in 0..3 {
                if r < s[k] {
                    // Density contribution from this and outer layers
                    let mut total = 0.0;
                    for j in k..3 {
                        total += w[j];
                    }
                    return total * evaluate_chebyshev(height_series, h);
                }
            }
            0.0
        }

        // Numerical integration along a ray
        fn numerical_tau(
            p: [f32; 3],
            l: [f32; 3],
            sigma_t: f32,
            w: &[f32; 3],
            s: &[f32; 3],
            height_series: &thyllore_math_core::ChebyshevSeries,
        ) -> f32 {
            let mut tau = 0.0;
            let steps = 1000;
            for i in 0..steps {
                let t = (i as f32 + 0.5) / steps as f32;
                let x = p[0] + t * l[0];
                let y = p[1] + t * l[1];
                let z = p[2] + t * l[2];
                let r = (x * x + z * z).sqrt();
                let dens = layered_density(r, y, w, s, height_series);
                tau += sigma_t * dens / steps as f32;
            }
            tau
        }

        // Test with a ray through the center
        let p = [0.0, 0.5, 0.0];
        let l = [1.0, 0.0, 0.0];
        let analytical = evaluate_self_shadow_optical_depth(p, l, &coefficients, sigma_t);
        let numerical = numerical_tau(p, l, sigma_t, &w, &s, &height_series);

        // Relative error should be < 1e-2
        let rel_error = (analytical - numerical).abs() / numerical.max(1e-6);
        assert!(rel_error < 1e-2, "relative error {} >= 1e-2", rel_error);
    }

    #[test]
    fn test_evaluate_self_shadow_optical_depth_basic_properties() {
        let coefficients = fit_flame_coefficients(&FlameProfile::default());
        let sigma_t = 1.0;

        // Test p=[0, 0.1, 0] with light direction (0,1,0) - should have tau > 0
        let p = [0.0, 0.1, 0.0];
        let l_up = [0.0, 1.0, 0.0];
        let tau_up = evaluate_self_shadow_optical_depth(p, l_up, &coefficients, sigma_t);
        assert!(tau_up > 0.0, "tau should be > 0 for upward light");
        assert!(tau_up.is_finite(), "tau should be finite");

        // Test p=[0, 0.1, 0] with light direction (1,0,0) - should have tau > 0
        let l_side = [1.0, 0.0, 0.0];
        let tau_side = evaluate_self_shadow_optical_depth(p, l_side, &coefficients, sigma_t);
        assert!(tau_side > 0.0, "tau should be > 0 for side light");
        assert!(tau_side.is_finite(), "tau should be finite");

        // Test p=[5, 0.5, 0] - outside the flame, tau should be ~0
        let p_outside = [5.0, 0.5, 0.0];
        let tau_outside =
            evaluate_self_shadow_optical_depth(p_outside, l_up, &coefficients, sigma_t);
        assert!(
            tau_outside < 1e-3,
            "tau should be ~0 for point outside flame"
        );
    }

    #[test]
    fn test_evaluate_self_shadow_optical_depth_smooth_density() {
        // Test relative error < 0.5 compared to numerical integration of exp(-4r^2)*F(h)
        let coefficients = fit_flame_coefficients(&FlameProfile::default());
        let sigma_t = 1.0;

        // Numerical integration of exp(-4r^2)*F(h) along a ray
        fn numerical_smooth_tau(p: [f32; 3], l: [f32; 3], sigma_t: f32) -> f32 {
            let mut tau = 0.0;
            let steps = 1000;
            for i in 0..steps {
                let t = (i as f32 + 0.5) / steps as f32;
                let x = p[0] + t * l[0];
                let y = p[1] + t * l[1];
                let z = p[2] + t * l[2];
                let r = (x * x + z * z).sqrt();
                let dens = (-4.0 * r * r).exp() * (1.0 - y * y); // F(h) approximation
                tau += sigma_t * dens / steps as f32;
            }
            tau
        }

        // Test with a ray through the center
        let p = [0.0, 0.5, 0.0];
        let l = [1.0, 0.0, 0.0];
        let analytical = evaluate_self_shadow_optical_depth(p, l, &coefficients, sigma_t);
        let numerical = numerical_smooth_tau(p, l, sigma_t);

        // Relative error should be < 0.5 (layer approximation is coarse)
        let rel_error = (analytical - numerical).abs() / numerical.max(1e-6);
        assert!(rel_error < 0.5, "relative error {} >= 0.5", rel_error);
    }

    #[test]
    fn test_flame_model_matrix_inverse_parity() {
        let mut effect = FlameEffect::default();
        effect.position = Vector3::new(1.0, 2.0, 3.0);
        effect.rotation = Quaternion::from(cgmath::Euler::new(Deg(0.0), Deg(0.0), Deg(30.0)));
        let model = build_flame_model_matrix(&effect);
        let inverse = build_flame_inverse_model_matrix(&effect);
        let identity = model * inverse;
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (identity[i][j] - (if i == j { 1.0 } else { 0.0 })).abs() < 1e-4,
                    "identity[{}][{}] = {}",
                    i,
                    j,
                    identity[i][j]
                );
            }
        }
    }
}
