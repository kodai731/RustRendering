use cgmath::{Matrix4, SquareMatrix, Vector4};
use thyllore_math_core::{fit_chebyshev, integrate_chebyshev, pack_coefficients_vec4};

pub const HEIGHT_PRIMITIVE_COEFFICIENT_COUNT: usize = 12;
pub const HEIGHT_COEFFICIENT_COUNT: usize = 8;
pub const RADIAL_COEFFICIENT_COUNT: usize = 8;

pub struct FlameProfile {
    pub sigma_t: f32,
    pub height_falloff: fn(f64) -> f64,
    pub radial_falloff: fn(f64) -> f64,
}

impl Default for FlameProfile {
    fn default() -> Self {
        Self {
            sigma_t: 1.0,
            height_falloff: default_height_falloff,
            radial_falloff: default_radial_falloff,
        }
    }
}

pub fn default_height_falloff(height01: f64) -> f64 {
    1.0 - height01 * height01 * (3.0 - 2.0 * height01)
}

pub fn default_radial_falloff(radius01: f64) -> f64 {
    (-4.0 * radius01 * radius01).exp()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameCoefficients {
    pub height_primitive: [[f32; 4]; 3],
    pub radial: [[f32; 4]; 2],
    pub height: [[f32; 4]; 2],
}

pub fn fit_flame_coefficients(profile: &FlameProfile) -> FlameCoefficients {
    let height_primitive_source = fit_chebyshev(
        profile.height_falloff,
        (0.0, 1.0),
        HEIGHT_PRIMITIVE_COEFFICIENT_COUNT - 1,
    );
    let height_primitive = integrate_chebyshev(&height_primitive_source);
    let height_series = fit_chebyshev(profile.height_falloff, (0.0, 1.0), HEIGHT_COEFFICIENT_COUNT);
    let radial_series = fit_chebyshev(profile.radial_falloff, (0.0, 1.0), RADIAL_COEFFICIENT_COUNT);

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
}

impl FlameShadingMode {
    pub fn as_shader_value(self) -> i32 {
        match self {
            FlameShadingMode::Analytic => 0,
            FlameShadingMode::ReferenceRaymarch => 1,
            FlameShadingMode::DebugThickness => 2,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "analytic" => Some(FlameShadingMode::Analytic),
            "raymarch" => Some(FlameShadingMode::ReferenceRaymarch),
            "thickness" => Some(FlameShadingMode::DebugThickness),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameRenderSettings {
    pub shading_mode: FlameShadingMode,
    pub reference_step_count: u32,
}

impl Default for FlameRenderSettings {
    fn default() -> Self {
        Self {
            shading_mode: FlameShadingMode::Analytic,
            reference_step_count: 128,
        }
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
    pub _padding: f32,
    pub color_base: Vector4<f32>,
    pub color_tip: Vector4<f32>,
}

impl Default for FlameUBO {
    fn default() -> Self {
        Self {
            model: Matrix4::identity(),
            inverse_model: Matrix4::identity(),
            height_primitive_coefficients: [[0.0; 4]; 3],
            radial_coefficients: [[0.0; 4]; 2],
            height_coefficients: [[0.0; 4]; 2],
            time: 0.0,
            sigma_t: 1.0,
            intensity: 1.0,
            height_axis_scale: 1.0,
            noise_amplitude: 0.5,
            noise_frequency: 4.0,
            noise_scroll_speed: 1.0,
            _padding: 0.0,
            color_base: Vector4::new(1.0, 0.45, 0.1, 1.0),
            color_tip: Vector4::new(1.0, 0.1, 0.02, 1.0),
        }
    }
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
    fn test_flame_ubo_layout_is_std140_compatible() {
        assert_eq!(std::mem::size_of::<FlameUBO>(), 304);
        assert_eq!(std::mem::align_of::<FlameUBO>() % 4, 0);
    }
}
