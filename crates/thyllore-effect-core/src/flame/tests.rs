use super::*;
use crate::flame_trail::{FlameTrailSample, FlameTrailState};
use cgmath::{Deg, InnerSpace, Matrix3, Matrix4, Quaternion, Vector2, Vector3, Vector4};
use thyllore_color_core::blackbody_rgb;
use thyllore_math_core::{
    evaluate_chebyshev, fit_chebyshev, fit_erf_response, integrate_chebyshev,
    pack_coefficients_vec4, parametric_height_falloff, smooth_step,
};

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
    let ubo = build_flame_ubo(
        &effect,
        &FlameBaked::default(),
        &FlameTemporalAccum::default(),
    );
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
    assert_eq!(ubo.noise_amplitude, effect.noise.amplitude);
    assert_eq!(
        ubo.height_primitive_coefficients,
        effect.coefficients.height_primitive
    );
}

#[test]
fn test_effective_sigma_t_zero_optical_depth_uses_sigma_t() {
    let mut effect = FlameEffect::default();
    effect.sigma_t = 2.5;
    effect.radius = 1.7;
    assert_eq!(effective_sigma_t(&effect), 2.5);
    let ubo = build_flame_ubo(
        &effect,
        &FlameBaked::default(),
        &FlameTemporalAccum::default(),
    );
    assert_eq!(ubo.sigma_t, 2.5);
}

#[test]
fn test_effective_sigma_t_keeps_optical_depth_across_radius() {
    let mut effect = FlameEffect::default();
    effect.sigma_t = 4.0;
    effect.optical_depth = 4.0;
    for radius in [0.5, 1.0, 2.43] {
        effect.radius = radius;
        let tau = effective_sigma_t(&effect) * radius;
        assert!((tau - 4.0).abs() < 1e-5, "radius {radius}: tau {tau}");
    }
}

#[test]
fn test_flame_ubo_layout_is_std140_compatible() {
    // trail_coefficients is [[f32; 4]; 4] (64 bytes) instead of [f32; 4] (16 bytes)
    assert_eq!(
        std::mem::size_of::<FlameUBO>(),
        784 + 16 + 16 + 16 + 32 + 128 + 128 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 6848 + 1536 - 240 + 16 // rise_accel widened FlameWarpStyle to two vec4 rows
            + 48
            + 16
            + std::mem::size_of::<FlameMixParams>()
            + std::mem::size_of::<FlameSegmentParams>()
            + std::mem::size_of::<FlameThermalParams>()
            + std::mem::size_of::<FlameTwistField>()
            + std::mem::size_of::<[FlameMeanderMode; 2]>()
            + std::mem::size_of::<FlameBranchField>()
            + std::mem::size_of::<FlamePuffField>()
            + std::mem::size_of::<FlameFlowField>()
            + std::mem::size_of::<FlameEdgeStyle>()
            - 16
    );
    assert_eq!(std::mem::size_of::<FlameEdgeStyle>(), 32);
    assert_eq!(std::mem::size_of::<FlameSupportMotion>(), 16);
    assert_eq!(std::mem::size_of::<FlameMixParams>(), 32);
    assert_eq!(std::mem::size_of::<FlameSegmentParams>(), 16);
    assert_eq!(std::mem::size_of::<FlameThermalParams>(), 32);
    assert_eq!(std::mem::size_of::<FlameTwistField>(), 48);
    assert_eq!(std::mem::size_of::<[FlameMeanderMode; 2]>(), 64);
    assert_eq!(std::mem::size_of::<FlameBranchElement>(), 48);
    assert_eq!(std::mem::size_of::<FlameBranchAgeProfile>(), 32);
    assert_eq!(
        std::mem::size_of::<FlameBranchField>(),
        64 + 32 + 48 * BRANCH_MAX_ELEMENTS
    );
    assert_eq!(
        std::mem::size_of::<FlamePuffField>(),
        16 + 16 * PUFF_MAX_COUNT
    );
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
    let effect = FlameEffect::default();
    let temporal = FlameTemporalAccum {
        weight: 0.5,
        frame_index,
    };
    let ubo = build_flame_ubo(&effect, &FlameBaked::default(), &temporal);
    let expected_y = (frame_index % 16384) as f32;
    assert_eq!(
        ubo.temporal_data.frame_index, expected_y,
        "temporal_data.frame_index should be (frame_index %% 16384) as f32 to avoid precision loss"
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
    let tau_outside = evaluate_self_shadow_optical_depth(p_outside, l_up, &coefficients, sigma_t);
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

#[test]
fn test_effective_edge_window() {
    // noise_amplitude = 1.5 (REF): should match original values exactly
    let mut effect = FlameEffect::default();
    let (elo, ehi) = effective_edge_window(&effect.edge, &effect.noise);
    assert!((elo - 0.27).abs() < 1e-6, "elo={}", elo);
    assert!((ehi - 0.33).abs() < 1e-6, "ehi={}", ehi);

    // noise_amplitude = 3.0: half-width doubles (gamma=1.0)
    effect.noise.amplitude = 3.0;
    let (elo, ehi) = effective_edge_window(&effect.edge, &effect.noise);
    assert!((elo - 0.24).abs() < 1e-6, "elo={}", elo);
    assert!((ehi - 0.36).abs() < 1e-6, "ehi={}", ehi);

    // noise_amplitude = 0.0: lower bound clamp applies (hw clamped to 0.25*hw0)
    effect.noise.amplitude = 0.0;
    let (elo, ehi) = effective_edge_window(&effect.edge, &effect.noise);
    // hw0 = 0.03, clamped hw = 0.25 * 0.03 = 0.0075
    // c = 0.3, so elo = 0.3 - 0.0075 = 0.2925, ehi = 0.3 + 0.0075 = 0.3075
    assert!((elo - 0.2925).abs() < 1e-6, "elo={}", elo);
    assert!((ehi - 0.3075).abs() < 1e-6, "ehi={}", ehi);
}

#[test]
fn test_noise_contrast_scales_edge_window() {
    // contrast = 1.0: authored window returned bit-identically
    let mut effect = FlameEffect::default();
    let (lo, hi) = contrast_scaled_edges(&effect.edge, &effect.noise);
    assert_eq!(lo, effect.edge.low);
    assert_eq!(hi, effect.edge.high);

    // contrast = 2.0: half-width halves around fixed center 0.3
    effect.noise.contrast = 2.0;
    let (lo, hi) = contrast_scaled_edges(&effect.edge, &effect.noise);
    assert!((lo - 0.285).abs() < 1e-6, "lo={}", lo);
    assert!((hi - 0.315).abs() < 1e-6, "hi={}", hi);

    // contrast = 0.5: half-width doubles (softer)
    effect.noise.contrast = 0.5;
    let (lo, hi) = contrast_scaled_edges(&effect.edge, &effect.noise);
    assert!((lo - 0.24).abs() < 1e-6, "lo={}", lo);
    assert!((hi - 0.36).abs() < 1e-6, "hi={}", hi);

    // out-of-range contrast clamps to [0.25, 4.0]
    effect.noise.contrast = 100.0;
    let (lo, hi) = contrast_scaled_edges(&effect.edge, &effect.noise);
    assert!((lo - (0.3 - 0.03 / 4.0)).abs() < 1e-6, "lo={}", lo);
    assert!((hi - (0.3 + 0.03 / 4.0)).abs() < 1e-6, "hi={}", hi);

    // effective window rides on top: amp at REF keeps the scaled window
    effect.noise.contrast = 2.0;
    let (elo, ehi) = effective_edge_window(&effect.edge, &effect.noise);
    assert!((elo - 0.285).abs() < 1e-6, "elo={}", elo);
    assert!((ehi - 0.315).abs() < 1e-6, "ehi={}", ehi);
}
