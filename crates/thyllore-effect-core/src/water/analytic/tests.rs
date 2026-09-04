use cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector3};

use super::laplace_beltrami_basis::{
    compute_laplace_beltrami_modes_cached, LAPLACE_BELTRAMI_MODE_COUNT,
    LAPLACE_BELTRAMI_SLOTS_PER_MODE,
};
use super::pick::{pick_torus, water_total_height_and_gradient};
use super::wave::{generate_water_wave_modes, water_height_and_gradient, water_perturbed_normal};
use crate::water::effect::WaterTorusEffect;
use crate::water::gpu::systems::build_laplace_beltrami_modes;
use thyllore_math_core::torus_surface_normal;

#[test]
fn test_laplace_beltrami_blend_zero_matches_flat_wave_only() {
    let effect = WaterTorusEffect::default();
    assert_eq!(effect.wave_lb_blend, 0.0);

    let (u, v) = (1.2f32, 0.8f32);
    let (total_h, total_h_u, total_h_v) = water_total_height_and_gradient(&effect, u, v, 0);

    let modes = generate_water_wave_modes(
        effect.wave_amplitude,
        effect.wave_frequency,
        effect.wave_speed,
        effect.wave_dispersion,
        0,
    );
    let (flat_h, flat_h_u, flat_h_v) = water_height_and_gradient(
        u,
        v,
        effect.time,
        (effect.flow_longitudinal, effect.flow_meridional),
        &modes,
    );

    assert_eq!(
        (total_h, total_h_u, total_h_v),
        (flat_h, flat_h_u, flat_h_v)
    );

    let laplace_beltrami_modes = build_laplace_beltrami_modes(&effect);
    assert!(
        laplace_beltrami_modes
            .iter()
            .flatten()
            .all(|&value| value == 0.0),
        "laplace_beltrami_modes should be all zeros when wave_lb_blend=0"
    );
}

#[test]
fn test_laplace_beltrami_gradient_matches_central_difference() {
    let effect = WaterTorusEffect {
        wave_lb_blend: 0.5,
        ..WaterTorusEffect::default()
    };

    let (u, v) = (1.2f32, 0.8f32);
    let delta = 1e-3;

    let (_, h_u, h_v) = water_total_height_and_gradient(&effect, u, v, 0);

    let (h_u_ahead, ..) = water_total_height_and_gradient(&effect, u + delta, v, 0);
    let (h_u_behind, ..) = water_total_height_and_gradient(&effect, u - delta, v, 0);
    let h_u_central = (h_u_ahead - h_u_behind) / (2.0 * delta);

    let (h_v_ahead, ..) = water_total_height_and_gradient(&effect, u, v + delta, 0);
    let (h_v_behind, ..) = water_total_height_and_gradient(&effect, u, v - delta, 0);
    let h_v_central = (h_v_ahead - h_v_behind) / (2.0 * delta);

    let relative_error = |analytic: f32, central: f32| (analytic - central).abs() / central.abs();

    assert!(
        relative_error(h_u, h_u_central) < 1e-2,
        "h_u={h_u:.8} central={h_u_central:.8}"
    );
    assert!(
        relative_error(h_v, h_v_central) < 1e-2,
        "h_v={h_v:.8} central={h_v_central:.8}"
    );
}

#[test]
fn test_build_laplace_beltrami_modes_matches_cached_modes_and_amplitude_sum() {
    let effect = WaterTorusEffect {
        wave_lb_blend: 0.5,
        ..WaterTorusEffect::default()
    };

    let laplace_beltrami_modes = build_laplace_beltrami_modes(&effect);
    let cached = compute_laplace_beltrami_modes_cached(effect.major_radius, effect.minor_radius);

    for (k, mode) in cached.iter().enumerate() {
        let packed_m = laplace_beltrami_modes[LAPLACE_BELTRAMI_SLOTS_PER_MODE * k][0];
        assert_eq!(
            packed_m, mode.m as f32,
            "slot {k}: packed m={packed_m} != cached m={}",
            mode.m
        );
    }

    let amplitude_sum: f32 = (0..LAPLACE_BELTRAMI_MODE_COUNT)
        .map(|k| laplace_beltrami_modes[LAPLACE_BELTRAMI_SLOTS_PER_MODE * k][2])
        .sum();
    let expected_sum = effect.wave_amplitude * effect.wave_lb_blend;
    assert!(
        (amplitude_sum - expected_sum).abs() < 1e-6,
        "lb amplitude sum {amplitude_sum:.8} should be ~{expected_sum:.8}"
    );
}

#[test]
fn test_pick_torus_identity_transform() {
    let major_radius = 5.0;
    let minor_radius = 1.0;
    let model = Matrix4::identity();
    let inverse_model = model.invert().unwrap();

    let ray_origin = Vector3::new(0.0, 0.0, -10.0);
    let ray_dir = Vector3::new(0.0, 0.0, 1.0);

    let hit = pick_torus(
        ray_origin,
        ray_dir,
        model,
        inverse_model,
        major_radius,
        minor_radius,
    );
    assert!(hit.is_some());

    let t = hit.unwrap();
    let expected_t = 10.0 - (major_radius + minor_radius);
    assert!(
        (t - expected_t).abs() < 1e-4,
        "pick hit={:.4} expected={:.4}",
        t,
        expected_t
    );
}

#[test]
fn test_wave_modes_determinism() {
    let wave_amplitude = 0.02;
    let wave_frequency = 6.0;
    let wave_speed = 1.0;

    // Determinism: same args -> same modes
    let modes_a = generate_water_wave_modes(wave_amplitude, wave_frequency, wave_speed, 0.0, 0);
    let modes_b = generate_water_wave_modes(wave_amplitude, wave_frequency, wave_speed, 0.0, 0);
    assert_eq!(modes_a, modes_b, "modes should be deterministic");

    // (m, n) != (0, 0) for all modes
    for (i, mode) in modes_a.iter().enumerate() {
        assert!(mode.m != 0 || mode.n != 0, "mode[{}] has (m,n)=(0,0)", i);
    }

    // Σ amplitude ≈ wave_amplitude
    let sum: f32 = modes_a.iter().map(|m| m.amplitude).sum();
    assert!(
        (sum - wave_amplitude).abs() < 1e-4,
        "sum of amplitudes={:.6}, expected={:.6}",
        sum,
        wave_amplitude
    );
}

#[test]
fn test_wave_numerical_gradient() {
    let modes = generate_water_wave_modes(0.02, 6.0, 1.0, 0.0, 0);
    let flow = (0.2, 0.0);
    let u = 0.5;
    let v = 0.3;
    let time = 1.0;

    let (_h, h_u, h_v) = water_height_and_gradient(u, v, time, flow, &modes);

    let delta = 1e-3;
    let (h_u_num, _, _) = water_height_and_gradient(u + delta, v, time, flow, &modes);
    let (h_u_ref, _, _) = water_height_and_gradient(u - delta, v, time, flow, &modes);
    let h_u_central = (h_u_num - h_u_ref) / (2.0 * delta);

    let (h_v_num, ..) = water_height_and_gradient(u, v + delta, time, flow, &modes);
    let (h_v_ref, ..) = water_height_and_gradient(u, v - delta, time, flow, &modes);
    let h_v_central = (h_v_num - h_v_ref) / (2.0 * delta);

    assert!(
        (h_u - h_u_central).abs() < 1e-3,
        "h_u={:.6}, central={:.6}",
        h_u,
        h_u_central
    );
    assert!(
        (h_v - h_v_central).abs() < 1e-3,
        "h_v={:.6}, central={:.6}",
        h_v,
        h_v_central
    );
}

#[test]
fn test_wave_periodicity() {
    let modes = generate_water_wave_modes(0.02, 6.0, 1.0, 0.0, 0);
    let flow = (0.2, 0.0);
    let time = 1.0;

    for _ in 0..100 {
        let u: f32 = fastrand::f32() * 10.0;
        let v: f32 = fastrand::f32() * 10.0;

        let (h, _, _) = water_height_and_gradient(u, v, time, flow, &modes);
        let (h_shifted, _, _) = water_height_and_gradient(
            u + 2.0 * std::f32::consts::PI,
            v + 2.0 * std::f32::consts::PI,
            time,
            flow,
            &modes,
        );

        assert!(
            (h - h_shifted).abs() < 1e-4,
            "h={:.6}, h_shifted={:.6}, diff={:.8}",
            h,
            h_shifted,
            (h - h_shifted).abs()
        );
    }
}

#[test]
fn test_perturbed_normal_identity() {
    let major_radius = 5.0;
    let minor_radius = 1.0;

    for _ in 0..100 {
        let u: f32 = fastrand::f32() * 2.0 * std::f32::consts::PI;
        let v: f32 = fastrand::f32() * 2.0 * std::f32::consts::PI;

        // When h = h_u = h_v = 0, perturbed normal should match torus_surface_normal
        let n_perturbed = water_perturbed_normal(u, v, 0.0, 0.0, 0.0, major_radius, minor_radius);
        let n_expected = torus_surface_normal(u, v);

        let diff = (n_perturbed - n_expected).magnitude();
        assert!(
            diff < 1e-6,
            "perturbed normal differs from surface normal by {:.8}",
            diff
        );

        // Should be unit length
        let mag = n_perturbed.magnitude();
        assert!(
            (mag - 1.0).abs() < 1e-6,
            "perturbed normal magnitude={:.8}",
            mag
        );
    }
}
