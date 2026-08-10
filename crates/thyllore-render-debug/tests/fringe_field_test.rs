use thyllore_render_core::flame_wave::*;
use thyllore_render_debug::fringe_field::{flow_warp_with_rate, sample_wave_field, WarpParams};
use thyllore_render_debug::flame_wave_mirror::{
    evaluate_wave_displacement_warp_with_rate, evaluate_wave_flow_warp_with_rate,
    evaluate_wave_noise_local_lowpass_reduced,
};

/// 5 deterministic (w, rate) pairs for testing.
const TEST_CASES: &[[f32; 6]] = &[
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 2.0, 3.0, 0.5, 0.5, 0.5],
    [-1.5, 0.7, -0.3, 0.0, 1.0, 0.0],
    [0.5, -0.5, 1.0, -0.25, 0.0, 0.75],
    [3.14, 0.0, 0.0, 0.0, 0.0, 0.0],
];

#[test]
fn test_sample_wave_field_matches_mirror() {
    let modes = generate_wave_modes();
    let node_spacing = 0.05;
    let eddy_time = 1.234;

    for case in TEST_CASES {
        let w = [case[0], case[1], case[2]];
        let rate = [case[3], case[4], case[5]];

        let sample = sample_wave_field(&modes, w, rate, node_spacing, eddy_time, [0.0, 0.0], 0.0);
        let (noise_ref, sigma_ref) =
            evaluate_wave_noise_local_lowpass_reduced(&modes, w, rate, node_spacing, eddy_time, None, [0.0, 0.0], 0.0);

        assert!(
            sample.noise == noise_ref,
            "w={:?} rate={:?}: noise mismatch: got {:?}, expected {:?}",
            w, rate, sample.noise, noise_ref
        );
        assert!(
            sample.sigma == sigma_ref,
            "w={:?} rate={:?}: sigma mismatch: got {:?}, expected {:?}",
            w, rate, sample.sigma, sigma_ref
        );
    }
}

#[test]
fn test_mode_contrib_sum_matches_z() {
    let modes = generate_wave_modes();
    let node_spacing = 0.05;
    let eddy_time = 1.234;

    for case in TEST_CASES {
        let w = [case[0], case[1], case[2]];
        let rate = [case[3], case[4], case[5]];

        let sample = sample_wave_field(&modes, w, rate, node_spacing, eddy_time, [0.0, 0.0], 0.0);
        let contrib_sum: f32 = sample.mode_contrib.iter().sum();

        assert!(
            (contrib_sum - sample.z).abs() < 1e-5,
            "w={:?} rate={:?}: mode_contrib sum {:?} vs z {:?}, diff {:?}",
            w, rate, contrib_sum, sample.z, contrib_sum - sample.z
        );
    }
}

#[test]
fn test_flow_warp_with_rate_matches_mirror_zero_advect() {
    let warp_modes = generate_wave_warp_modes();
    let warp_freq = 2.0;
    let strengths = [0.16, 0.45, 0.79];

    for strength in &strengths {
        let strength = *strength;
        let pb = [1.0, 2.0, 3.0];
        let dir = [0.5, -0.3, 0.8];

        // Constant strain profile (s_base = s_tip, 1/K = 1) so the profile
        // reduces to `strength` for any h and matches the parametric mirror.
        let params = WarpParams {
            strain_params: [strength, strength, 0.0, 1.0],
            warp_freq,
            advect: [0.0, 0.0, 0.0],
            aniso_axis_advect: 0.0,
            height_primitive: [[0.0; 4]; 3],
            mu_zw: [0.0, 0.0],
            displacement_form: false,
        };
        let h = 0.5;

        let (q_new, rate_new) = flow_warp_with_rate(&warp_modes, &params, pb, dir, h);
        let (q_ref, rate_ref) = evaluate_wave_flow_warp_with_rate(
            &warp_modes, pb, dir, warp_freq, 0.35, strength,
        );

        assert!(
            q_new[0] == q_ref[0] && q_new[1] == q_ref[1] && q_new[2] == q_ref[2],
            "h={}: q mismatch: got [{:?},{:?},{:?}], expected [{:?},{:?},{:?}]",
            h, q_new[0], q_new[1], q_new[2], q_ref[0], q_ref[1], q_ref[2]
        );
        assert!(
            rate_new[0] == rate_ref[0] && rate_new[1] == rate_ref[1] && rate_new[2] == rate_ref[2],
            "h={}: rate mismatch: got [{:?},{:?},{:?}], expected [{:?},{:?},{:?}]",
            h, rate_new[0], rate_new[1], rate_new[2], rate_ref[0], rate_ref[1], rate_ref[2]
        );
    }
}

#[test]
fn test_flow_warp_displacement_form_matches_mirror() {
    let warp_modes = generate_wave_warp_modes();
    let warp_freq = 2.0;
    for strength in [0.05f32, 0.12, 0.2] {
        let pb = [1.0, 2.0, 3.0];
        let dir = [0.5, -0.3, 0.8];
        let params = WarpParams {
            strain_params: [strength, strength, 0.0, 1.0],
            warp_freq,
            advect: [0.0, 0.0, 0.0],
            aniso_axis_advect: 0.0,
            height_primitive: [[0.0; 4]; 3],
            mu_zw: [0.0, 0.0],
            displacement_form: true,
        };
        let (q_new, rate_new) = flow_warp_with_rate(&warp_modes, &params, pb, dir, 0.5);
        let (q_ref, rate_ref) = evaluate_wave_displacement_warp_with_rate(
            &warp_modes, pb, dir, warp_freq, 0.35, strength,
        );
        assert_eq!(q_new, q_ref, "strength {strength}: q mismatch");
        assert_eq!(rate_new, rate_ref, "strength {strength}: rate mismatch");
    }
}
