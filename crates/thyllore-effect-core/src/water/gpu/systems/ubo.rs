use crate::water::analytic::laplace_beltrami_basis::{
    compute_laplace_beltrami_modes_cached, LAPLACE_BELTRAMI_MODE_COUNT,
    LAPLACE_BELTRAMI_SLOTS_PER_MODE,
};
use crate::water::analytic::{generate_water_wave_modes, WATER_WAVE_MODE_COUNT};
use crate::water::effect::WaterTorusEffect;
use crate::water::*;
use cgmath::{Matrix4, SquareMatrix};
use thyllore_math_core::LinearCongruentialGenerator;

/// Compute inverse(proj * view) using f64 precision to minimize fp32 rounding error.
/// Returns the result as Matrix4<f32>.
pub fn inverse_view_proj_f64(proj: Matrix4<f32>, view: Matrix4<f32>) -> Matrix4<f32> {
    let p: Matrix4<f64> = proj.cast().unwrap();
    let v: Matrix4<f64> = view.cast().unwrap();
    let inv = (p * v)
        .invert()
        .expect("view-proj matrix must be invertible");
    inv.cast().unwrap()
}

const LAPLACE_BELTRAMI_PHASE_SEED: u64 = 999;

fn laplace_beltrami_relative_weight(mode_index: usize) -> f32 {
    (-(mode_index as f32) / 2.0).exp2()
}

pub fn build_laplace_beltrami_modes(effect: &WaterTorusEffect) -> [[f32; 4]; 20] {
    let mut packed_modes = [[0.0f32; 4]; 20];
    if effect.wave_lb_blend <= 0.0 {
        return packed_modes;
    }

    let modes = compute_laplace_beltrami_modes_cached(effect.major_radius, effect.minor_radius);
    let weight_sum: f32 = (0..LAPLACE_BELTRAMI_MODE_COUNT)
        .map(laplace_beltrami_relative_weight)
        .sum();
    let amplitude_sum = effect.wave_amplitude * effect.wave_lb_blend;
    let mut phase_random = LinearCongruentialGenerator::from_seed(LAPLACE_BELTRAMI_PHASE_SEED);

    for (k, mode) in modes.iter().enumerate() {
        let slot = LAPLACE_BELTRAMI_SLOTS_PER_MODE * k;
        let amplitude = amplitude_sum * laplace_beltrami_relative_weight(k) / weight_sum;
        let omega = effect.wave_speed * (mode.lambda as f32).sqrt().sqrt();
        let phase = phase_random.next_angle_f32();

        packed_modes[slot] = [mode.m as f32, omega, amplitude, phase];

        for i in 0..4 {
            packed_modes[slot + 1][i] = mode.phi_cheb[i];
            packed_modes[slot + 2][i] = mode.phi_cheb[i + 4];
            packed_modes[slot + 3][i] = mode.dphi_cheb[i];
            packed_modes[slot + 4][i] = mode.dphi_cheb[i + 4];
        }
    }

    packed_modes
}

pub fn build_water_ubo(effect: &WaterTorusEffect, frame_index: u32) -> WaterUBO {
    let model = build_water_model_matrix(effect);
    let inverse_model = model.invert().unwrap_or(Matrix4::identity());

    let modes = generate_water_wave_modes(
        effect.wave_amplitude * (1.0 - effect.wave_lb_blend),
        effect.wave_frequency,
        effect.wave_speed,
        effect.wave_dispersion,
        frame_index,
    );

    let mut wave_modes: [[f32; 4]; 16] = [[0.0; 4]; 16];
    for (k, mode) in modes.iter().enumerate() {
        let i = k * 2;
        wave_modes[i][0] = mode.m as f32;
        wave_modes[i][1] = mode.n as f32;
        wave_modes[i][2] = mode.amplitude;
        wave_modes[i][3] = mode.omega;
        if i + 1 < 16 {
            wave_modes[i + 1][0] = mode.phase;
            wave_modes[i + 1][1] = 0.0;
            wave_modes[i + 1][2] = 0.0;
            wave_modes[i + 1][3] = 0.0;
        }
    }

    WaterUBO {
        model,
        inverse_model,
        radii: [
            effect.major_radius,
            effect.minor_radius,
            effect.caustic_strength,
            0.0,
        ],
        absorption: [
            effect.absorption[0],
            effect.absorption[1],
            effect.absorption[2],
            effect.ior,
        ],
        flow: [
            effect.flow_longitudinal,
            effect.flow_meridional,
            effect.time,
            0.0,
        ],
        composite: [
            effect.reflect_strength,
            effect.refract_strength,
            WATER_WAVE_MODE_COUNT as f32,
            0.0,
        ],
        tint: [effect.tint[0], effect.tint[1], effect.tint[2], 0.0],
        lighting: [
            effect.light_intensity,
            effect.highlight_sharpness,
            effect.sky_brightness,
            effect.scatter_strength,
        ],
        scattering: [effect.scatter_anisotropy, 0.0, 0.0, 0.0],
        temporal: [0.0, 0.0, 0.0, 0.0],
        wave_modes,
        inv_view_proj: Matrix4::identity(),
        lb_modes: build_laplace_beltrami_modes(effect),
    }
}

impl Default for WaterUBO {
    fn default() -> Self {
        build_water_ubo(&WaterTorusEffect::default(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_amplitude_sum(ubo: &WaterUBO) -> f32 {
        (0..WATER_WAVE_MODE_COUNT)
            .map(|k| ubo.wave_modes[k * 2][2])
            .sum()
    }

    fn laplace_beltrami_amplitude_sum(ubo: &WaterUBO) -> f32 {
        (0..LAPLACE_BELTRAMI_MODE_COUNT)
            .map(|k| ubo.lb_modes[k * LAPLACE_BELTRAMI_SLOTS_PER_MODE][2])
            .sum()
    }

    #[test]
    fn test_zero_blend_zeroes_laplace_beltrami_modes_and_preserves_flat_modes() {
        let effect = WaterTorusEffect::default();
        let ubo = build_water_ubo(&effect, 0);

        assert!(ubo.lb_modes.iter().flatten().all(|&value| value == 0.0));

        let expected = generate_water_wave_modes(
            effect.wave_amplitude,
            effect.wave_frequency,
            effect.wave_speed,
            effect.wave_dispersion,
            0,
        );
        for (k, mode) in expected.iter().enumerate() {
            let i = k * 2;
            assert_eq!(ubo.wave_modes[i][0], mode.m as f32);
            assert_eq!(ubo.wave_modes[i][1], mode.n as f32);
            assert_eq!(ubo.wave_modes[i][2], mode.amplitude);
            assert_eq!(ubo.wave_modes[i][3], mode.omega);
            assert_eq!(ubo.wave_modes[i + 1][0], mode.phase);
        }
    }

    #[test]
    fn test_half_blend_splits_amplitude_between_laplace_beltrami_and_flat_modes() {
        let effect = WaterTorusEffect {
            wave_lb_blend: 0.5,
            ..WaterTorusEffect::default()
        };
        let ubo = build_water_ubo(&effect, 0);
        let expected_sum = effect.wave_amplitude * 0.5;

        assert_eq!(ubo.lb_modes[0][0], 1.0);
        for k in 1..LAPLACE_BELTRAMI_MODE_COUNT {
            let ratio = ubo.lb_modes[k * LAPLACE_BELTRAMI_SLOTS_PER_MODE][2]
                / ubo.lb_modes[(k - 1) * LAPLACE_BELTRAMI_SLOTS_PER_MODE][2];
            assert!(
                (ratio - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
                "amplitude ratio at k={k} is {ratio}, expected 2^-1/2"
            );
        }
        assert!(
            (laplace_beltrami_amplitude_sum(&ubo) - expected_sum).abs() < 1e-6,
            "lb amplitude sum {} should be ~{expected_sum}",
            laplace_beltrami_amplitude_sum(&ubo)
        );
        assert!(
            (flat_amplitude_sum(&ubo) - expected_sum).abs() < 1e-6,
            "flat amplitude sum {} should be ~{expected_sum}",
            flat_amplitude_sum(&ubo)
        );
    }
}
