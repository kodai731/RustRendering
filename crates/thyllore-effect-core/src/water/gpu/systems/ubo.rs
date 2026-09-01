use crate::water::analytic::lb_basis::compute_lb_modes_cached;
use crate::water::analytic::{generate_water_wave_modes, next_unit_f64, WATER_WAVE_MODE_COUNT};
use crate::water::effect::WaterTorusEffect;
use crate::water::*;
use cgmath::{Matrix4, SquareMatrix};

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

const LB_MODE_COUNT: usize = 4;
const LB_SLOTS_PER_MODE: usize = 5;
const LB_PHASE_SEED: u64 = 999;

fn lb_relative_weight(mode_index: usize) -> f32 {
    (-(mode_index as f32) / 2.0).exp2()
}

fn build_lb_modes(effect: &WaterTorusEffect) -> [[f32; 4]; 20] {
    let mut lb_modes = [[0.0f32; 4]; 20];
    if effect.wave_lb_blend <= 0.0 {
        return lb_modes;
    }

    let modes = compute_lb_modes_cached(effect.major_radius, effect.minor_radius);
    let weight_sum: f32 = (0..LB_MODE_COUNT).map(lb_relative_weight).sum();
    let amplitude_sum = effect.wave_amplitude * effect.wave_lb_blend;
    let mut phase_state = LB_PHASE_SEED;

    for (k, mode) in modes.iter().enumerate() {
        let slot = LB_SLOTS_PER_MODE * k;
        let amplitude = amplitude_sum * lb_relative_weight(k) / weight_sum;
        let omega = effect.wave_speed * (mode.lambda as f32).sqrt().sqrt();
        let phase = (next_unit_f64(&mut phase_state) * 2.0 * std::f64::consts::PI) as f32;

        lb_modes[slot] = [mode.m as f32, omega, amplitude, phase];

        for i in 0..4 {
            lb_modes[slot + 1][i] = mode.phi_cheb[i];
            lb_modes[slot + 2][i] = mode.phi_cheb[i + 4];
            lb_modes[slot + 3][i] = mode.dphi_cheb[i];
            lb_modes[slot + 4][i] = mode.dphi_cheb[i + 4];
        }
    }

    lb_modes
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
        temporal: [0.0, 0.0, 0.0, 0.0],
        wave_modes,
        inv_view_proj: Matrix4::identity(),
        lb_modes: build_lb_modes(effect),
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

    fn lb_amplitude_sum(ubo: &WaterUBO) -> f32 {
        (0..LB_MODE_COUNT)
            .map(|k| ubo.lb_modes[k * LB_SLOTS_PER_MODE][2])
            .sum()
    }

    #[test]
    fn test_zero_blend_zeroes_lb_modes_and_preserves_flat_modes() {
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
    fn test_half_blend_splits_amplitude_between_lb_and_flat_modes() {
        let effect = WaterTorusEffect {
            wave_lb_blend: 0.5,
            ..WaterTorusEffect::default()
        };
        let ubo = build_water_ubo(&effect, 0);
        let expected_sum = effect.wave_amplitude * 0.5;

        assert_eq!(ubo.lb_modes[0][0], 1.0);
        for k in 1..LB_MODE_COUNT {
            let ratio = ubo.lb_modes[k * LB_SLOTS_PER_MODE][2]
                / ubo.lb_modes[(k - 1) * LB_SLOTS_PER_MODE][2];
            assert!(
                (ratio - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
                "amplitude ratio at k={k} is {ratio}, expected 2^-1/2"
            );
        }
        assert!(
            (lb_amplitude_sum(&ubo) - expected_sum).abs() < 1e-6,
            "lb amplitude sum {} should be ~{expected_sum}",
            lb_amplitude_sum(&ubo)
        );
        assert!(
            (flat_amplitude_sum(&ubo) - expected_sum).abs() < 1e-6,
            "flat amplitude sum {} should be ~{expected_sum}",
            flat_amplitude_sum(&ubo)
        );
    }
}
