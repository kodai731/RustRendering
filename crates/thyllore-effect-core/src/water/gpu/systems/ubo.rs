use crate::water::analytic::{generate_water_wave_modes, WATER_WAVE_MODE_COUNT};
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

pub fn build_water_ubo(effect: &WaterTorusEffect, frame_index: u32) -> WaterUBO {
    let model = build_water_model_matrix(effect);
    let inverse_model = model.invert().unwrap_or(Matrix4::identity());

    let modes = generate_water_wave_modes(
        effect.wave_amplitude,
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
    }
}

impl Default for WaterUBO {
    fn default() -> Self {
        build_water_ubo(&WaterTorusEffect::default(), 0)
    }
}
