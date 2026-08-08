//! Pure-function layer of the wave field: sample_wave_field records per-mode
//! contributions, flow_warp_with_rate is the anisoCompress/Expand warp from
//! flame_noise_field.glsl.

use thyllore_render_core::flame_wave::*;

/// Per-mode contribution and weight from [`sample_wave_field`].
#[derive(Clone, Debug)]
pub struct FieldSample {
    pub noise: f32,
    pub sigma: f32,
    pub z: f32,
    pub mode_contrib: Vec<f32>,
    pub mode_weight: Vec<f32>,
}

/// Warp parameters for [`flow_warp_with_rate`].
#[derive(Clone, Copy, Debug)]
pub struct WarpParams {
    pub warp_amp: f32,
    pub warp_freq: f32,
    pub advect: [f32; 3],
    pub aniso_axis_advect: f32,
}

/// Anisotropic compression along the advection axis (mirror of
/// `flameAnisoCompress` in flame_noise_field.glsl).
/// Compute the anisotropy axis from advect and aniso_axis_advect.
fn compute_axis(advect: [f32; 3], aniso_axis_advect: f32) -> [f32; 3] {
    let default_axis = [0.0, 1.0, 0.0];
    let advect_sq = advect[0] * advect[0] + advect[1] * advect[1] + advect[2] * advect[2];
    if aniso_axis_advect > 0.0 && advect_sq > 1e-8 {
        let inv_len = 1.0 / advect_sq.sqrt();
        let norm_advect = [advect[0] * inv_len, advect[1] * inv_len, advect[2] * inv_len];
        let t = aniso_axis_advect.min(1.0).max(0.0);
        [
            default_axis[0] + (norm_advect[0] - default_axis[0]) * t,
            default_axis[1] + (norm_advect[1] - default_axis[1]) * t,
            default_axis[2] + (norm_advect[2] - default_axis[2]) * t,
        ]
    } else {
        default_axis
    }
}

/// Sample the wave field at w with rate of change along rate, recording each
/// mode's contribution and low-pass weight.
pub fn sample_wave_field(
    modes: &[WaveMode],
    w: [f32; 3],
    rate: [f32; 3],
    node_spacing: f32,
    eddy_time: f32,
    skipped_power: [f32; 2],
    skipped_env_coeff: f32,
) -> FieldSample {
    let (jitter_psi, jitter_psi_rate) = wave_jitter_state(w, rate);
    let carrier = |_index: usize, mode: &WaveMode| {
        let angle = mode.k[0] * w[0]
            + mode.k[1] * w[1]
            + mode.k[2] * w[2]
            + mode.phase
            + mode.eddy_rate * eddy_time
            + wave_mode_jitter_phase(&mode.jitter, &jitter_psi);
        let beta_carrier = mode.k[0] * rate[0]
            + mode.k[1] * rate[1]
            + mode.k[2] * rate[2]
            + wave_mode_jitter_phase(&mode.jitter, &jitter_psi_rate);
        let value = angle.sin();
        let x = beta_carrier * node_spacing / std::f32::consts::PI;
        let weight = (-(x * x) * (x * x)).exp();
        (value, weight)
    };

    let mut z_low = 0.0f32;
    let mut unresolved_power = 0.0f32;
    let mut mode_contrib: Vec<f32> = Vec::new();
    let mut mode_weight: Vec<f32> = Vec::new();

    // Low-octave modes (env_coeff == 0)
    for (index, mode) in modes.iter().enumerate().filter(|(_, m)| m.env_coeff == 0.0) {
        let (value, weight) = carrier(index, mode);
        let contrib = weight * mode.amplitude * value;
        z_low += contrib;
        unresolved_power += 0.5 * mode.amplitude * mode.amplitude * (1.0 - weight * weight);
        mode_contrib.push(contrib);
        mode_weight.push(weight);
    }

    let mut z = z_low;
    // High-octave modes (env_coeff != 0)
    for (index, mode) in modes.iter().enumerate().filter(|(_, m)| m.env_coeff != 0.0) {
        let (value, weight) = carrier(index, mode);
        let envelope = 1.0 + mode.env_coeff * z_low;
        let contrib = envelope * weight * mode.amplitude * value;
        z += contrib;
        unresolved_power +=
            envelope * envelope * 0.5 * mode.amplitude * mode.amplitude * (1.0 - weight * weight);
        mode_contrib.push(contrib);
        mode_weight.push(weight);
    }

    unresolved_power += skipped_power[0];
    let env_skip = 1.0 + skipped_env_coeff * z_low;
    unresolved_power += skipped_power[1] * env_skip * env_skip;

    let (inverse_scale, amplitude) = crate::flame_wave_mirror::get_wave_shaping_params_cached();
    let noise = WAVE_NOISE_MEAN + amplitude * (z * inverse_scale).tanh();
    let sigma_noise = unresolved_power.sqrt();

    FieldSample {
        noise,
        sigma: sigma_noise,
        z,
        mode_contrib,
        mode_weight,
    }
}

/// Flow warp with rate using anisotropic compression/expansion (mirror of
/// Flow warp with rate using anisotropic compression/expansion (mirror of
/// `flameWaveFlowWarpRate` in flame_noise_field.glsl).
pub fn flow_warp_with_rate(
    warp_modes: &[WaveWarpMode],
    p: &WarpParams,
    pb: [f32; 3],
    dir: [f32; 3],
    h: f32,
) -> ([f32; 3], [f32; 3]) {
    let amp = p.warp_amp * (0.15 + (1.0 - 0.15) * h);
    if amp == 0.0 || warp_modes.is_empty() {
        return (pb, dir);
    }

    let axis = compute_axis(p.advect, p.aniso_axis_advect);

    // M: warp space — anisoCompress(pb, 0.35) * warp_freq - advect
    let z_compressed = aniso_compress_f64(pb, 0.35, axis);
    let mut z = [
        z_compressed[0] * p.warp_freq as f64 - p.advect[0] as f64,
        z_compressed[1] * p.warp_freq as f64 - p.advect[1] as f64,
        z_compressed[2] * p.warp_freq as f64 - p.advect[2] as f64,
    ];

    // v starts as dir transformed by linear part of M: anisoCompress(dir, 0.35) * warp_freq
    let v_compressed = aniso_compress_f64(dir, 0.35, axis);
    let mut v = [
        v_compressed[0] * p.warp_freq as f64,
        v_compressed[1] * p.warp_freq as f64,
        v_compressed[2] * p.warp_freq as f64,
    ];

    let strength = amp as f64 * WAVE_SHEAR_STRENGTH_SCALE as f64;

    for mode in warp_modes {
        let angle = mode.k[0] as f64 * z[0]
            + mode.k[1] as f64 * z[1]
            + mode.k[2] as f64 * z[2]
            + mode.phase as f64;
        let shear_value = strength * mode.amplitude as f64 * angle.cos();
        let f_prime = -strength * mode.amplitude as f64 * angle.sin();
        let k_dot_v = mode.k[0] as f64 * v[0]
            + mode.k[1] as f64 * v[1]
            + mode.k[2] as f64 * v[2];

        z[0] += mode.curl_direction[0] as f64 * shear_value;
        z[1] += mode.curl_direction[1] as f64 * shear_value;
        z[2] += mode.curl_direction[2] as f64 * shear_value;

        v[0] += mode.curl_direction[0] as f64 * f_prime * k_dot_v;
        v[1] += mode.curl_direction[1] as f64 * f_prime * k_dot_v;
        v[2] += mode.curl_direction[2] as f64 * f_prime * k_dot_v;
    }

    // M^-1: back to physical space
    let warp_freq = p.warp_freq as f64;
    let q_pre = [
        z[0] / warp_freq + p.advect[0] as f64 / warp_freq,
        z[1] / warp_freq + p.advect[1] as f64 / warp_freq,
        z[2] / warp_freq + p.advect[2] as f64 / warp_freq,
    ];
    let q = aniso_expand_f64(q_pre, 0.35, axis);

    let rate_pre = [v[0] / warp_freq, v[1] / warp_freq, v[2] / warp_freq];
    let rate = aniso_expand_f64(rate_pre, 0.35, axis);

    (
        [q[0] as f32, q[1] as f32, q[2] as f32],
        [rate[0] as f32, rate[1] as f32, rate[2] as f32],
    )
}

/// Anisotropic compression along the advection axis (f64 version).
fn aniso_compress_f64(v: [f32; 3], axial_scale: f32, axis: [f32; 3]) -> [f64; 3] {
    let v0 = v[0] as f64;
    let v1 = v[1] as f64;
    let v2 = v[2] as f64;
    let a0 = axis[0] as f64;
    let a1 = axis[1] as f64;
    let a2 = axis[2] as f64;
    let dot = v0 * a0 + v1 * a1 + v2 * a2;
    let factor = 1.0 - axial_scale as f64;
    [v0 - dot * a0 * factor, v1 - dot * a1 * factor, v2 - dot * a2 * factor]
}

/// Inverse of aniso_compress_f64 — anisotropic expansion along the advection axis.
fn aniso_expand_f64(v: [f64; 3], axial_scale: f32, axis: [f32; 3]) -> [f64; 3] {
    let a0 = axis[0] as f64;
    let a1 = axis[1] as f64;
    let a2 = axis[2] as f64;
    let dot = v[0] * a0 + v[1] * a1 + v[2] * a2;
    let factor = 1.0 / axial_scale as f64 - 1.0;
    [v[0] + dot * a0 * factor, v[1] + dot * a1 * factor, v[2] + dot * a2 * factor]
}
