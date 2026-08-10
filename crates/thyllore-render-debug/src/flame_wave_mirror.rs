//! Debug-only CPU mirrors of the wave-basis flame field (GLSL
//! flame_noise_field.glsl / flame_radial_integral.glsl). Nothing here is used
//! by the product renderer — the GPU evaluates the field; these mirrors exist
//! so the shader math can be tested and diagnosed on the CPU.

use std::sync::OnceLock;

use thyllore_render_core::flame_wave::*;

/// Cached shaping parameters for the default `WAVE_TANH_SCALE`.
/// Computed once via `wave_shaping_params` and stored in a `OnceLock`.
pub fn get_wave_shaping_params_cached() -> (f32, f32) {
    static CACHED: OnceLock<(f32, f32)> = OnceLock::new();
    *CACHED.get_or_init(|| wave_shaping_params(WAVE_TANH_SCALE))
}

/// Derivative of tanh shaping: g'(z) = (amplitude / s) * (1 - tanh(z/s)^2).
/// Since tanh(z/s) = (shaped_noise - WAVE_NOISE_MEAN) / amplitude, this is computed from shaped_noise.
pub fn wave_shaping_derivative(shaped_noise: f32, inverse_scale: f32, amplitude: f32) -> f32 {
    let tanh_val = (shaped_noise - WAVE_NOISE_MEAN) / amplitude;
    let s = 1.0 / inverse_scale;
    (amplitude / s) * (1.0 - tanh_val * tanh_val)
}

/// Unscaled warp displacement at a warp coordinate (mirror of
/// `flameWaveWarpOffset` without the warp_amp * mix(0.15, 1, h) factor and the
/// warp_y_scale component scaling, which stay with the caller).
pub fn evaluate_wave_warp(modes: &[WaveWarpMode], wp: [f32; 3]) -> [f32; 3] {
    let mut displacement = [0.0f32; 3];
    for mode in modes {
        let angle =
            mode.k[0] * wp[0] + mode.k[1] * wp[1] + mode.k[2] * wp[2] + mode.phase;
        let value = mode.amplitude * angle.sin();
        displacement[0] += mode.curl_direction[0] * value;
        displacement[1] += mode.curl_direction[1] * value;
        displacement[2] += mode.curl_direction[2] * value;
    }
    displacement
}

/// Linear map M: scale y by axial_scale, then multiply all by warp_frequency.
fn wave_linear_map(p: [f64; 3], warp_frequency: f64, axial_scale: f64) -> [f64; 3] {
    [
        p[0] * warp_frequency,
        p[1] * axial_scale * warp_frequency,
        p[2] * warp_frequency,
    ]
}

/// Inverse of the linear map M.
fn wave_linear_map_inverse(p: [f64; 3], warp_frequency: f64, axial_scale: f64) -> [f64; 3] {
    [
        p[0] / warp_frequency,
        p[1] / (axial_scale * warp_frequency),
        p[2] / warp_frequency,
    ]
}

/// Flow map q = M_inv * (S_N ∘ ... ∘ S_1)(M * point), where each shear
/// S_i(z) = z + strength * a_i * cos(k_i·z + φ_i) * c_i is applied in index order;
/// f64 keeps the shear pockets (|J| ~ 100) resolved.
fn wave_flow_map(
    modes: &[WaveWarpMode],
    point: [f64; 3],
    warp_frequency: f64,
    axial_scale: f64,
    strength: f64,
) -> [f64; 3] {
    let mut z = wave_linear_map(point, warp_frequency, axial_scale);

    for mode in modes {
        let angle = mode.k[0] as f64 * z[0]
            + mode.k[1] as f64 * z[1]
            + mode.k[2] as f64 * z[2]
            + mode.phase as f64;
        let value = strength * mode.amplitude as f64 * angle.cos();
        for axis in 0..3 {
            z[axis] += mode.curl_direction[axis] as f64 * value;
        }
    }

    wave_linear_map_inverse(z, warp_frequency, axial_scale)
}

/// Evaluate the flow map warp at a warp coordinate: the divergence-free
/// counterpart of [`evaluate_wave_warp`], volume preserving by construction.
/// `strength` is the FINAL shear strength (the render path passes
/// flameWarpStrength(h) = strain(h) / K; no hidden scale is applied here).
pub fn evaluate_wave_flow_warp(
    modes: &[WaveWarpMode],
    point: [f32; 3],
    warp_frequency: f32,
    axial_scale: f32,
    strength: f32,
) -> [f32; 3] {
    let warped = wave_flow_map(
        modes,
        [point[0] as f64, point[1] as f64, point[2] as f64],
        warp_frequency as f64,
        axial_scale as f64,
        strength as f64,
    );
    [warped[0] as f32, warped[1] as f32, warped[2] as f32]
}

/// Like [`evaluate_wave_flow_warp`] but also returns the rate `dw/dt` of the
/// warped coordinate along a ray direction. The rate is computed by pushing
/// the direction vector through the Jacobian-vector product of each shear:
///   v += c_i * (f_i' * dot(k_i, v))
/// where f_i' = -strength * a_i * sin(k_i·z + φ_i).
/// Returns `(warp 後の点, レイ方向レート dw/dt)`.
pub fn evaluate_wave_flow_warp_with_rate(
    modes: &[WaveWarpMode],
    point: [f32; 3],
    direction: [f32; 3],
    warp_frequency: f32,
    axial_scale: f32,
    strength: f32,
) -> ([f32; 3], [f32; 3]) {
    let mut z = wave_linear_map(
        [point[0] as f64, point[1] as f64, point[2] as f64],
        warp_frequency as f64,
        axial_scale as f64,
    );
    // Initial v is direction transformed by the linear part of wave_linear_map.
    let mut v = [
        direction[0] as f64 * warp_frequency as f64,
        direction[1] as f64 * axial_scale as f64 * warp_frequency as f64,
        direction[2] as f64 * warp_frequency as f64,
    ];

    let strength_scaled = strength as f64;

    for mode in modes {
        let angle = mode.k[0] as f64 * z[0]
            + mode.k[1] as f64 * z[1]
            + mode.k[2] as f64 * z[2]
            + mode.phase as f64;
        // Update z: same as wave_flow_map
        let value = strength_scaled * mode.amplitude as f64 * angle.cos();
        for axis in 0..3 {
            z[axis] += mode.curl_direction[axis] as f64 * value;
        }
        // Update v: Jacobian-vector product
        // f_i' = -strength * a_i * sin(k_i·z + φ_i)
        let fp = -strength_scaled * mode.amplitude as f64 * angle.sin();
        let k_dot_v = mode.k[0] as f64 * v[0]
            + mode.k[1] as f64 * v[1]
            + mode.k[2] as f64 * v[2];
        for axis in 0..3 {
            v[axis] += mode.curl_direction[axis] as f64 * fp * k_dot_v;
        }
    }

    let warped_point = wave_linear_map_inverse(z, warp_frequency as f64, axial_scale as f64);
    // Transform v by the inverse linear map to get dw/dt in noise coordinate space.
    let rate = [
        v[0] / warp_frequency as f64,
        v[1] / (axial_scale as f64 * warp_frequency as f64),
        v[2] / warp_frequency as f64,
    ];

    (
        [warped_point[0] as f32, warped_point[1] as f32, warped_point[2] as f32],
        [rate[0] as f32, rate[1] as f32, rate[2] as f32],
    )
}

/// Displacement-form warp with rate (mirror of the flameWarpMapJvp
/// displacement branch): every mode evaluated at the input z, so the Jacobian
/// is the sum I + sum c_m f'_m k_m^T applied to the direction.
pub fn evaluate_wave_displacement_warp_with_rate(
    modes: &[WaveWarpMode],
    point: [f32; 3],
    direction: [f32; 3],
    warp_frequency: f32,
    axial_scale: f32,
    strength: f32,
) -> ([f32; 3], [f32; 3]) {
    let z0 = wave_linear_map(
        [point[0] as f64, point[1] as f64, point[2] as f64],
        warp_frequency as f64,
        axial_scale as f64,
    );
    let v0 = [
        direction[0] as f64 * warp_frequency as f64,
        direction[1] as f64 * axial_scale as f64 * warp_frequency as f64,
        direction[2] as f64 * warp_frequency as f64,
    ];
    let strength = strength as f64;
    let mut displacement = [0.0f64; 3];
    let mut rate_sum = [0.0f64; 3];
    for mode in modes {
        let angle = mode.k[0] as f64 * z0[0]
            + mode.k[1] as f64 * z0[1]
            + mode.k[2] as f64 * z0[2]
            + mode.phase as f64;
        let value = strength * mode.amplitude as f64 * angle.cos();
        let fp = -strength * mode.amplitude as f64 * angle.sin();
        let k_dot_v = mode.k[0] as f64 * v0[0]
            + mode.k[1] as f64 * v0[1]
            + mode.k[2] as f64 * v0[2];
        for axis in 0..3 {
            displacement[axis] += mode.curl_direction[axis] as f64 * value;
            rate_sum[axis] += mode.curl_direction[axis] as f64 * fp * k_dot_v;
        }
    }
    let z = [
        z0[0] + displacement[0],
        z0[1] + displacement[1],
        z0[2] + displacement[2],
    ];
    let v = [v0[0] + rate_sum[0], v0[1] + rate_sum[1], v0[2] + rate_sum[2]];
    let warped_point = wave_linear_map_inverse(z, warp_frequency as f64, axial_scale as f64);
    let rate = [
        v[0] / warp_frequency as f64,
        v[1] / (axial_scale as f64 * warp_frequency as f64),
        v[2] / warp_frequency as f64,
    ];
    (
        [warped_point[0] as f32, warped_point[1] as f32, warped_point[2] as f32],
        [rate[0] as f32, rate[1] as f32, rate[2] as f32],
    )
}

/// Clenshaw evaluation of a Chebyshev series at x in [-1, 1]
/// (mirror of `flameWaveCfClenshaw`).
pub fn evaluate_wave_cf_chebyshev(coeffs: &[f32; WAVE_CF_CHEB_COEFFS], x: f32) -> f32 {
    let t = 2.0 * x;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    for k in (1..WAVE_CF_CHEB_COEFFS).rev() {
        let b0 = t * b1 - b2 + coeffs[k];
        b2 = b1;
        b1 = b0;
    }
    x * b1 - b2 + coeffs[0]
}

/// Modulator displacement field and its ray rate at the unwarped warp
/// coordinate `m0` (direction `dm0` through the same linear map):
///   v    = sum_j a_j c_j sin(k_j . m0 + phi_j)
///   vdot = sum_j a_j c_j cos(k_j . m0 + phi_j) * (k_j . dm0)
/// (mirror of `flameWaveCfPsiVectors` before the aniso/frequency transform).
pub fn wave_cf_modulator_state(
    modulators: &[WaveWarpMode],
    m0: [f32; 3],
    dm0: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let mut v = [0.0f32; 3];
    let mut vdot = [0.0f32; 3];
    for mode in modulators {
        let angle = mode.k[0] * m0[0] + mode.k[1] * m0[1] + mode.k[2] * m0[2] + mode.phase;
        let s = mode.amplitude * angle.sin();
        let c = mode.amplitude
            * angle.cos()
            * (mode.k[0] * dm0[0] + mode.k[1] * dm0[1] + mode.k[2] * dm0[2]);
        for axis in 0..3 {
            v[axis] += mode.curl_direction[axis] * s;
            vdot[axis] += mode.curl_direction[axis] * c;
        }
    }
    (v, vdot)
}

/// Precomputed closed-form context (per parameter set, not per sample).
pub struct WaveCfContext {
    pub cheb_sin: [f32; WAVE_CF_CHEB_COEFFS],
    pub cheb_cos: [f32; WAVE_CF_CHEB_COEFFS],
    pub depth_scale: [f32; WAVE_MODE_COUNT],
}

/// Per-sample closed-form state: `psi_vector` / `rate_vector` are the
/// modulator field v / vdot pushed through the erosion coordinate transform
/// and scaled by the warp displacement amplitude, so per mode
/// psi_n = cap_scale_n * dot(k_n, psi_vector).
pub struct WaveCfSample<'a> {
    pub ctx: &'a WaveCfContext,
    pub psi_vector: [f32; 3],
    pub rate_vector: [f32; 3],
    pub amp_disp: f32,
}

impl WaveCfSample<'_> {
    fn mode_terms(&self, index: usize, k: [f32; 3]) -> (f32, f32) {
        let depth = self.amp_disp * self.ctx.depth_scale[index];
        let scale = if depth > WAVE_CF_CAP {
            WAVE_CF_CAP / depth
        } else {
            1.0
        };
        let psi =
            scale * (k[0] * self.psi_vector[0] + k[1] * self.psi_vector[1] + k[2] * self.psi_vector[2]);
        let rate = scale
            * (k[0] * self.rate_vector[0] + k[1] * self.rate_vector[1] + k[2] * self.rate_vector[2]);
        (psi, rate)
    }

    /// sin(angle + psi) via the family-T expansion sinA*T_c + cosA*T_s, plus
    /// the FM contribution to the phase rate along the ray.
    fn wave_value(&self, index: usize, k: [f32; 3], angle: f32) -> (f32, f32) {
        let (psi, rate) = self.mode_terms(index, k);
        let x = (psi / WAVE_CF_PSI_BOUND).clamp(-1.0, 1.0);
        let value = angle.sin() * evaluate_wave_cf_chebyshev(&self.ctx.cheb_cos, x)
            + angle.cos() * evaluate_wave_cf_chebyshev(&self.ctx.cheb_sin, x);
        (value, rate)
    }
}

/// Pointwise wave noise at a warped coordinate (mirror of `flameWaveNoiseSum`):
/// mean-matched to fbm3 so `flameNoiseErosionFromValue` keeps its calibration.
/// `eddy_time` is `noise_scroll_speed * time`.
pub fn evaluate_wave_noise(modes: &[WaveMode], w: [f32; 3], eddy_time: f32) -> f32 {
    evaluate_wave_noise_cf(modes, w, eddy_time, None)
}

/// Like [`evaluate_wave_noise`] but with the optional closed-form pseudo-FM
/// phase modulation applied to every carrier.
pub fn evaluate_wave_noise_cf(
    modes: &[WaveMode],
    w: [f32; 3],
    eddy_time: f32,
    cf: Option<&WaveCfSample>,
) -> f32 {
    let (jitter_psi, _) = wave_jitter_state(w, [0.0; 3]);
    let carrier = |index: usize, mode: &WaveMode| {
        let angle = mode.k[0] * w[0]
            + mode.k[1] * w[1]
            + mode.k[2] * w[2]
            + mode.phase
            + mode.eddy_rate * eddy_time
            + wave_mode_jitter_phase(&mode.jitter, &jitter_psi);
        match cf {
            Some(sample) => sample.wave_value(index, mode.k, angle).0,
            None => angle.sin(),
        }
    };
    let mut z_low = 0.0f32;
    for (index, mode) in modes.iter().enumerate().filter(|(_, m)| m.env_coeff == 0.0) {
        z_low += mode.amplitude * carrier(index, mode);
    }
    let mut z = z_low;
    for (index, mode) in modes.iter().enumerate().filter(|(_, m)| m.env_coeff != 0.0) {
        z += (1.0 + mode.env_coeff * z_low) * mode.amplitude * carrier(index, mode);
    }
    let (inverse_scale, amplitude) = get_wave_shaping_params_cached();
    WAVE_NOISE_MEAN + amplitude * (z * inverse_scale).tanh()
}


#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_math_core::{evaluate_erf_response, fit_erf_response};

    #[test]
    fn test_warp_modes_are_deterministic_and_calibrated() {
        let modes = generate_wave_warp_modes();
        assert_eq!(modes, generate_wave_warp_modes());

        let mut sums = [0.0f64; 3];
        let mut sums_sq = [0.0f64; 3];
        let count = 30usize;
        let total = (count * count * count) as f64;
        for ix in 0..count {
            for iy in 0..count {
                for iz in 0..count {
                    let wp = [
                        ix as f32 * 1.13 + 0.29,
                        iy as f32 * 1.07 + 0.41,
                        iz as f32 * 1.19 + 0.11,
                    ];
                    let displacement = evaluate_wave_warp(&modes, wp);
                    for axis in 0..3 {
                        sums[axis] += displacement[axis] as f64;
                        sums_sq[axis] += (displacement[axis] as f64).powi(2);
                    }
                }
            }
        }
        for axis in 0..3 {
            let mean = sums[axis] / total;
            let std = (sums_sq[axis] / total - mean * mean).sqrt();
            assert!(mean.abs() < 0.02, "axis {axis} mean {mean}");
            assert!(
                (std - WAVE_WARP_COMPONENT_STD as f64).abs() < 0.05,
                "axis {axis} std {std}"
            );
        }
    }

    #[test]
    fn test_field_statistics_match_fbm_reference() {
        let modes = generate_wave_modes();
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let count = 40usize;
        let total = (count * count * count) as f64;
        for ix in 0..count {
            for iy in 0..count {
                for iz in 0..count {
                    let w = [
                        ix as f32 * 0.83 + 0.31,
                        iy as f32 * 0.79 + 0.17,
                        iz as f32 * 0.87 + 0.53,
                    ];
                    let value = evaluate_wave_noise(&modes, w, 0.0) as f64;
                    sum += value;
                    sum_sq += value * value;
                }
            }
        }
        let mean = sum / total;
        let std = (sum_sq / total - mean * mean).sqrt();
        assert!((mean - WAVE_NOISE_MEAN as f64).abs() < 0.01, "mean {mean}");
        assert!(
            (std - WAVE_NOISE_STD as f64).abs() < 0.015,
            "std {std} vs {WAVE_NOISE_STD}"
        );
    }

    #[test]
    fn test_attenuation_uses_warped_rate() {
        // Non-divergence-free warp stretches in some directions and compresses in others,
        // so a single ray's warped sigma can be either larger or smaller than the unwarped one.
        // The claim is statistical: over many rays the mean warped sigma is larger (the long
        // tail of stretched rays dominates), while at least one ray is compressed.
        let modes = generate_wave_modes();
        let warp_modes = generate_wave_warp_modes();
        let warp_frequency = 1.0;
        let axial_scale = 0.35;
        let strength = 0.35;

        // Generate 64 rays using LCG (same pattern as flow_warp_sample_points).
        let mut state: u64 = 12345;
        let mut next_coordinate = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };

        let mut sum_sigma_unwarped: f32 = 0.0;
        let mut sum_sigma_warped: f32 = 0.0;
        let mut compressed_count: usize = 0;

        for _ in 0..64 {
            let origin: [f32; 3] = [next_coordinate(), next_coordinate(), next_coordinate()];
            let dir: [f32; 3] = [next_coordinate(), next_coordinate(), next_coordinate()];

            // (a) Unwarped rate: the ray direction transformed by the linear part of
            // wave_linear_map (no shear deformation), same as the initial v in
            // evaluate_wave_flow_warp_with_rate.
            let unwrapped_rate = [
                dir[0] * warp_frequency,
                dir[1] * axial_scale * warp_frequency,
                dir[2] * warp_frequency,
            ];
            let unwrapped_rates: [[f32; 3]; 3] = [unwrapped_rate, unwrapped_rate, unwrapped_rate];
            let (_weights_unwrapped, sigma_unwrapped) =
                wave_ray_attenuation(&modes, &unwrapped_rates, 0.1);

            // (b) Warped rate: compute at t0, midpoint, t1 of the segment.
            let (_warp_point_t0, warp_rate_t0) = evaluate_wave_flow_warp_with_rate(
                &warp_modes,
                origin,
                dir,
                warp_frequency,
                axial_scale,
                strength,
            );
            let mid_origin = [origin[0] + 0.05 * dir[0], origin[1] + 0.05 * dir[1], origin[2] + 0.05 * dir[2]];
            let (_warp_point_mid, warp_rate_mid) = evaluate_wave_flow_warp_with_rate(
                &warp_modes,
                mid_origin,
                dir,
                warp_frequency,
                axial_scale,
                strength,
            );
            let end_origin = [origin[0] + 0.1 * dir[0], origin[1] + 0.1 * dir[1], origin[2] + 0.1 * dir[2]];
            let (_warp_point_t1, warp_rate_t1) = evaluate_wave_flow_warp_with_rate(
                &warp_modes,
                end_origin,
                dir,
                warp_frequency,
                axial_scale,
                strength,
            );
            let warped_rates: [[f32; 3]; 3] = [warp_rate_t0, warp_rate_mid, warp_rate_t1];
            let (_weights_warped, sigma_warped) =
                wave_ray_attenuation(&modes, &warped_rates, 0.1);

            sum_sigma_unwarped += sigma_unwrapped;
            sum_sigma_warped += sigma_warped;
            if sigma_warped < sigma_unwrapped {
                compressed_count += 1;
            }
        }

        let mean_sigma_unwarped = sum_sigma_unwarped / 64.0;
        let mean_sigma_warped = sum_sigma_warped / 64.0;

        // Mean warped sigma should be larger (stretched rays dominate the average).
        assert!(
            mean_sigma_warped > mean_sigma_unwarped,
            "mean warped sigma {} should be larger than mean unwarped sigma {}",
            mean_sigma_warped,
            mean_sigma_unwarped
        );

        // At least one ray should be compressed (non-divergence-free warp compresses in some directions).
        assert!(
            compressed_count > 0,
            "expected at least one compressed ray, got {} (all rays were stretched)",
            compressed_count
        );
    }

    /// Closed form vs dense quadrature of the true pointwise field, in a
    /// regime where the nodes resolve every mode (the model's exact limit).

    /// Support edges landing strictly inside a segment must be resolved at the
    /// actual crossing, not at segment granularity: the closed form matches the
    /// support-masked dense quadrature for every sub-segment edge offset, and
    /// fully dead segments emit exactly zero.

    /// No steps between adjacent parallel rays: the closed form must vary
    /// smoothly as the ray sweeps (the band quantization this basis removes
    /// would show up as jumps).

    #[test]
    fn test_cf_chebyshev_tables_accurate() {
        let (cheb_sin, cheb_cos) = wave_cf_chebyshev_tables();
        let steps = 4001;
        let mut max_err_sin = 0.0f32;
        let mut max_err_cos = 0.0f32;
        for i in 0..steps {
            let x = -1.0 + 2.0 * i as f32 / (steps - 1) as f32;
            let psi = x * WAVE_CF_PSI_BOUND;
            max_err_sin =
                max_err_sin.max((evaluate_wave_cf_chebyshev(&cheb_sin, x) - psi.sin()).abs());
            max_err_cos =
                max_err_cos.max((evaluate_wave_cf_chebyshev(&cheb_cos, x) - psi.cos()).abs());
        }
        assert!(max_err_sin < 2e-4, "sin approximation error {max_err_sin}");
        assert!(max_err_cos < 2e-4, "cos approximation error {max_err_cos}");
    }

    /// Builds the closed-form sample at a point exactly like the shader chain
    /// (warp-side aniso 0.35, erosion-side aniso ANISO_Y, both diagonal in y).
    fn cf_sample_at<'a>(
        ctx: &'a WaveCfContext,
        modulators: &[WaveWarpMode],
        pb: [f32; 3],
        dir: [f32; 3],
        amp_disp: f32,
        warp_frequency: f32,
        noise_frequency: f32,
        noise_aniso_y: f32,
    ) -> WaveCfSample<'a> {
        let m0 = [pb[0] * warp_frequency, pb[1] * 0.35 * warp_frequency, pb[2] * warp_frequency];
        let dm0 = [dir[0] * warp_frequency, dir[1] * 0.35 * warp_frequency, dir[2] * warp_frequency];
        let (v, vdot) = wave_cf_modulator_state(modulators, m0, dm0);
        let scale = noise_frequency * amp_disp;
        WaveCfSample {
            ctx,
            psi_vector: [v[0] * scale, v[1] * noise_aniso_y * scale, v[2] * scale],
            rate_vector: [vdot[0] * scale, vdot[1] * noise_aniso_y * scale, vdot[2] * scale],
            amp_disp,
        }
    }

    fn cf_test_context(
        erosion_modes: &[WaveMode],
        modulators: &[WaveWarpMode],
        noise_frequency: f32,
        noise_aniso_y: f32,
    ) -> WaveCfContext {
        let (cheb_sin, cheb_cos) = wave_cf_chebyshev_tables();
        WaveCfContext {
            cheb_sin,
            cheb_cos,
            depth_scale: wave_cf_depth_scale(erosion_modes, modulators, noise_frequency, |v| {
                [v[0], v[1] * noise_aniso_y, v[2]]
            }),
        }
    }

    /// The pseudo-FM carriers keep the field statistics: sin(A + psi) has the
    /// same second moment as sin(A), so mean/std stay calibrated to fbm3.
    #[test]
    fn test_cf_field_statistics_match_reference() {
        let modes = generate_wave_modes();
        let modulators = generate_wave_warp_modes();
        let (warp_frequency, noise_frequency, aniso_y) = (5.0f32, 6.0f32, 0.35f32);
        let amp_disp = 1.4 * (0.15 + 0.85 * 0.8);
        let ctx = cf_test_context(&modes, &modulators, noise_frequency, aniso_y);

        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let count = 40usize;
        let total = (count * count * count) as f64;
        for ix in 0..count {
            for iy in 0..count {
                for iz in 0..count {
                    let pb = [
                        ix as f32 * 0.083 + 0.031,
                        iy as f32 * 0.079 + 0.017,
                        iz as f32 * 0.087 + 0.053,
                    ];
                    let sample = cf_sample_at(
                        &ctx,
                        &modulators,
                        pb,
                        [0.0; 3],
                        amp_disp,
                        warp_frequency,
                        noise_frequency,
                        aniso_y,
                    );
                    let w = [
                        pb[0] * noise_frequency,
                        pb[1] * aniso_y * noise_frequency,
                        pb[2] * noise_frequency,
                    ];
                    let value = evaluate_wave_noise_cf(&modes, w, 0.0, Some(&sample)) as f64;
                    sum += value;
                    sum_sq += value * value;
                }
            }
        }
        let mean = sum / total;
        let std = (sum_sq / total - mean * mean).sqrt();
        assert!((mean - WAVE_NOISE_MEAN as f64).abs() < 0.015, "mean {mean}");
        assert!(
            (std - WAVE_NOISE_STD as f64).abs() < 0.02,
            "std {std} vs {WAVE_NOISE_STD}"
        );
    }

    /// The RMS cap keeps every psi inside the Chebyshev fit domain.
    #[test]
    fn test_cf_psi_within_bound() {
        let modes = generate_wave_modes();
        let modulators = generate_wave_warp_modes();
        let (warp_frequency, noise_frequency, aniso_y) = (5.0f32, 6.0f32, 0.35f32);
        let ctx = cf_test_context(&modes, &modulators, noise_frequency, aniso_y);
        for amp_disp in [0.3f32, 1.4, 3.0] {
            for point in flow_warp_sample_points() {
                let pb = [point[0] as f32, point[1] as f32, point[2] as f32];
                let sample = cf_sample_at(
                    &ctx,
                    &modulators,
                    pb,
                    [0.577; 3],
                    amp_disp,
                    warp_frequency,
                    noise_frequency,
                    aniso_y,
                );
                for (n, mode) in modes.iter().enumerate() {
                    let (psi, _) = sample.mode_terms(n, mode.k);
                    assert!(
                        psi.abs() <= WAVE_CF_PSI_BOUND * 1.0001,
                        "psi {psi} out of bound at mode {n}, amp {amp_disp}"
                    );
                }
            }
        }
    }

    /// Slow ray: the low-pass version with cf reduces to the pointwise cf field.

    #[test]
    fn test_flow_warp_determinism() {
        let modes = generate_wave_warp_modes();
        let point = [0.5f32, 0.3, 0.7];
        let warp_frequency = 1.0;
        let axial_scale = 0.35;
        let strength = 0.35;
        let result1 = evaluate_wave_flow_warp(&modes, point, warp_frequency, axial_scale, strength);
        let result2 = evaluate_wave_flow_warp(&modes, point, warp_frequency, axial_scale, strength);
        assert_eq!(result1, result2, "flow warp is not deterministic");
    }

    fn flow_warp_sample_points() -> Vec<[f64; 3]> {
        let mut state: u64 = 12345;
        let mut next_coordinate = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
        };
        (0..200)
            .map(|_| [next_coordinate(), next_coordinate(), next_coordinate()])
            .collect()
    }

    fn matrix_determinant(m: &[[f64; 3]; 3]) -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    fn matrix_frobenius_norm(m: &[[f64; 3]; 3]) -> f64 {
        m.iter()
            .flatten()
            .map(|entry| entry * entry)
            .sum::<f64>()
            .sqrt()
    }

    /// The curl basis makes each mode divergence-free, so the shear warp preserves
    /// volume exactly (det = 1); the Frobenius bound rejects a near-zero deformation.
    #[test]
    fn test_flow_warp_volume_preservation() {
        let modes = generate_wave_warp_modes();
        let points = flow_warp_sample_points();

        let warp_frequency: f64 = 5.0;
        let axial_scale: f64 = 0.35;
        let strength: f64 = 1.35;

        // m_diag = [wf, axial*wf, wf] for the linear map M
        let m_diag = [warp_frequency, axial_scale * warp_frequency, warp_frequency];

        let mut frobenius_sum = 0.0f64;
        for point in &points {
            // Initialize z = M * p and J as 3x3 identity
            let mut z = wave_linear_map(*point, warp_frequency, axial_scale);
            let mut j: [[f64; 3]; 3] = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ];

            // Apply each shear S_i and accumulate Jacobian: J = J_i * J
            for mode in &modes {
                let angle = mode.k[0] as f64 * z[0]
                    + mode.k[1] as f64 * z[1]
                    + mode.k[2] as f64 * z[2]
                    + mode.phase as f64;
                let fi_prime = -strength * mode.amplitude as f64 * angle.sin();
                let c0 = mode.curl_direction[0] as f64;
                let c1 = mode.curl_direction[1] as f64;
                let c2 = mode.curl_direction[2] as f64;
                let k0 = mode.k[0] as f64;
                let k1 = mode.k[1] as f64;
                let k2 = mode.k[2] as f64;

                // J_i = I + c * (f_i' * k)^T, so J_i[j][l] = delta_jl + c[j] * fi_prime * k[l]
                // New J = J_i * J: new_j[j][l] = sum_m (delta_jm + c[j]*fi_prime*k[m]) * J[m][l]
                let mut new_j: [[f64; 3]; 3] = [[0.0; 3]; 3];
                for row in 0..3 {
                    let cj = [c0, c1, c2][row];
                    for col in 0..3 {
                        // delta_jm * J[m][l] = J[row][col] (when m == row)
                        let mut sum_m = j[row][col];
                        for m in 0..3 {
                            let km = [k0, k1, k2][m];
                            sum_m += cj * fi_prime * km * j[m][col];
                        }
                        new_j[row][col] = sum_m;
                    }
                }
                j = new_j;

                // Update z with shear: z += strength * a_i * cos(angle) * c_i
                let value = strength * mode.amplitude as f64 * angle.cos();
                z[0] += c0 * value;
                z[1] += c1 * value;
                z[2] += c2 * value;
            }

            // Transform J back to local frame: J_local = M^-1 * J * M
            // Since M is diagonal, J_local[i][l] = J[i][l] * m_diag[l] / m_diag[i]
            let mut j_local: [[f64; 3]; 3] = [[0.0; 3]; 3];
            for i in 0..3 {
                for l in 0..3 {
                    j_local[i][l] = j[i][l] * m_diag[l] / m_diag[i];
                }
            }

            let determinant = matrix_determinant(&j_local);
            assert!(
                (determinant - 1.0).abs() < 1e-6,
                "volume not preserved at {point:?}: det = {determinant:.15}"
            );
            assert!(
                determinant > 0.0,
                "folding detected at {point:?}: det = {determinant:.15}"
            );
            frobenius_sum += matrix_frobenius_norm(&j_local);
        }

        let mean_frobenius = frobenius_sum / points.len() as f64;
        assert!(
            mean_frobenius >= 8.0,
            "deformation too small: mean Frobenius norm {mean_frobenius:.4}"
        );

        // Divergence-free check: |k_i · curl_direction_i| < 1e-5 for all modes
        for mode in &modes {
            let dot = mode.k[0] as f64 * mode.curl_direction[0] as f64
                + mode.k[1] as f64 * mode.curl_direction[1] as f64
                + mode.k[2] as f64 * mode.curl_direction[2] as f64;
            assert!(
                dot.abs() < 1e-5,
                "mode not divergence-free: k·curl_dir = {:.8}",
                dot
            );
        }
    }

    #[test]
    fn test_flow_warp_no_folding() {
        let modes = generate_wave_warp_modes();
        let points = flow_warp_sample_points();

        let warp_frequency: f64 = 5.0;
        let axial_scale: f64 = 0.35;
        let strength: f64 = 1.35;

        // m_diag = [wf, axial*wf, wf] for the linear map M
        let m_diag = [warp_frequency, axial_scale * warp_frequency, warp_frequency];

        for point in &points {
            // Initialize z = M * p and J as 3x3 identity
            let mut z = wave_linear_map(*point, warp_frequency, axial_scale);
            let mut j: [[f64; 3]; 3] = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ];

            // Apply each shear S_i and accumulate Jacobian: J = J_i * J
            for mode in &modes {
                let angle = mode.k[0] as f64 * z[0]
                    + mode.k[1] as f64 * z[1]
                    + mode.k[2] as f64 * z[2]
                    + mode.phase as f64;
                let fi_prime = -strength * mode.amplitude as f64 * angle.sin();
                let c0 = mode.curl_direction[0] as f64;
                let c1 = mode.curl_direction[1] as f64;
                let c2 = mode.curl_direction[2] as f64;
                let k0 = mode.k[0] as f64;
                let k1 = mode.k[1] as f64;
                let k2 = mode.k[2] as f64;

                // J_i = I + c * (f_i' * k)^T, so J_i[j][l] = delta_jl + c[j] * fi_prime * k[l]
                // New J = J_i * J: new_j[j][l] = sum_m (delta_jm + c[j]*fi_prime*k[m]) * J[m][l]
                let mut new_j: [[f64; 3]; 3] = [[0.0; 3]; 3];
                for row in 0..3 {
                    let cj = [c0, c1, c2][row];
                    for col in 0..3 {
                        let mut sum_m = j[row][col];
                        for m in 0..3 {
                            let km = [k0, k1, k2][m];
                            sum_m += cj * fi_prime * km * j[m][col];
                        }
                        new_j[row][col] = sum_m;
                    }
                }
                j = new_j;

                // Update z with shear: z += strength * a_i * cos(angle) * c_i
                let value = strength * mode.amplitude as f64 * angle.cos();
                z[0] += c0 * value;
                z[1] += c1 * value;
                z[2] += c2 * value;
            }

            // Transform J back to local frame: J_local = M^-1 * J * M
            let mut j_local: [[f64; 3]; 3] = [[0.0; 3]; 3];
            for i in 0..3 {
                for l in 0..3 {
                    j_local[i][l] = j[i][l] * m_diag[l] / m_diag[i];
                }
            }

            let determinant = matrix_determinant(&j_local);
            assert!(
                determinant > 0.0,
                "folding detected at {point:?}: det = {determinant:.15}"
            );
        }
    }



}
