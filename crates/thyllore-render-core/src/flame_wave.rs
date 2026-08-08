//! Wave-basis erosion noise: an analytic sum of random wave modes
//!   e(w) = sum_n a_n sin(k_n . w + phi_n + eddy_n t)
//! replacing the fbm lattice as the erosion source when `turbulence_model == 2`.
//! The field is defined by position alone — no lattice, no cells, no localized
//! elements — so the ray restriction is an exact 1D quasi-periodic function and
//! the closed-form integral needs no radial bands (band-boundary-coherence.md).
//!
//! Modes are deterministic (spherical Fibonacci directions, low-discrepancy
//! log-uniform magnitudes and phases) and parameter-free in normalized units:
//! frequency, anisotropy, advection and the erosion amplitude mapping stay the
//! runtime levers they are for the fbm basis, applied to the same warped
//! coordinate `anisoCompress(p) * noiseFrequency - advect`. Time evolution is
//! sweeping (advection translates the coordinate, so omega = k . U falls out)
//! plus a per-mode eddy-turnover rate ~ |k|^(2/3) scaled by noise_scroll_speed.
//! GLSL mirror: shaders/include/flame_wave.glsl (flameWaveNoiseSum /
//! flameWaveOccupancySegments).

use thyllore_math_core::{integrate_erf_response_linear, ErfResponseModel};

use crate::flame_radial::{envelope_fade, eroded_argument};

use std::sync::OnceLock;

/// UBO slot capacity (2 vec4 per mode): erosion + warp/modulator + detail
/// tables plus the closed-form shear layers appended at WAVE_CF_SHEAR_SLOT.
pub const WAVE_MODE_SLOTS: usize = 178;
/// Slot index of the closed-form transport shear layers (after the detail table).
pub const WAVE_CF_SHEAR_SLOT: usize = 176;
/// Active mode count (N=96: N=48 still showed fine interference fringes on
/// the real flame under the warp; N=16 reads as discrete elements).
pub const WAVE_MODE_COUNT: usize = 96;
/// Detail mode count for lattice-free contour wiggle and boundary displacement.
pub const WAVE_DETAIL_MODE_COUNT: usize = 64;
/// Uniform segments of the band-free closed form over the support crossing.
pub const FLAME_WAVE_SEGMENTS: usize = 64;

/// fbm3 statistics the wave field reproduces so the erosion mapping
/// `amp * mix(0.2, 1, h) * (noise - 0.35)` keeps its calibration:
/// mean = 0.5 * (0.5 + 0.25 + 0.125), std measured over the lattice.
pub const WAVE_NOISE_MEAN: f32 = 0.4375;
pub const WAVE_NOISE_STD: f32 = 0.106;
/// Wavenumber span matching the fbm 3-octave spectrum (lacunarity ~2, 3 octaves).
pub const WAVE_K_RATIO: f32 = 8.0;
/// Tanh shaping scale factor: s = WAVE_TANH_SCALE * WAVE_NOISE_STD.
pub const WAVE_TANH_SCALE: f32 = 0.6;

const GOLDEN_RATIO: f64 = 1.618033988749895;
/// Plastic-constant streams decorrelate magnitude and phase from the
/// direction index (a monotone magnitude over the Fibonacci spiral would
/// cluster low wavenumbers around the poles).
const PLASTIC_INV: f64 = 0.7548776662466927;
const PLASTIC_INV_SQ: f64 = 0.5698402909980532;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaveMode {
    /// Wave vector in warped-coordinate units (|k| spans 2*pi*[1, WAVE_K_RATIO]).
    pub k: [f32; 3],
    pub amplitude: f32,
    pub phase: f32,
    /// Eddy-turnover angular rate |k_hat|^(2/3) with alternating sign; the
    /// shader multiplies by noise_scroll_speed * time.
    pub eddy_rate: f32,
    /// Low-octave envelope coefficient: 0 for bottom-octave (and detail/warp)
    /// modes, `mu / sigma_low` for higher octaves. A high mode's contribution
    /// is multiplied by `1 + env_coeff * z_low`, tying fine detail to the
    /// coarse structure (cross-scale coupling the independent phases lack).
    pub env_coeff: f32,
}

/// Deterministic mode table in normalized units. Every call returns the same
/// table; runtime levers (frequency, anisotropy, advection, amplitude) apply
/// through the shared warped coordinate and the erosion mapping instead.
pub fn generate_wave_modes() -> [WaveMode; WAVE_MODE_COUNT] {
    generate_wave_modes_with_ratio(WAVE_K_RATIO)
}

/// Envelope modulation strength: high-octave amplitudes ride `1 + mu * z_low/sigma_low`
/// (multiplicative cascade driven by the field's own bottom octave). Chosen from
/// the 2D study (geometric_replacement_plan.md「fmT を fbm の層構造に近づける」):
/// mu 0.6 lifts coh15 0.574 -> 0.634 and skew +0.07 -> +0.18 toward fbm.
pub const WAVE_ENV_MU: f32 = 0.6;

/// Applies the low-octave envelope to an erosion mode table: marks modes above
/// the bottom octave with `env_coeff = mu / sigma_low` and renormalizes all
/// amplitudes so the total variance stays WAVE_NOISE_STD^2
/// (Var[z_low + (1+mu*z_low/sigma_low)*z_high] = sigma_low^2 + (1+mu^2)*sigma_high^2
/// for independent phases). `mu <= 0` leaves the table untouched.
pub fn apply_wave_envelope(modes: &mut [WaveMode], mu: f32) {
    if mu <= 0.0 {
        return;
    }
    let split = 2.0 * (2.0 * std::f64::consts::PI) as f32;
    let mut power_low = 0.0f64;
    let mut power_high = 0.0f64;
    for mode in modes.iter() {
        let k_mag = (mode.k[0] * mode.k[0] + mode.k[1] * mode.k[1] + mode.k[2] * mode.k[2]).sqrt();
        let power = 0.5 * (mode.amplitude as f64) * (mode.amplitude as f64);
        if k_mag < split {
            power_low += power;
        } else {
            power_high += power;
        }
    }
    if power_low <= 0.0 || power_high <= 0.0 {
        return;
    }
    let scale = (WAVE_NOISE_STD as f64)
        / (power_low + (1.0 + (mu as f64) * (mu as f64)) * power_high).sqrt();
    let sigma_low_scaled = power_low.sqrt() * scale;
    let coeff = (mu as f64 / sigma_low_scaled) as f32;
    for mode in modes.iter_mut() {
        mode.amplitude *= scale as f32;
        let k_mag = (mode.k[0] * mode.k[0] + mode.k[1] * mode.k[1] + mode.k[2] * mode.k[2]).sqrt();
        mode.env_coeff = if k_mag < split { 0.0 } else { coeff };
    }
}

/// Like [`generate_wave_modes`] but with an explicit `k_ratio` (the upper bound of
/// magnitude on the log scale, replacing the hardcoded `WAVE_K_RATIO`).
pub fn generate_wave_modes_with_ratio(k_ratio: f32) -> [WaveMode; WAVE_MODE_COUNT] {
    let mut modes = [WaveMode::default(); WAVE_MODE_COUNT];
    fill_wave_modes(&mut modes, k_ratio);
    modes
}

fn fill_wave_modes(modes: &mut [WaveMode], k_ratio: f32) {
    let count = modes.len();
    let mut power_sum = 0.0f64;
    for (n, mode) in modes.iter_mut().enumerate() {
        let vertical = 1.0 - 2.0 * (n as f64 + 0.5) / count as f64;
        let ring = (1.0 - vertical * vertical).max(0.0).sqrt();
        let azimuth = std::f64::consts::TAU * ((n as f64 / GOLDEN_RATIO).fract());
        let magnitude_u = (0.5 / count as f64 + n as f64 * PLASTIC_INV).fract();
        let magnitude =
            std::f64::consts::TAU * ((k_ratio as f64).ln() * magnitude_u).exp();
        mode.k = [
            (ring * azimuth.cos() * magnitude) as f32,
            (vertical * magnitude) as f32,
            (ring * azimuth.sin() * magnitude) as f32,
        ];
        // Kolmogorov amplitude a_n ~ k^(-5/6); normalized below.
        let amplitude = (magnitude / std::f64::consts::TAU).powf(-5.0 / 6.0);
        mode.amplitude = amplitude as f32;
        power_sum += 0.5 * amplitude * amplitude;

        let phase_u = (n as f64 * PLASTIC_INV_SQ + 0.5 * PLASTIC_INV).fract();
        mode.phase = (std::f64::consts::TAU * phase_u) as f32;
        let sign = if (n as f64 * PLASTIC_INV + 0.25).fract() < 0.5 {
            1.0
        } else {
            -1.0
        };
        mode.eddy_rate = (sign * (magnitude / std::f64::consts::TAU).powf(2.0 / 3.0)) as f32;
    }
    let scale = WAVE_NOISE_STD / (power_sum.sqrt() as f32);
    for mode in &mut *modes {
        mode.amplitude *= scale;
    }
}

/// Lattice-free replacement for the fbm behind the contour wiggle and the
/// boundary displacement. Same spectrum as the erosion table (so the character
/// matches the fbm it replaces) but only 16 modes, and `eddy_rate = 0` because
/// those two fields carry time in their coordinate like the fbm did.
pub fn generate_wave_detail_modes() -> [WaveMode; WAVE_DETAIL_MODE_COUNT] {
    let mut modes = [WaveMode::default(); WAVE_DETAIL_MODE_COUNT];
    fill_wave_modes(&mut modes, WAVE_K_RATIO);
    for mode in &mut modes {
        mode.eddy_rate = 0.0;
    }
    modes
}

/// Shaping parameters for tanh wave shaping: `(inverse_scale, amplitude)`.
/// `inverse_scale` = 1.0 / (WAVE_TANH_SCALE * WAVE_NOISE_STD).
/// `amplitude` = WAVE_NOISE_STD / sqrt(E[tanh(Z/s)^2]) where Z ~ N(0, WAVE_NOISE_STD^2),
/// computed by deterministic numerical integration (trapezoidal rule, 512 subdivisions over ±5σ).
pub fn wave_shaping_params(tanh_scale: f32) -> (f32, f32) {
    let s = tanh_scale * WAVE_NOISE_STD;
    let inverse_scale = 1.0 / s;
    // Compute E[tanh(Z/s)^2] where Z ~ N(0, WAVE_NOISE_STD^2).
    // Change variable: z = sigma * x where x ~ N(0,1), so tanh(z/s) = tanh(sigma/s * x).
    let sigma = WAVE_NOISE_STD as f64;
    let ratio = sigma / (s as f64); // sigma/s
    let n = 512;
    let range = 5.0; // ±5 sigma of standard normal
    let dx = 2.0 * range / n as f64;
    let mut integral = 0.0f64;
    for i in 0..n {
        let x_mid = -range + (i as f64 + 0.5) * dx;
        let tanh_val = (ratio * x_mid).tanh();
        let integrand = tanh_val * tanh_val * (-0.5 * x_mid * x_mid).exp();
        integral += integrand * dx;
    }
    // Normalize by sqrt(2*pi) for the Gaussian density
    let expected_value = integral / (2.0 * std::f64::consts::PI).sqrt();
    let amplitude = WAVE_NOISE_STD / (expected_value.sqrt() as f32);
    (inverse_scale, amplitude)
}

/// Cached shaping parameters for the default `WAVE_TANH_SCALE`.
/// Computed once via `wave_shaping_params` and stored in a `OnceLock`.
fn get_wave_shaping_params_cached() -> (f32, f32) {
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

/// Analytic replacement of the fbm domain warp: a low-wavenumber vector
/// displacement field of wave modes, calibrated to the fbm warp statistics
/// (per-component std 2 * fbm std = 0.212 before the warp_amp scaling).
/// The billowing shear of the flame look is this nonlinear composition — a
/// purely spectral anisotropy of the erosion modes cannot reproduce it.
pub const WAVE_WARP_MODE_COUNT: usize = 16;
pub const WAVE_WARP_COMPONENT_STD: f32 = 0.212;
pub const WAVE_WARP_K_RATIO: f32 = 2.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaveWarpMode {
    pub k: [f32; 3],
    pub amplitude: f32,
    pub phase: f32,
    /// Displacement direction (unit), orthogonal to `k` so the mode is
    /// divergence-free; the y component is scaled by warp_y_scale at
    /// evaluation like the fbm warp.
    pub curl_direction: [f32; 3],
}

fn fibonacci_direction(u: f64, azimuth_u: f64) -> [f64; 3] {
    let vertical = 1.0 - 2.0 * u;
    let ring = (1.0 - vertical * vertical).max(0.0).sqrt();
    let azimuth = std::f64::consts::TAU * azimuth_u;
    [ring * azimuth.cos(), vertical, ring * azimuth.sin()]
}

pub fn generate_wave_warp_modes() -> [WaveWarpMode; WAVE_WARP_MODE_COUNT] {
    let count = WAVE_WARP_MODE_COUNT;
    let mut modes = [WaveWarpMode::default(); WAVE_WARP_MODE_COUNT];
    let mut power_sum = 0.0f64;
    for (n, mode) in modes.iter_mut().enumerate() {
        let wave_direction = fibonacci_direction(
            (n as f64 + 0.5) / count as f64,
            (n as f64 / GOLDEN_RATIO).fract(),
        );
        let magnitude_u = (0.25 / count as f64 + n as f64 * PLASTIC_INV_SQ).fract();
        let magnitude =
            std::f64::consts::TAU * ((WAVE_WARP_K_RATIO as f64).ln() * magnitude_u).exp();
        mode.k = [
            (wave_direction[0] * magnitude) as f32,
            (wave_direction[1] * magnitude) as f32,
            (wave_direction[2] * magnitude) as f32,
        ];
        let amplitude = (magnitude / std::f64::consts::TAU).powf(-5.0 / 6.0);
        mode.amplitude = amplitude as f32;
        power_sum += 0.5 * amplitude * amplitude;
        mode.phase =
            (std::f64::consts::TAU * (n as f64 * PLASTIC_INV + 0.75 * PLASTIC_INV_SQ).fract())
                as f32;
        let displacement = fibonacci_direction(
            ((n as f64 + 0.5) * PLASTIC_INV).fract(),
            ((n as f64 + 2.0) * PLASTIC_INV_SQ + 0.5).fract(),
        );
        // k × displacement is orthogonal to k, so every mode is divergence-free.
        let curl = [
            wave_direction[1] * displacement[2] - wave_direction[2] * displacement[1],
            wave_direction[2] * displacement[0] - wave_direction[0] * displacement[2],
            wave_direction[0] * displacement[1] - wave_direction[1] * displacement[0],
        ];
        let curl_norm = (curl[0] * curl[0] + curl[1] * curl[1] + curl[2] * curl[2]).sqrt();
        mode.curl_direction = [
            (curl[0] / curl_norm) as f32,
            (curl[1] / curl_norm) as f32,
            (curl[2] / curl_norm) as f32,
        ];
    }
    // Random directions spread each mode's power over the three components
    // (1/3 each): normalize so every displacement component has the fbm warp std.
    let scale = WAVE_WARP_COMPONENT_STD * (3.0f64.sqrt() as f32) / (power_sum.sqrt() as f32);
    for mode in &mut modes {
        mode.amplitude *= scale;
    }
    modes
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

/// Scales `warp_amp * mix(0.15, 1, h)` into the shear strength: at the default
/// warp_amp of 1.4 the top of the flame matches the displacement warp's
/// deformation.
pub const WAVE_SHEAR_STRENGTH_SCALE: f32 = 0.96;

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
        strength as f64 * WAVE_SHEAR_STRENGTH_SCALE as f64,
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

    let strength_scaled = strength as f64 * WAVE_SHEAR_STRENGTH_SCALE as f64;

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

/// Closed-form (family-T) variant: the 16-shear transport is replaced by at
/// most two single-frequency shear layers, and the erosion carrier phases are
/// pseudo-FM modulated by the 16 warp modes evaluated at the UNWARPED
/// coordinate — sin(A + psi) expanded as sinA*T_c(psi) + cosA*T_s(psi) with
/// T_c/T_s Chebyshev polynomials, so every factor stays a finite sum/product
/// of sinusoids (geometric_replacement_plan.md「厳密閉形式化」).
/// Per-mode modulation depth is RMS-capped at WAVE_CF_CAP; the pointwise bound
/// |psi| <= sqrt(2 * modulators * CAP^2) fixes the Chebyshev fit domain.
pub const WAVE_CF_CAP: f32 = 2.0;
/// sqrt(16 * 8): Sum |gamma| <= sqrt(16 * Sum gamma^2) with Sum gamma^2 <= 2*CAP^2.
pub const WAVE_CF_PSI_BOUND: f32 = 11.313_708;
pub const WAVE_CF_CHEB_COEFFS: usize = 21;
/// Default shear gain of the transport layers (0.15-0.3 balances organic base
/// and streaks; > 0.5 turns directionally monotone — 2D study).
pub const WAVE_CF_SHEAR_GAIN: f32 = 0.3;
pub const WAVE_CF_SHEAR_LAYERS: usize = 2;

/// Chebyshev interpolation of `f` at WAVE_CF_CHEB_COEFFS points over
/// [-WAVE_CF_PSI_BOUND, WAVE_CF_PSI_BOUND], returned as series coefficients in
/// x = psi / bound (deterministic closed-form DCT, no fitting libraries).
fn chebyshev_fit(f: impl Fn(f64) -> f64) -> [f32; WAVE_CF_CHEB_COEFFS] {
    let n = WAVE_CF_CHEB_COEFFS;
    let bound = WAVE_CF_PSI_BOUND as f64;
    let samples: Vec<f64> = (0..n)
        .map(|i| {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / n as f64;
            f(bound * theta.cos())
        })
        .collect();
    let mut coeffs = [0.0f32; WAVE_CF_CHEB_COEFFS];
    for (k, slot) in coeffs.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for (i, sample) in samples.iter().enumerate() {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / n as f64;
            sum += sample * (k as f64 * theta).cos();
        }
        *slot = (sum * if k == 0 { 1.0 } else { 2.0 } / n as f64) as f32;
    }
    coeffs
}

/// `(T_s, T_c)`: Chebyshev tables of sin and cos over the psi bound.
pub fn wave_cf_chebyshev_tables() -> (
    [f32; WAVE_CF_CHEB_COEFFS],
    [f32; WAVE_CF_CHEB_COEFFS],
) {
    static CACHED: OnceLock<([f32; WAVE_CF_CHEB_COEFFS], [f32; WAVE_CF_CHEB_COEFFS])> =
        OnceLock::new();
    *CACHED.get_or_init(|| (chebyshev_fit(f64::sin), chebyshev_fit(f64::cos)))
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

/// Transport layers of the closed-form variant: the top-`layers` warp modes by
/// amplitude, each promoted to a single-frequency shear whose amplitude
/// `half / a_i * gain` splits the full warp power across two layers
/// (half = sqrt(sum a^2 / 2)); `strength * amplitude` at evaluation then
/// matches the 2D study's `s * (half / WA_i * gain)` coefficient exactly.
pub fn generate_wave_cf_shear_layers(
    warp_modes: &[WaveWarpMode],
    layers: usize,
    gain: f32,
) -> Vec<WaveWarpMode> {
    let half = (warp_modes.iter().map(|m| (m.amplitude as f64).powi(2)).sum::<f64>() / 2.0)
        .sqrt() as f32;
    let mut order: Vec<usize> = (0..warp_modes.len()).collect();
    order.sort_by(|&a, &b| {
        warp_modes[b]
            .amplitude
            .partial_cmp(&warp_modes[a].amplitude)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
        .into_iter()
        .take(layers)
        .map(|i| {
            let mut layer = warp_modes[i];
            layer.amplitude = half / warp_modes[i].amplitude * gain;
            layer
        })
        .collect()
}

/// Per-mode modulation depth per unit warp displacement amplitude:
/// D_n = noise_frequency * sqrt(0.5 * sum_j dot(k_n, aniso(c_j * a_j))^2).
/// The runtime depth is `amp_disp(h) * D_n` and the coupling cap scale
/// `min(1, CAP / depth)`; `noise_aniso` is the erosion-side anisotropy
/// (a displacement delta in pb shifts the carrier by k . aniso(delta) * freq).
pub fn wave_cf_depth_scale(
    erosion_modes: &[WaveMode],
    modulators: &[WaveWarpMode],
    noise_frequency: f32,
    noise_aniso: impl Fn([f32; 3]) -> [f32; 3],
) -> [f32; WAVE_MODE_COUNT] {
    let displaced: Vec<[f32; 3]> = modulators
        .iter()
        .map(|m| {
            noise_aniso([
                m.curl_direction[0] * m.amplitude,
                m.curl_direction[1] * m.amplitude,
                m.curl_direction[2] * m.amplitude,
            ])
        })
        .collect();
    let mut scale = [0.0f32; WAVE_MODE_COUNT];
    for (n, mode) in erosion_modes.iter().enumerate().take(WAVE_MODE_COUNT) {
        let mut power = 0.0f64;
        for c in &displaced {
            let g = (mode.k[0] * c[0] + mode.k[1] * c[1] + mode.k[2] * c[2]) as f64;
            power += 0.5 * g * g;
        }
        scale[n] = noise_frequency * power.sqrt() as f32;
    }
    scale
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
    let carrier = |index: usize, mode: &WaveMode| {
        let angle = mode.k[0] * w[0]
            + mode.k[1] * w[1]
            + mode.k[2] * w[2]
            + mode.phase
            + mode.eddy_rate * eddy_time;
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

/// Per-ray split of the mode set into a resolved part (evaluated at the
/// segment nodes, attenuated by the node-spacing low-pass) and an unresolved
/// remainder routed into the smoothed-response sigma. `rates` is a slice of
/// warped-coordinate rates `dw/dt` at multiple points along the ray; `node_spacing`
/// the node distance in t. For each mode, the weight is the minimum over all
/// rates (most attenuating = conservative). Both the weights and the sigma are
/// smooth in the ray — no per-ray integer mode partition (the appendix-7
/// quantization trap).
pub fn wave_ray_attenuation(
    modes: &[WaveMode],
    rates: &[[f32; 3]],
    node_spacing: f32,
) -> (Vec<f32>, f32) {
    let mut weights = Vec::with_capacity(modes.len());
    let mut unresolved_power = 0.0f32;
    for mode in modes {
        // Use the minimum weight across all rates (most attenuating).
        let mut min_weight = 1.0f32;
        for rate in rates {
            let beta = mode.k[0] * rate[0] + mode.k[1] * rate[1] + mode.k[2] * rate[2];
            let x = beta * node_spacing / std::f32::consts::PI;
            let weight = (-(x * x) * (x * x)).exp();
            if weight < min_weight {
                min_weight = weight;
            }
        }
        weights.push(min_weight);
        unresolved_power += 0.5 * mode.amplitude * mode.amplitude * (1.0 - min_weight * min_weight);
    }
    (weights, unresolved_power.sqrt())
}

/// Attenuated wave noise at a warped coordinate: the resolved part of the field
/// the segment nodes linearize (mirror of the node loop in
/// `flameWaveOccupancySegments`).
pub fn evaluate_wave_noise_attenuated(
    modes: &[WaveMode],
    weights: &[f32],
    w: [f32; 3],
    eddy_time: f32,
) -> f32 {
    let mut z = 0.0f32;
    for (mode, weight) in modes.iter().zip(weights) {
        let angle = mode.k[0] * w[0]
            + mode.k[1] * w[1]
            + mode.k[2] * w[2]
            + mode.phase
            + mode.eddy_rate * eddy_time;
        z += weight * mode.amplitude * angle.sin();
    }
    let (inverse_scale, amplitude) = get_wave_shaping_params_cached();
    WAVE_NOISE_MEAN + amplitude * (z * inverse_scale).tanh()
}

/// Local low-pass filtered wave noise at a warped coordinate: resolves modes by
/// the rate at this specific node position and returns `(noise, sigma_noise)`
/// where sigma is the square root of unresolved mode power. This replaces the
/// per-ray attenuation with per-node attenuation — each node's weights come from
/// its own rate, so stretching varies along the ray.
pub fn evaluate_wave_noise_local_lowpass(
    modes: &[WaveMode],
    w: [f32; 3],
    rate: [f32; 3],
    node_spacing: f32,
    eddy_time: f32,
) -> (f32, f32) {
    evaluate_wave_noise_local_lowpass_cf(modes, w, rate, node_spacing, eddy_time, None)
}

/// Like [`evaluate_wave_noise_local_lowpass`] but with the optional closed-form
/// pseudo-FM modulation: each carrier becomes sinA*T_c(psi) + cosA*T_s(psi) and
/// its low-pass beta gains the FM phase rate d psi/dt along the ray.
pub fn evaluate_wave_noise_local_lowpass_cf(
    modes: &[WaveMode],
    w: [f32; 3],
    rate: [f32; 3],
    node_spacing: f32,
    eddy_time: f32,
    cf: Option<&WaveCfSample>,
) -> (f32, f32) {
    let carrier = |index: usize, mode: &WaveMode| {
        let angle = mode.k[0] * w[0]
            + mode.k[1] * w[1]
            + mode.k[2] * w[2]
            + mode.phase
            + mode.eddy_rate * eddy_time;
        let beta_carrier = mode.k[0] * rate[0] + mode.k[1] * rate[1] + mode.k[2] * rate[2];
        let (value, beta_fm) = match cf {
            Some(sample) => sample.wave_value(index, mode.k, angle),
            None => (angle.sin(), 0.0),
        };
        let x = (beta_carrier + beta_fm) * node_spacing / std::f32::consts::PI;
        let weight = (-(x * x) * (x * x)).exp();
        (value, weight)
    };
    let mut z_low = 0.0f32;
    let mut unresolved_power = 0.0f32;
    for (index, mode) in modes.iter().enumerate().filter(|(_, m)| m.env_coeff == 0.0) {
        let (value, weight) = carrier(index, mode);
        z_low += weight * mode.amplitude * value;
        unresolved_power += 0.5 * mode.amplitude * mode.amplitude * (1.0 - weight * weight);
    }
    let mut z = z_low;
    for (index, mode) in modes.iter().enumerate().filter(|(_, m)| m.env_coeff != 0.0) {
        let (value, weight) = carrier(index, mode);
        let envelope = 1.0 + mode.env_coeff * z_low;
        z += envelope * weight * mode.amplitude * value;
        unresolved_power +=
            envelope * envelope * 0.5 * mode.amplitude * mode.amplitude * (1.0 - weight * weight);
    }
    let (inverse_scale, amplitude) = get_wave_shaping_params_cached();
    let noise = WAVE_NOISE_MEAN + amplitude * (z * inverse_scale).tanh();
    let sigma_noise = unresolved_power.sqrt();
    (noise, sigma_noise)
}

/// One emission segment of the band-free occupancy integral: emission integral
/// and density-weighted mean t, ready for the per-segment Beer-Lambert
/// composite (no radial bands anywhere in the pipeline).
#[derive(Clone, Copy, Debug, Default)]
pub struct WaveSegmentEmission {
    pub emission: f32,
    pub t_mean: f32,
}

/// Band-free occupancy of the eroded threshold field over `[t0, t1]`: one
/// support crossing, `FLAME_WAVE_SEGMENTS` uniform segments, density AND
/// erosion exact at every node (both are analytic — nothing sampled from a
/// realization), the argument linear between nodes, each segment integrated
/// with the closed-form erf response. `noise_at(t)` returns `(noise, sigma_noise)`
/// where sigma is the local unresolved mode power at that node;
/// `erosion_scale_at(t)` is the `amp * mix(0.2, 1, h)` mapping. Segments whose node
/// densities all vanish are skipped before any erosion work (exact membership
/// at node resolution). Mirror of `flameWaveOccupancySegments`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_wave_occupancy_segments(
    model: &ErfResponseModel,
    t0: f32,
    t1: f32,
    density_at: impl Fn(f32) -> f32,
    noise_at: impl Fn(f32) -> (f32, f32),
    erosion_scale_at: impl Fn(f32) -> f32,
    flood_fade_scale: f32,
    carve_residual: f32,
) -> [WaveSegmentEmission; FLAME_WAVE_SEGMENTS] {
    let mut segments = [WaveSegmentEmission::default(); FLAME_WAVE_SEGMENTS];
    let dt = (t1 - t0) / FLAME_WAVE_SEGMENTS as f32;
    if dt <= 0.0 {
        return segments;
    }

    let mut density = [0.0f32; FLAME_WAVE_SEGMENTS + 1];
    for (node, slot) in density.iter_mut().enumerate() {
        *slot = density_at(t0 + node as f32 * dt);
    }
    let mut argument = [0.0f32; FLAME_WAVE_SEGMENTS + 1];
    let mut noise_values = [0.0f32; FLAME_WAVE_SEGMENTS + 1];
    let mut sigma_values = [0.0f32; FLAME_WAVE_SEGMENTS + 1];
    for node in 0..=FLAME_WAVE_SEGMENTS {
        let adjacent_support = density[node] > 0.0
            || (node > 0 && density[node - 1] > 0.0)
            || (node < FLAME_WAVE_SEGMENTS && density[node + 1] > 0.0);
        argument[node] = if adjacent_support {
            let t = t0 + node as f32 * dt;
            let (noise_val, sigma_local) = noise_at(t);
            noise_values[node] = noise_val;
            sigma_values[node] = sigma_local;
            let erosion = erosion_scale_at(t) * (noise_val - 0.35);
            eroded_argument(density[node], erosion, flood_fade_scale)
        } else {
            0.0
        };
    }

    let (inverse_scale, amplitude) = get_wave_shaping_params_cached();

    for (segment, slot) in segments.iter_mut().enumerate() {
        let t_prev = t0 + segment as f32 * dt;
        slot.t_mean = t_prev + 0.5 * dt;
        if density[segment] <= 0.0 && density[segment + 1] <= 0.0 {
            continue;
        }
        let shaping_deriv_avg = 0.5
            * (wave_shaping_derivative(noise_values[segment], inverse_scale, amplitude)
                + wave_shaping_derivative(noise_values[segment + 1], inverse_scale, amplitude));
        let sigma_local = 0.5 * (sigma_values[segment] + sigma_values[segment + 1]);
        let sigma_eff = sigma_local
            * shaping_deriv_avg
            * erosion_scale_at(t_prev + 0.5 * dt).abs()
            * 0.5
            * (envelope_fade(density[segment], flood_fade_scale)
                + envelope_fade(density[segment + 1], flood_fade_scale));
        let slope = (argument[segment + 1] - argument[segment]) / dt;
        let (mut integral, mut first_moment) = integrate_erf_response_linear(
            model,
            sigma_eff,
            argument[segment] - slope * t_prev,
            slope,
            t_prev,
            t_prev + dt,
        );
        if carve_residual > 0.0 {
            let plain_slope = (density[segment + 1] - density[segment]) / dt;
            let (plain_integral, plain_moment) = integrate_erf_response_linear(
                model,
                0.0,
                density[segment] - plain_slope * t_prev,
                plain_slope,
                t_prev,
                t_prev + dt,
            );
            integral += carve_residual * (plain_integral - integral);
            first_moment += carve_residual * (plain_moment - first_moment);
        }
        slot.emission = integral.max(0.0);
        if integral > 1e-6 {
            slot.t_mean = (first_moment / integral).clamp(t_prev, t_prev + dt);
        }
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_math_core::{evaluate_erf_response, fit_erf_response};

    #[test]
    fn test_mode_table_is_deterministic_and_normalized() {
        let modes = generate_wave_modes();
        assert_eq!(modes, generate_wave_modes());

        let power: f32 = modes.iter().map(|m| 0.5 * m.amplitude * m.amplitude).sum();
        assert!((power.sqrt() - WAVE_NOISE_STD).abs() < 1e-4);

        for mode in &modes {
            let magnitude =
                (mode.k[0] * mode.k[0] + mode.k[1] * mode.k[1] + mode.k[2] * mode.k[2]).sqrt();
            let tau = std::f32::consts::TAU;
            assert!(magnitude > tau * 0.99 && magnitude < tau * WAVE_K_RATIO * 1.01);
            assert!(mode.amplitude > 0.0);
        }

        // Direction coverage: no two modes closer than a few degrees (a
        // degenerate spiral would leave lattice-like preferred directions).
        for i in 0..modes.len() {
            for j in (i + 1)..modes.len() {
                let a = modes[i].k;
                let b = modes[j].k;
                let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
                let cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
                assert!(cos < 0.999, "modes {i} and {j} nearly collinear");
            }
        }
    }

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
    fn test_attenuation_splits_power_smoothly() {
        let modes = generate_wave_modes();
        // Slow ray: everything resolved, sigma vanishes.
        let rates_slow: [[f32; 3]; 3] = [[0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [0.01, 0.0, 0.0]];
        let (weights, sigma) = wave_ray_attenuation(&modes, &rates_slow, 0.1);
        assert!(weights.iter().all(|w| *w > 0.99));
        assert!(sigma < 1e-3);
        // Fast ray: the top of the spectrum is unresolved, sigma bounded by the
        // total field std.
        let rates_fast: [[f32; 3]; 3] = [
            [1.0, 0.4, 0.2],
            [1.0, 0.4, 0.2],
            [1.0, 0.4, 0.2],
        ];
        let (weights_fast, sigma_fast) = wave_ray_attenuation(&modes, &rates_fast, 0.5);
        assert!(weights_fast.iter().any(|w| *w < 0.5));
        assert!(sigma_fast > 0.01 && sigma_fast < WAVE_NOISE_STD * 1.01);
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
    #[test]
    fn test_occupancy_matches_dense_quadrature_when_resolved() {
        let modes = generate_wave_modes();
        let warp_modes = generate_wave_warp_modes();
        let model = fit_erf_response(0.27, 0.33);
        let flood_fade_scale = 0.33;
        let carve_residual = 0.12;
        let amp = 1.5;
        let (t0, t1) = (0.0f32, 0.4f32);
        let warp_frequency = 1.0;
        let axial_scale = 0.35;
        let strength = 0.35;

        for (case, origin) in [[0.1f32, 0.2, 0.05], [0.35, 0.6, 0.1], [-0.2, 0.4, 0.3]]
            .into_iter()
            .enumerate()
        {
            let dir = [0.3f32, 0.5, 0.1];
            let warped_at = move |t: f32| {
                [
                    origin[0] + t * dir[0],
                    origin[1] + t * dir[1],
                    origin[2] + t * dir[2],
                ]
            };
            let height_at = move |t: f32| (origin[1] + t * dir[1]).clamp(0.0, 1.0);
            let density_at = move |t: f32| {
                let h = height_at(t);
                let envelope = (1.0 - (2.0 * h - 1.0) * (2.0 * h - 1.0)).max(0.0);
                envelope * 0.8
            };
            let erosion_scale_at = move |t: f32| amp * (0.2 + 0.8 * height_at(t));

            let dt = (t1 - t0) / FLAME_WAVE_SEGMENTS as f32;
            let segments = evaluate_wave_occupancy_segments(
                &model,
                t0,
                t1,
                density_at,
                |t| {
                    let (_, rate) = evaluate_wave_flow_warp_with_rate(
                        &warp_modes,
                        warped_at(t),
                        dir,
                        warp_frequency,
                        axial_scale,
                        strength,
                    );
                    evaluate_wave_noise_local_lowpass(&modes, warped_at(t), rate, dt, 0.0)
                },
                erosion_scale_at,
                flood_fade_scale,
                carve_residual,
            );
            let closed: f32 = segments.iter().map(|s| s.emission).sum();

            let steps = 4000;
            let quad_dt = (t1 - t0) / steps as f32;
            let mut reference = 0.0f32;
            for i in 0..steps {
                let t = t0 + (i as f32 + 0.5) * quad_dt;
                let density = density_at(t);
                let erosion =
                    erosion_scale_at(t) * (evaluate_wave_noise(&modes, warped_at(t), 0.0) - 0.35);
                let argument = eroded_argument(density, erosion, flood_fade_scale);
                let carved = evaluate_erf_response(&model, argument, 0.0);
                let plain = evaluate_erf_response(&model, density, 0.0);
                reference += (carved + carve_residual * (plain - carved)) * quad_dt;
            }

            let tolerance = 0.03 * (t1 - t0);
            assert!(
                (closed - reference).abs() < tolerance,
                "case {case}: closed {closed} vs reference {reference}"
            );
        }
    }

    /// No steps between adjacent parallel rays: the closed form must vary
    /// smoothly as the ray sweeps (the band quantization this basis removes
    /// would show up as jumps).
    #[test]
    fn test_occupancy_is_smooth_across_parallel_rays() {
        let modes = generate_wave_modes();
        let warp_modes = generate_wave_warp_modes();
        let model = fit_erf_response(0.27, 0.33);
        let dir = [0.1f32, 1.0, 0.05];
        let (t0, t1) = (0.0f32, 1.0f32);
        let dt = (t1 - t0) / FLAME_WAVE_SEGMENTS as f32;
        let warp_frequency = 1.0;
        let axial_scale = 0.35;
        let strength = 0.35;

        let mut previous: Option<f32> = None;
        for step in 0..200 {
            let x0 = -0.5 + step as f32 * 0.005;
            let warped_at = move |t: f32| [x0 + t * dir[0], t * dir[1], 0.2 + t * dir[2]];
            let height_at = move |t: f32| (t * dir[1]).clamp(0.0, 1.0);
            let density_at = move |t: f32| {
                let h = height_at(t);
                let envelope = (1.0 - (2.0 * h - 1.0) * (2.0 * h - 1.0)).max(0.0);
                let radial = (1.0 - (x0 + t * dir[0]) * (x0 + t * dir[0])).max(0.0);
                envelope * radial
            };
            let segments = evaluate_wave_occupancy_segments(
                &model,
                t0,
                t1,
                density_at,
                |t| {
                    let (_, rate) = evaluate_wave_flow_warp_with_rate(
                        &warp_modes,
                        warped_at(t),
                        dir,
                        warp_frequency,
                        axial_scale,
                        strength,
                    );
                    evaluate_wave_noise_local_lowpass(&modes, warped_at(t), rate, dt, 0.0)
                },
                |t| 1.5 * (0.2 + 0.8 * height_at(t)),
                0.33,
                0.12,
            );
            let total: f32 = segments.iter().map(|s| s.emission).sum();
            if let Some(prev) = previous {
                assert!(
                    (total - prev).abs() < 0.02,
                    "occupancy jump {} -> {} at step {}",
                    prev,
                    total,
                    step
                );
            }
            previous = Some(total);
        }
    }

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

    #[test]
    fn test_cf_shear_layers_split_warp_power() {
        let warp_modes = generate_wave_warp_modes();
        let layers = generate_wave_cf_shear_layers(&warp_modes, 2, WAVE_CF_SHEAR_GAIN);
        assert_eq!(layers.len(), 2);
        let half = (warp_modes.iter().map(|m| (m.amplitude as f64).powi(2)).sum::<f64>() / 2.0)
            .sqrt() as f32;
        let mut sorted: Vec<f32> = warp_modes.iter().map(|m| m.amplitude).collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for (layer, original_amp) in layers.iter().zip(&sorted) {
            assert!((layer.amplitude - half / original_amp * WAVE_CF_SHEAR_GAIN).abs() < 1e-6);
            // Single-frequency shear stays exactly volume preserving: c ⊥ k.
            let dot = layer.k[0] * layer.curl_direction[0]
                + layer.k[1] * layer.curl_direction[1]
                + layer.k[2] * layer.curl_direction[2];
            assert!(dot.abs() < 1e-5, "shear layer not divergence-free: {dot}");
        }
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
    fn test_cf_local_lowpass_matches_pointwise_when_resolved() {
        let modes = generate_wave_modes();
        let modulators = generate_wave_warp_modes();
        let (warp_frequency, noise_frequency, aniso_y) = (5.0f32, 6.0f32, 0.35f32);
        let ctx = cf_test_context(&modes, &modulators, noise_frequency, aniso_y);
        let pb = [0.31f32, 0.45, 0.12];
        let sample = cf_sample_at(
            &ctx,
            &modulators,
            pb,
            [0.001; 3],
            1.2,
            warp_frequency,
            noise_frequency,
            aniso_y,
        );
        let w = [
            pb[0] * noise_frequency,
            pb[1] * aniso_y * noise_frequency,
            pb[2] * noise_frequency,
        ];
        let pointwise = evaluate_wave_noise_cf(&modes, w, 0.3, Some(&sample));
        let (lowpassed, sigma) = evaluate_wave_noise_local_lowpass_cf(
            &modes,
            w,
            [0.001; 3],
            0.01,
            0.3,
            Some(&sample),
        );
        assert!((pointwise - lowpassed).abs() < 1e-4);
        assert!(sigma < 1e-3);
    }

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

    #[test]
    fn test_local_lowpass_varies_along_ray() {
        let modes = generate_wave_modes();
        let warp_modes = generate_wave_warp_modes();
        let dt = 1.0 / FLAME_WAVE_SEGMENTS as f32;

        // (a) Consistency: for a given node, evaluate_wave_noise_local_lowpass with a single rate
        // must match evaluate_wave_noise_attenuated using weights from wave_ray_attenuation with
        // that same rate (within 1e-5).
        let w = [0.3f32, 0.5, 0.7];
        let rate = [0.1f32, 0.8, 0.05];
        let (noise_local, sigma_local) = evaluate_wave_noise_local_lowpass(&modes, w, rate, dt, 0.0);
        let rates: [[f32; 3]; 3] = [rate, rate, rate];
        let (weights, sigma_ray) = wave_ray_attenuation(&modes, &rates, dt);
        let noise_attenuated = evaluate_wave_noise_attenuated(&modes, &weights, w, 0.0);
        assert!(
            (noise_local - noise_attenuated).abs() < 1e-5,
            "consistency: noise_local={:.8} vs noise_attenuated={:.8}",
            noise_local,
            noise_attenuated
        );
        assert!(
            (sigma_local - sigma_ray).abs() < 1e-5,
            "consistency: sigma_local={:.8} vs sigma_ray={:.8}",
            sigma_local,
            sigma_ray
        );

        // (b) Locality: sample 32 rays where warp is stretching. For each ray, calculate local
        // sigma at all 65 nodes. At least one ray must have max / min > 1.2.
        let mut found_varying = false;
        for ray_idx in 0..32 {
            let origin: [f32; 3] = [
                ray_idx as f32 * 0.1 + 0.1,
                0.3 + ray_idx as f32 * 0.02,
                0.2 + ray_idx as f32 * 0.03,
            ];
            let direction = [0.05f32, 1.0, 0.05];
            let mut sigmas = [0.0f32; FLAME_WAVE_SEGMENTS + 1];
            for node in 0..=FLAME_WAVE_SEGMENTS {
                let t = node as f32 * dt;
                let point = [
                    origin[0] + t * direction[0],
                    origin[1] + t * direction[1],
                    origin[2] + t * direction[2],
                ];
                let (warped, rate) = evaluate_wave_flow_warp_with_rate(
                    &warp_modes,
                    point,
                    direction,
                    1.0,
                    0.35,
                    0.35,
                );
                let (_noise, sigma) =
                    evaluate_wave_noise_local_lowpass(&modes, warped, rate, dt, 0.0);
                sigmas[node] = sigma;
            }
            let min_sigma = sigmas.iter().copied().fold(f32::INFINITY, f32::min);
            let max_sigma = sigmas.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if min_sigma > 1e-6 {
                let ratio = max_sigma / min_sigma;
                if ratio > 1.2 {
                    found_varying = true;
                    break;
                }
            }
        }
        assert!(
            found_varying,
            "no ray showed local sigma variation > 1.2x along the ray"
        );
    }

    #[test]
    fn test_detail_modes_match_fbm_statistics() {
        let modes = generate_wave_detail_modes();

        // Check sqrt(sum(0.5 * amplitude^2)) matches WAVE_NOISE_STD within 1e-4
        let power_sum: f32 = modes.iter().map(|m| 0.5 * m.amplitude * m.amplitude).sum();
        let std_dev = power_sum.sqrt();
        assert!(
            (std_dev - WAVE_NOISE_STD).abs() < 1e-4,
            "detail mode std {:.6} vs expected {:.6}",
            std_dev,
            WAVE_NOISE_STD
        );

        // Check all eddy_rate are 0.0
        for (i, m) in modes.iter().enumerate() {
            assert!(
                m.eddy_rate == 0.0,
                "mode {} has non-zero eddy_rate: {}",
                i,
                m.eddy_rate
            );
        }

    // Sample 8000 random points and check statistics
        let sample_count = 8000;
        let mut sum_val: f64 = 0.0;
        for i in 0..sample_count {
            // Deterministic pseudo-random from index (same technique as mode generation)
            let p = [
                ((i as f64 * PLASTIC_INV).fract() * 10.0) as f32,
                ((i as f64 * PLASTIC_INV_SQ).fract() * 10.0) as f32,
                ((i as f64 / GOLDEN_RATIO).fract() * 10.0) as f32,
            ];
            let mut val: f64 = 0.0;
            for m in &modes {
                let dot = m.k[0] as f64 * p[0] as f64
                    + m.k[1] as f64 * p[1] as f64
                    + m.k[2] as f64 * p[2] as f64;
                val += m.amplitude as f64 * (dot + m.phase as f64).sin();
            }
            sum_val += val;
        }
        let mean = sum_val / sample_count as f64;
        assert!(
            mean.abs() < 0.01,
            "detail mode field mean {:.6} not within |0.01|",
            mean
        );

        // Compute std deviation of the samples
        let mut sum_sq: f64 = 0.0;
        for i in 0..sample_count {
            let p = [
                ((i as f64 * PLASTIC_INV).fract() * 10.0) as f32,
                ((i as f64 * PLASTIC_INV_SQ).fract() * 10.0) as f32,
                ((i as f64 / GOLDEN_RATIO).fract() * 10.0) as f32,
            ];
            let mut val: f64 = 0.0;
            for m in &modes {
                let dot = m.k[0] as f64 * p[0] as f64
                    + m.k[1] as f64 * p[1] as f64
                    + m.k[2] as f64 * p[2] as f64;
                val += m.amplitude as f64 * (dot + m.phase as f64).sin();
            }
            sum_sq += (val - mean) * (val - mean);
        }
        let sample_std = (sum_sq / sample_count as f64).sqrt();
        let expected_std = WAVE_NOISE_STD as f64;
        let lower = expected_std * 0.85;
        let upper = expected_std * 1.15;
        assert!(
            sample_std >= lower && sample_std <= upper,
            "detail mode field std {:.6} not within ±15% of {:.6} (range [{:.6}, {:.6}])",
            sample_std,
            expected_std,
            lower,
            upper
        );
    }

    #[test]
    fn test_apply_wave_envelope_preserves_variance_and_marks_octaves() {
        let mut modes = generate_wave_modes();
        apply_wave_envelope(&mut modes, WAVE_ENV_MU);

        let split = 2.0 * std::f32::consts::TAU;
        let mut power_low = 0.0f64;
        let mut power_high = 0.0f64;
        let mut high_count = 0;
        for mode in &modes {
            let k_mag =
                (mode.k[0] * mode.k[0] + mode.k[1] * mode.k[1] + mode.k[2] * mode.k[2]).sqrt();
            let power = 0.5 * (mode.amplitude as f64) * (mode.amplitude as f64);
            if k_mag < split {
                assert_eq!(mode.env_coeff, 0.0);
                power_low += power;
            } else {
                assert!(mode.env_coeff > 0.0);
                high_count += 1;
                power_high += power;
            }
        }
        assert!(high_count > 0 && high_count < modes.len());

        let mu = WAVE_ENV_MU as f64;
        let total_std = (power_low + (1.0 + mu * mu) * power_high).sqrt();
        assert!(
            (total_std - WAVE_NOISE_STD as f64).abs() < 1e-4,
            "effective std {} != {}",
            total_std,
            WAVE_NOISE_STD
        );

        let coeff = modes.iter().find(|m| m.env_coeff > 0.0).unwrap().env_coeff as f64;
        let expected = mu / power_low.sqrt();
        assert!(
            ((coeff - expected) / expected).abs() < 1e-4,
            "env_coeff {} != mu/sigma_low {}",
            coeff,
            expected
        );
    }
}
