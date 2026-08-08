//! Wave-basis erosion noise: an analytic sum of random wave modes
//!   e(w) = sum_n a_n sin(k_n . w + phi_n + eddy_n t)
//! the erosion source of the flame field (the sole turbulence basis since P6).
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
//! This module holds only what the product needs: the deterministic mode
//! tables and the UBO-side parameters. The CPU evaluation mirrors of the GLSL
//! (flameWaveNoiseSum / flameWaveOccupancySegments in flame_noise_field.glsl /
//! flame_radial_integral.glsl) live in thyllore-render-debug (test-only crate).

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
    /// Rank-M phase-jitter mixing coefficients: the carrier phase gains
    /// `sum_m jitter[m] * Psi_m(w)` over the shared fields WAVE_JITTER_K.
    pub jitter: [f32; WAVE_JITTER_RANK],
}

/// Rank-M phase jitter of the erosion carriers. The bottom octave packs many
/// near-equal |k| modes whose pairwise difference frequencies pile up on the
/// same low screen frequency — a collective beat that renders as quasi-periodic
/// fringes and that a SINGLE shared modulation field cannot break (every pair
/// difference stays proportional to that one field; fringe_beat_analysis.md).
/// With M >= 2 independent low-wavenumber fields Psi_m(w) = sin(kappa_m . w + phi_m)
/// and per-mode mixing coefficients c_{n,m}, the beat phase of pair (i, j)
/// gains `sum_m (c_i - c_j)_m Psi_m` — a pair-specific spatial modulation, so
/// the beats cannot align into one fringe family. Phase-only: |k| is untouched,
/// the per-mode power and the erf closed form are exact as before; the jitter's
/// ray rate joins the node low-pass beta like the cf psiRateVec.
pub const WAVE_JITTER_RANK: usize = 3;
/// |kappa| = 2*pi*{0.45, 0.62, 0.85}: below the bottom octave (2*pi) so the
/// jitter varies slower than every carrier, comparable to the beat wavenumbers.
/// Directions are unit cyclic permutations of (0.36, 0.48, -0.80).
pub const WAVE_JITTER_K: [[f32; 3]; WAVE_JITTER_RANK] = [
    [1.017_876, 1.357_168, -2.261_947],
    [-3.116_460, 1.402_407, 1.869_876],
    [2.563_540, -4.272_566, 1.922_655],
];
pub const WAVE_JITTER_PHASE: [f32; WAVE_JITTER_RANK] = [1.234_568, 3.456_790, 5.678_901];
/// Max |mixing coefficient| per field: uniform c in [-D, D] gives the per-mode
/// phase jitter a spatial RMS of D * sqrt(RANK / 6) ~= 1.3 rad at D = 1.85 and
/// a pair-difference RMS of ~1.85 rad — enough to decohere a fringe over one
/// jitter wavelength without changing the field statistics (phase-only).
pub const WAVE_JITTER_DEPTH: f32 = 1.85;

/// Shared jitter fields at a warped coordinate: value Psi_m and its ray rate
/// dPsi_m/dt for rate = dw/dt (zero rate is fine for point evaluation).
/// The field wavevectors are WAVE_JITTER_K scaled by the runtime kappa scale
/// (read_env_wave_jitter_freq / waveJitter[0].w on the GPU).
/// Mirror of flameWaveJitterState in flame_noise_field.glsl.
pub fn wave_jitter_state(
    w: [f32; 3],
    rate: [f32; 3],
) -> ([f32; WAVE_JITTER_RANK], [f32; WAVE_JITTER_RANK]) {
    let scale = read_env_wave_jitter_freq();
    let mut psi = [0.0f32; WAVE_JITTER_RANK];
    let mut psi_rate = [0.0f32; WAVE_JITTER_RANK];
    for m in 0..WAVE_JITTER_RANK {
        let k = WAVE_JITTER_K[m];
        let angle =
            scale * (k[0] * w[0] + k[1] * w[1] + k[2] * w[2]) + WAVE_JITTER_PHASE[m];
        psi[m] = angle.sin();
        psi_rate[m] =
            angle.cos() * scale * (k[0] * rate[0] + k[1] * rate[1] + k[2] * rate[2]);
    }
    (psi, psi_rate)
}

/// Per-mode jitter phase `sum_m c_{n,m} * Psi_m` from a precomputed field state.
pub fn wave_mode_jitter_phase(jitter: &[f32; WAVE_JITTER_RANK], psi: &[f32; WAVE_JITTER_RANK]) -> f32 {
    jitter[0] * psi[0] + jitter[1] * psi[1] + jitter[2] * psi[2]
}

static WAVE_JITTER_ENV: OnceLock<f32> = OnceLock::new();

/// Runtime scale of the rank-M phase jitter (THYLLORE_FLAME_WAVE_JITTER,
/// default 1.0; 0 disables). Applied where the mode table is consumed (UBO
/// packing, replay probes) so the deterministic table stays parameter-free.
pub fn read_env_wave_jitter() -> f32 {
    *WAVE_JITTER_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_WAVE_JITTER")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0)
    })
}

static WAVE_JITTER_FREQ_ENV: OnceLock<f32> = OnceLock::new();

/// Runtime scale of the jitter field wavevectors (THYLLORE_FLAME_WAVE_JITTER_FREQ,
/// default WAVE_JITTER_FREQ_DEFAULT). Larger = the jitter varies on a finer
/// spatial scale, decohering beats within a smaller view. 3.0 is the measured
/// knee of the close-up tip fringes (probe peak 32711 off / 23189 at 1.0 /
/// 9155 at 3.0, no further gain at 4-6); at 1.0 the jitter wavelength exceeds
/// a close-up view and the beats stay locally coherent.
pub const WAVE_JITTER_FREQ_DEFAULT: f32 = 3.0;

pub fn read_env_wave_jitter_freq() -> f32 {
    *WAVE_JITTER_FREQ_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_WAVE_JITTER_FREQ")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(WAVE_JITTER_FREQ_DEFAULT)
    })
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
        // Jitter mixing coefficients: three low-discrepancy streams (sqrt(2)-1,
        // sqrt(3)-1, sqrt(5)-2) decorrelated from the direction/magnitude/phase
        // streams, uniform in [-DEPTH, DEPTH].
        const JITTER_STREAMS: [f64; WAVE_JITTER_RANK] = [
            0.414_213_562_373_095_1,
            0.732_050_807_568_877_2,
            0.236_067_977_499_789_7,
        ];
        for (m, stream) in JITTER_STREAMS.iter().enumerate() {
            let u = (n as f64 * stream + 0.5 * PLASTIC_INV * (m as f64 + 1.0)).fract();
            mode.jitter[m] = ((2.0 * u - 1.0) * WAVE_JITTER_DEPTH as f64) as f32;
        }
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

/// Scales `warp_amp * mix(0.15, 1, h)` into the shear strength: at the default
/// warp_amp of 1.4 the top of the flame matches the displacement warp's
/// deformation.
pub const WAVE_SHEAR_STRENGTH_SCALE: f32 = 0.96;

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

#[cfg(test)]
mod tests {
    use super::*;

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
