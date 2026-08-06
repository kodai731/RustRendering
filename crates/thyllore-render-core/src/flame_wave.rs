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

/// UBO slot capacity (2 vec4 per mode).
pub const WAVE_MODE_SLOTS: usize = 112;
/// Active mode count (N=96: N=48 still showed fine interference fringes on
/// the real flame under the warp; N=16 reads as discrete elements).
pub const WAVE_MODE_COUNT: usize = 96;
/// Uniform segments of the band-free closed form over the support crossing.
pub const FLAME_WAVE_SEGMENTS: usize = 64;

/// fbm3 statistics the wave field reproduces so the erosion mapping
/// `amp * mix(0.2, 1, h) * (noise - 0.35)` keeps its calibration:
/// mean = 0.5 * (0.5 + 0.25 + 0.125), std measured over the lattice.
pub const WAVE_NOISE_MEAN: f32 = 0.4375;
pub const WAVE_NOISE_STD: f32 = 0.106;
/// Wavenumber span matching the fbm 3-octave spectrum (lacunarity ~2, 3 octaves).
pub const WAVE_K_RATIO: f32 = 4.0;

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
}

/// Deterministic mode table in normalized units. Every call returns the same
/// table; runtime levers (frequency, anisotropy, advection, amplitude) apply
/// through the shared warped coordinate and the erosion mapping instead.
pub fn generate_wave_modes() -> [WaveMode; WAVE_MODE_COUNT] {
    let count = WAVE_MODE_COUNT;
    let mut modes = [WaveMode::default(); WAVE_MODE_COUNT];
    let mut power_sum = 0.0f64;
    for (n, mode) in modes.iter_mut().enumerate() {
        let vertical = 1.0 - 2.0 * (n as f64 + 0.5) / count as f64;
        let ring = (1.0 - vertical * vertical).max(0.0).sqrt();
        let azimuth = std::f64::consts::TAU * ((n as f64 / GOLDEN_RATIO).fract());
        let magnitude_u = (0.5 / count as f64 + n as f64 * PLASTIC_INV).fract();
        let magnitude =
            std::f64::consts::TAU * ((WAVE_K_RATIO as f64).ln() * magnitude_u).exp();
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
    for mode in &mut modes {
        mode.amplitude *= scale;
    }
    modes
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
    /// Displacement direction (unit); the y component is scaled by
    /// warp_y_scale at evaluation like the fbm warp.
    pub direction: [f32; 3],
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
        mode.direction = [
            displacement[0] as f32,
            displacement[1] as f32,
            displacement[2] as f32,
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
        displacement[0] += mode.direction[0] * value;
        displacement[1] += mode.direction[1] * value;
        displacement[2] += mode.direction[2] * value;
    }
    displacement
}

/// Pointwise wave noise at a warped coordinate (mirror of `flameWaveNoiseSum`):
/// mean-matched to fbm3 so `flameNoiseErosionFromValue` keeps its calibration.
/// `eddy_time` is `noise_scroll_speed * time`.
pub fn evaluate_wave_noise(modes: &[WaveMode], w: [f32; 3], eddy_time: f32) -> f32 {
    let mut sum = WAVE_NOISE_MEAN;
    for mode in modes {
        let angle = mode.k[0] * w[0]
            + mode.k[1] * w[1]
            + mode.k[2] * w[2]
            + mode.phase
            + mode.eddy_rate * eddy_time;
        sum += mode.amplitude * angle.sin();
    }
    sum
}

/// Per-ray split of the mode set into a resolved part (evaluated at the
/// segment nodes, attenuated by the node-spacing low-pass) and an unresolved
/// remainder routed into the smoothed-response sigma. `dir_w` is the linear
/// part of the warped coordinate along t; `node_spacing` the node distance in
/// t. Both the weights and the sigma are smooth in the ray — no per-ray
/// integer mode partition (the appendix-7 quantization trap).
pub fn wave_ray_attenuation(
    modes: &[WaveMode],
    dir_w: [f32; 3],
    node_spacing: f32,
) -> (Vec<f32>, f32) {
    let mut weights = Vec::with_capacity(modes.len());
    let mut unresolved_power = 0.0f32;
    for mode in modes {
        let beta = mode.k[0] * dir_w[0] + mode.k[1] * dir_w[1] + mode.k[2] * dir_w[2];
        let x = beta * node_spacing / std::f32::consts::PI;
        let weight = (-(x * x) * (x * x)).exp();
        weights.push(weight);
        unresolved_power += 0.5 * mode.amplitude * mode.amplitude * (1.0 - weight * weight);
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
    let mut sum = WAVE_NOISE_MEAN;
    for (mode, weight) in modes.iter().zip(weights) {
        let angle = mode.k[0] * w[0]
            + mode.k[1] * w[1]
            + mode.k[2] * w[2]
            + mode.phase
            + mode.eddy_rate * eddy_time;
        sum += weight * mode.amplitude * angle.sin();
    }
    sum
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
/// with the closed-form erf response. `sigma_noise` carries the unresolved
/// mode power; `erosion_scale_at(t)` is the `amp * mix(0.2, 1, h)` mapping and
/// `noise_at(t)` the attenuated wave noise along the ray. Segments whose node
/// densities all vanish are skipped before any erosion work (exact membership
/// at node resolution). Mirror of `flameWaveOccupancySegments`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_wave_occupancy_segments(
    model: &ErfResponseModel,
    t0: f32,
    t1: f32,
    density_at: impl Fn(f32) -> f32,
    noise_at: impl Fn(f32) -> f32,
    erosion_scale_at: impl Fn(f32) -> f32,
    sigma_noise: f32,
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
    for node in 0..=FLAME_WAVE_SEGMENTS {
        let adjacent_support = density[node] > 0.0
            || (node > 0 && density[node - 1] > 0.0)
            || (node < FLAME_WAVE_SEGMENTS && density[node + 1] > 0.0);
        argument[node] = if adjacent_support {
            let t = t0 + node as f32 * dt;
            let erosion = erosion_scale_at(t) * (noise_at(t) - 0.35);
            eroded_argument(density[node], erosion, flood_fade_scale)
        } else {
            0.0
        };
    }

    for (segment, slot) in segments.iter_mut().enumerate() {
        let t_prev = t0 + segment as f32 * dt;
        slot.t_mean = t_prev + 0.5 * dt;
        if density[segment] <= 0.0 && density[segment + 1] <= 0.0 {
            continue;
        }
        let sigma_eff = sigma_noise
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
        let (weights, sigma) = wave_ray_attenuation(&modes, [0.01, 0.0, 0.0], 0.1);
        assert!(weights.iter().all(|w| *w > 0.99));
        assert!(sigma < 1e-3);
        // Fast ray: the top of the spectrum is unresolved, sigma bounded by the
        // total field std.
        let (weights_fast, sigma_fast) = wave_ray_attenuation(&modes, [1.0, 0.4, 0.2], 0.5);
        assert!(weights_fast.iter().any(|w| *w < 0.5));
        assert!(sigma_fast > 0.01 && sigma_fast < WAVE_NOISE_STD * 1.01);
    }

    /// Closed form vs dense quadrature of the true pointwise field, in a
    /// regime where the nodes resolve every mode (the model's exact limit).
    #[test]
    fn test_occupancy_matches_dense_quadrature_when_resolved() {
        let modes = generate_wave_modes();
        let model = fit_erf_response(0.27, 0.33);
        let flood_fade_scale = 0.33;
        let carve_residual = 0.12;
        let amp = 1.5;
        let (t0, t1) = (0.0f32, 0.4f32);

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
            let (weights, sigma_noise) = wave_ray_attenuation(&modes, dir, dt);
            let segments = evaluate_wave_occupancy_segments(
                &model,
                t0,
                t1,
                density_at,
                |t| evaluate_wave_noise_attenuated(&modes, &weights, warped_at(t), 0.0),
                erosion_scale_at,
                sigma_noise,
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
        let model = fit_erf_response(0.27, 0.33);
        let dir = [0.1f32, 1.0, 0.05];
        let (t0, t1) = (0.0f32, 1.0f32);
        let dt = (t1 - t0) / FLAME_WAVE_SEGMENTS as f32;
        let (weights, sigma_noise) = wave_ray_attenuation(&modes, dir, dt);

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
                |t| evaluate_wave_noise_attenuated(&modes, &weights, warped_at(t), 0.0),
                |t| 1.5 * (0.2 + 0.8 * height_at(t)),
                sigma_noise,
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
}
