//! Continuous-functional integral of the erf-bridge response along a linear
//! mean argument carrying an oscillatory perturbation (P1b of the continuous
//! ray integrator redesign):
//!   int_{s0}^{s1} phi(x0 + x1 s - sum_n a_n sin(omega_n s + phase_n)) ds
//! The smooth mean line is integrated with the existing closed form; each mode
//! is split by its capture rate g_n = exp(-k_n^2), k_n = omega_n alpha / sqrt(2),
//! alpha = s_tot / |x1| (transition-shell crossing time): the uncaptured share
//! folds into the smoothing sigma (fixed point, monotone), the captured share
//! enters as the exact Gaussian x sine integral of the erf main term via the
//! Faddeeva stable form. The estimator is a continuous functional of [s0, s1]
//! — no sampling lattice exists, so no lattice can imprint (fringe source 4).

use crate::erf_response::{integrate_erf_response_linear, smooth_erf_response, ErfResponseModel};
use crate::faddeeva::f_stable;

/// One resolved wave mode of the argument perturbation, in argument units:
/// the argument is `x0 + x1 s - amplitude * sin(omega s + phase)`.
#[derive(Clone, Copy, Debug)]
pub struct OscillatoryMode {
    pub amplitude: f32,
    pub omega: f32,
    pub phase: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OscillatoryErfIntegral {
    pub integral: f32,
    pub mean_integral: f32,
    pub first_moment: f32,
    pub sigma_smooth: f32,
    pub linear_correction: f32,
    pub captured_count: usize,
}

const FIXED_POINT_ITERATIONS: usize = 3;
const CAPTURE_SKIP_THRESHOLD: f32 = 1e-6;
const FLAT_SLOPE_THRESHOLD: f32 = 1e-6;
const SQRT_2: f32 = std::f32::consts::SQRT_2;

fn capture_rate(omega: f32, alpha: f32) -> f32 {
    if omega == 0.0 {
        return 1.0;
    }
    let k = omega.abs() * alpha / SQRT_2;
    if k.is_finite() {
        (-k * k).exp()
    } else {
        0.0
    }
}

fn erf_main_width(model: &ErfResponseModel, sigma: f32) -> f32 {
    1.0 / (SQRT_2 * smooth_erf_response(model, sigma).kappa_eff)
}

fn solve_sigma_fixed_point(
    model: &ErfResponseModel,
    sigma_base: f32,
    sigma_floor: f32,
    slope: f32,
    modes: &[OscillatoryMode],
    captures: &mut [f32],
) -> f32 {
    let mode_power: f32 = modes.iter().map(|m| 0.5 * m.amplitude * m.amplitude).sum();
    let mut folded_power = mode_power;
    let mut sigma_smooth = 0.0;
    for _ in 0..FIXED_POINT_ITERATIONS {
        sigma_smooth = (sigma_base * sigma_base + folded_power)
            .sqrt()
            .max(sigma_floor);
        let shell_width = erf_main_width(model, sigma_smooth);
        let alpha = if slope.abs() > FLAT_SLOPE_THRESHOLD {
            shell_width / slope.abs()
        } else {
            f32::INFINITY
        };
        folded_power = 0.0;
        for (mode, capture) in modes.iter().zip(captures.iter_mut()) {
            *capture = capture_rate(mode.omega, alpha);
            folded_power += 0.5 * mode.amplitude * mode.amplitude * (1.0 - *capture * *capture);
        }
    }
    (sigma_base * sigma_base + folded_power)
        .sqrt()
        .max(sigma_floor)
}

fn linear_correction_sloped(
    model: &ErfResponseModel,
    sigma_smooth: f32,
    x0: f32,
    x1: f32,
    s0: f32,
    s1: f32,
    modes: &[OscillatoryMode],
    captures: &[f32],
) -> f32 {
    let smoothed = smooth_erf_response(model, sigma_smooth);
    let shell_width = 1.0 / (SQRT_2 * smoothed.kappa_eff);
    let alpha = shell_width / x1.abs();
    let cross = (smoothed.center - x0) / x1;
    let xa = (s0 - cross) / (SQRT_2 * alpha);
    let xb = (s1 - cross) / (SQRT_2 * alpha);
    let mut correction = 0.0;
    for (mode, capture) in modes.iter().zip(captures.iter()) {
        if *capture < CAPTURE_SKIP_THRESHOLD {
            continue;
        }
        let (omega, phase, amplitude) = if mode.omega < 0.0 {
            (-mode.omega, -mode.phase, -mode.amplitude)
        } else {
            (mode.omega, mode.phase, mode.amplitude)
        };
        let k = omega * alpha / SQRT_2;
        let (br, bi) = f_stable(xb, k);
        let (ar, ai) = f_stable(xa, k);
        let (dr, di) = (br - ar, bi - ai);
        let theta = omega * cross + phase;
        let imaginary = theta.sin() * dr + theta.cos() * di;
        correction += amplitude * imaginary / (2.0 * x1.abs());
    }
    correction
}

fn linear_correction_flat(
    model: &ErfResponseModel,
    sigma_smooth: f32,
    x0: f32,
    x1: f32,
    s0: f32,
    s1: f32,
    modes: &[OscillatoryMode],
    captures: &[f32],
) -> f32 {
    let smoothed = smooth_erf_response(model, sigma_smooth);
    let mid = 0.5 * (s0 + s1);
    let span = s1 - s0;
    let u = smoothed.kappa_eff * (x0 + x1 * mid - smoothed.center);
    let response_slope =
        smoothed.kappa_eff * (-u * u).exp() * 0.5 * std::f32::consts::FRAC_2_SQRT_PI;
    let mut correction = 0.0;
    for (mode, capture) in modes.iter().zip(captures.iter()) {
        if *capture < CAPTURE_SKIP_THRESHOLD {
            continue;
        }
        let sine_integral = if mode.omega.abs() * span < 1e-6 {
            (mode.omega * mid + mode.phase).sin() * span
        } else {
            ((mode.omega * s0 + mode.phase).cos() - (mode.omega * s1 + mode.phase).cos())
                / mode.omega
        };
        correction += capture * response_slope * mode.amplitude * sine_integral;
    }
    correction
}

/// Capture rates and the folded smoothing sigma of the fixed point, for
/// callers that track the captured modes in the argument itself (deep
/// modulation regime): returns `(sigma_smooth, g_n per mode)`.
pub fn solve_capture_rates(
    model: &ErfResponseModel,
    sigma_base: f32,
    sigma_floor: f32,
    slope: f32,
    modes: &[OscillatoryMode],
) -> (f32, Vec<f32>) {
    let mut captures = vec![0.0f32; modes.len()];
    let sigma_smooth =
        solve_sigma_fixed_point(model, sigma_base, sigma_floor, slope, modes, &mut captures);
    (sigma_smooth, captures)
}

/// Mean (statistical) part only: the sigma fixed point folds every uncaptured
/// mode into the smoothing sigma and the smooth mean line is integrated with
/// the closed form. The captured share must be added separately through a
/// grid-independent [`ShellCrossingCorrection`] — re-linearizing it per
/// segment imprints the segment lattice (P1b diagnosis, 2026-08-09).
pub fn integrate_erf_response_statistical(
    model: &ErfResponseModel,
    sigma_base: f32,
    sigma_floor: f32,
    x0: f32,
    x1: f32,
    s0: f32,
    s1: f32,
    modes: &[OscillatoryMode],
) -> (f32, f32, f32) {
    if s1 <= s0 {
        return (0.0, 0.0, sigma_base.max(sigma_floor));
    }
    let mut captures = vec![0.0f32; modes.len()];
    let sigma_smooth =
        solve_sigma_fixed_point(model, sigma_base, sigma_floor, x1, modes, &mut captures);
    let (integral, first_moment) =
        integrate_erf_response_linear(model, sigma_smooth, x0, x1, s0, s1);
    (integral, first_moment, sigma_smooth)
}

/// One captured mode of a shell crossing, reduced to the slice-evaluation
/// form: amplitude x Im(e^{i theta} (F(xb, k) - F(xa, k))).
#[derive(Clone, Copy, Debug)]
pub struct CrossingMode {
    pub amplitude: f32,
    pub k: f32,
    pub theta: f32,
}

/// Linear (captured) correction of one mean-line shell crossing, linearized
/// once at the crossing itself so that no segment grid enters: slice
/// evaluations over adjacent intervals telescope exactly.
#[derive(Clone, Debug)]
pub struct ShellCrossingCorrection {
    pub center_time: f32,
    pub alpha: f32,
    pub inv_two_slope: f32,
    pub sigma_smooth: f32,
    pub modes: Vec<CrossingMode>,
}

/// Build the crossing correction at `center_time` (where the mean line hits
/// the response center) with local slope `slope`. Returns None when the mean
/// line is too flat to define a crossing (deep grazing = statistical limit).
pub fn solve_shell_crossing(
    model: &ErfResponseModel,
    sigma_base: f32,
    sigma_floor: f32,
    center_time: f32,
    slope: f32,
    modes: &[OscillatoryMode],
) -> Option<ShellCrossingCorrection> {
    if slope.abs() <= FLAT_SLOPE_THRESHOLD {
        return None;
    }
    let mut captures = vec![0.0f32; modes.len()];
    let sigma_smooth =
        solve_sigma_fixed_point(model, sigma_base, sigma_floor, slope, modes, &mut captures);
    let alpha = erf_main_width(model, sigma_smooth) / slope.abs();
    let kept: Vec<CrossingMode> = modes
        .iter()
        .zip(captures.iter())
        .filter(|(_, capture)| **capture >= CAPTURE_SKIP_THRESHOLD)
        .map(|(mode, _)| {
            let (omega, phase, amplitude) = if mode.omega < 0.0 {
                (-mode.omega, -mode.phase, -mode.amplitude)
            } else {
                (mode.omega, mode.phase, mode.amplitude)
            };
            CrossingMode {
                amplitude,
                k: omega * alpha / SQRT_2,
                theta: omega * center_time + phase,
            }
        })
        .collect();
    Some(ShellCrossingCorrection {
        center_time,
        alpha,
        inv_two_slope: 1.0 / (2.0 * slope.abs()),
        sigma_smooth,
        modes: kept,
    })
}

/// Correction contributed by the slice `[s0, s1]`; subtract from the mean
/// integral of the same slice. Adjacent slices telescope to the whole-basin
/// value regardless of how the basin is partitioned.
pub fn shell_crossing_slice(crossing: &ShellCrossingCorrection, s0: f32, s1: f32) -> f32 {
    if s1 <= s0 || crossing.modes.is_empty() {
        return 0.0;
    }
    let xa = (s0 - crossing.center_time) / (SQRT_2 * crossing.alpha);
    let xb = (s1 - crossing.center_time) / (SQRT_2 * crossing.alpha);
    let mut sum = 0.0;
    for mode in &crossing.modes {
        let (br, bi) = f_stable(xb, mode.k);
        let (ar, ai) = f_stable(xa, mode.k);
        let (dr, di) = (br - ar, bi - ai);
        sum += mode.amplitude * (mode.theta.sin() * dr + mode.theta.cos() * di);
    }
    sum * crossing.inv_two_slope
}

/// Integral and mean-weighted first moment of the smoothed response along the
/// oscillating linear argument over `[s0, s1]`. `sigma_base` is the non-modal
/// unresolved std (argument units); `sigma_floor` bounds the smoothing sigma
/// from below. The first moment is taken from the mean term only.
#[allow(clippy::too_many_arguments)]
pub fn integrate_erf_response_oscillatory(
    model: &ErfResponseModel,
    sigma_base: f32,
    sigma_floor: f32,
    x0: f32,
    x1: f32,
    s0: f32,
    s1: f32,
    modes: &[OscillatoryMode],
) -> OscillatoryErfIntegral {
    if s1 <= s0 {
        return OscillatoryErfIntegral::default();
    }

    let mut captures = vec![0.0f32; modes.len()];
    let sigma_smooth =
        solve_sigma_fixed_point(model, sigma_base, sigma_floor, x1, modes, &mut captures);

    let (mean_integral, first_moment) =
        integrate_erf_response_linear(model, sigma_smooth, x0, x1, s0, s1);

    let linear_correction = if x1.abs() > FLAT_SLOPE_THRESHOLD {
        linear_correction_sloped(model, sigma_smooth, x0, x1, s0, s1, modes, &captures)
    } else {
        linear_correction_flat(model, sigma_smooth, x0, x1, s0, s1, modes, &captures)
    };

    OscillatoryErfIntegral {
        integral: mean_integral - linear_correction,
        mean_integral,
        first_moment,
        sigma_smooth,
        linear_correction,
        captured_count: captures
            .iter()
            .filter(|g| **g >= CAPTURE_SKIP_THRESHOLD)
            .count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erf_response::{evaluate_erf_response, fit_erf_response};

    const EDGE_LOW: f32 = 0.27;
    const EDGE_HIGH: f32 = 0.33;

    fn quadrature_reference(
        model: &ErfResponseModel,
        sigma_base: f32,
        x0: f32,
        x1: f32,
        s0: f32,
        s1: f32,
        modes: &[OscillatoryMode],
    ) -> f32 {
        let steps = 200_000;
        let ds = (s1 - s0) as f64 / steps as f64;
        let mut total = 0.0f64;
        for i in 0..steps {
            let s = s0 as f64 + (i as f64 + 0.5) * ds;
            let mut x = x0 as f64 + x1 as f64 * s;
            for mode in modes {
                x -= mode.amplitude as f64 * (mode.omega as f64 * s + mode.phase as f64).sin();
            }
            total += evaluate_erf_response(model, x as f32, sigma_base) as f64 * ds;
        }
        total as f32
    }

    #[test]
    fn test_no_modes_matches_linear_closed_form() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        for (sigma_base, floor) in [(0.0f32, 0.0f32), (0.05, 0.0), (0.02, 0.08)] {
            let result = integrate_erf_response_oscillatory(
                &model,
                sigma_base,
                floor,
                0.6,
                -0.5,
                0.0,
                1.0,
                &[],
            );
            let sigma = sigma_base.max(floor);
            let (expected, expected_moment) =
                integrate_erf_response_linear(&model, sigma, 0.6, -0.5, 0.0, 1.0);
            assert!((result.integral - expected).abs() < 1e-6);
            assert!((result.first_moment - expected_moment).abs() < 1e-6);
            assert!(result.linear_correction.abs() < 1e-9);
        }
    }

    #[test]
    fn test_captured_mode_tracks_realization() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        let modes = [OscillatoryMode {
            amplitude: 0.02,
            omega: 3.0,
            phase: 1.0,
        }];
        let (x0, x1) = (0.65f32, -0.6f32);
        let reference = quadrature_reference(&model, 0.0, x0, x1, 0.0, 1.0, &modes);
        let result = integrate_erf_response_oscillatory(&model, 0.0, 0.0, x0, x1, 0.0, 1.0, &modes);
        let mean_only = result.mean_integral;
        assert_eq!(result.captured_count, 1);
        assert!(
            (result.integral - reference).abs() < 2e-3,
            "integral {} vs reference {}",
            result.integral,
            reference
        );
        assert!(
            (result.integral - reference).abs() < (mean_only - reference).abs(),
            "linear correction must improve on the mean term alone"
        );
    }

    #[test]
    fn test_uncaptured_modes_reach_statistical_limit() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        let modes: Vec<OscillatoryMode> = (0..8)
            .map(|i| OscillatoryMode {
                amplitude: 0.03,
                omega: 4000.0 + 900.0 * i as f32,
                phase: 0.7 * i as f32,
            })
            .collect();
        let (x0, x1) = (0.65f32, -0.6f32);
        let result =
            integrate_erf_response_oscillatory(&model, 0.01, 0.0, x0, x1, 0.0, 1.0, &modes);
        let folded: f32 = modes.iter().map(|m| 0.5 * m.amplitude * m.amplitude).sum();
        let sigma_stat = (0.01f32 * 0.01 + folded).sqrt();
        let (expected, _) = integrate_erf_response_linear(&model, sigma_stat, x0, x1, 0.0, 1.0);
        assert!((result.sigma_smooth - sigma_stat).abs() < 1e-5);
        assert!((result.integral - expected).abs() < 1e-5);
    }

    #[test]
    fn test_interval_splitting_is_invariant() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        let modes: Vec<OscillatoryMode> = (0..12)
            .map(|i| OscillatoryMode {
                amplitude: 0.015,
                omega: 2.0 + 55.0 * i as f32,
                phase: 1.3 * i as f32,
            })
            .collect();
        let (x0, x1) = (0.62f32, -0.55f32);
        let whole = integrate_erf_response_oscillatory(&model, 0.01, 0.0, x0, x1, 0.0, 1.0, &modes);
        let mut split_total = 0.0f32;
        for piece in 0..16 {
            let s0 = piece as f32 / 16.0;
            let s1 = (piece + 1) as f32 / 16.0;
            split_total +=
                integrate_erf_response_oscillatory(&model, 0.01, 0.0, x0, x1, s0, s1, &modes)
                    .integral;
        }
        assert!(
            (whole.integral - split_total).abs() < 1e-5,
            "whole {} vs split {}",
            whole.integral,
            split_total
        );
    }

    #[test]
    fn test_shell_crossing_slices_telescope_exactly() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        let modes: Vec<OscillatoryMode> = (0..10)
            .map(|i| OscillatoryMode {
                amplitude: 0.01,
                omega: 1.5 + 30.0 * i as f32,
                phase: 0.9 * i as f32,
            })
            .collect();
        let crossing = solve_shell_crossing(&model, 0.01, 0.0, 0.55, -0.6, &modes).unwrap();
        let whole = shell_crossing_slice(&crossing, 0.0, 1.0);
        let mut split = 0.0f32;
        for piece in 0..64 {
            split +=
                shell_crossing_slice(&crossing, piece as f32 / 64.0, (piece + 1) as f32 / 64.0);
        }
        assert!(
            (whole - split).abs() < 1e-6,
            "whole {} vs split {}",
            whole,
            split
        );
    }

    #[test]
    fn test_shell_crossing_matches_oscillatory_linear_term() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        let modes = [
            OscillatoryMode {
                amplitude: 0.02,
                omega: 3.0,
                phase: 1.0,
            },
            OscillatoryMode {
                amplitude: 0.01,
                omega: -7.0,
                phase: 0.4,
            },
        ];
        let (x0, x1) = (0.65f32, -0.6f32);
        let oscillatory =
            integrate_erf_response_oscillatory(&model, 0.0, 0.0, x0, x1, 0.0, 1.0, &modes);
        let center_time = (model.center - x0) / x1;
        let crossing = solve_shell_crossing(&model, 0.0, 0.0, center_time, x1, &modes).unwrap();
        let slice = shell_crossing_slice(&crossing, 0.0, 1.0);
        assert!(
            (slice - oscillatory.linear_correction).abs() < 2e-6,
            "crossing {} vs oscillatory {}",
            slice,
            oscillatory.linear_correction
        );
    }

    #[test]
    fn test_statistical_plus_crossing_tracks_realization() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        let modes = [OscillatoryMode {
            amplitude: 0.02,
            omega: 3.0,
            phase: 1.0,
        }];
        let (x0, x1) = (0.65f32, -0.6f32);
        let reference = quadrature_reference(&model, 0.0, x0, x1, 0.0, 1.0, &modes);
        let center_time = (model.center - x0) / x1;
        let crossing = solve_shell_crossing(&model, 0.0, 0.0, center_time, x1, &modes).unwrap();
        let mut total = 0.0f32;
        for piece in 0..8 {
            let (s0, s1) = (piece as f32 / 8.0, (piece + 1) as f32 / 8.0);
            let (mean, _, _) =
                integrate_erf_response_statistical(&model, 0.0, 0.0, x0, x1, s0, s1, &modes);
            total += mean - shell_crossing_slice(&crossing, s0, s1);
        }
        assert!(
            (total - reference).abs() < 2e-3,
            "total {} vs reference {}",
            total,
            reference
        );
    }

    #[test]
    fn test_flat_argument_keeps_constant_offset_exact() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        let modes = [OscillatoryMode {
            amplitude: 0.002,
            omega: 0.0,
            phase: 0.3,
        }];
        let x0 = 0.31f32;
        let reference = quadrature_reference(&model, 0.0, x0, 0.0, 0.0, 1.0, &modes);
        let result =
            integrate_erf_response_oscillatory(&model, 0.0, 0.0, x0, 0.0, 0.0, 1.0, &modes);
        assert!(
            (result.integral - reference).abs() < 3e-3,
            "integral {} vs reference {}",
            result.integral,
            reference
        );
        assert!(
            (result.integral - reference).abs() < (result.mean_integral - reference).abs(),
            "constant offset must be tracked by the flat-branch linear term"
        );
    }

    #[test]
    fn test_flat_argument_folds_oscillation_to_statistics() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        let modes = [OscillatoryMode {
            amplitude: 0.02,
            omega: 40.0,
            phase: 0.3,
        }];
        let x0 = 0.31f32;
        let result =
            integrate_erf_response_oscillatory(&model, 0.0, 0.0, x0, 0.0, 0.0, 1.0, &modes);
        let sigma_stat = (0.5f32 * 0.02 * 0.02).sqrt();
        let (expected, _) = integrate_erf_response_linear(&model, sigma_stat, x0, 0.0, 0.0, 1.0);
        assert!((result.sigma_smooth - sigma_stat).abs() < 1e-6);
        assert!((result.integral - expected).abs() < 1e-6);
    }
}
