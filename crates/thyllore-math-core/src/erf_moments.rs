//! Power moments of erf and linear-argument Gaussian weights over a centered interval.
//!
//! These are the M_erf / M_gauss families of the sharpness-preserving integration
//! design: with the erf bridge `erf(alpha s + beta)` along a ray, every band integral
//! of (polynomial) x (bridge) reduces to
//!   E(n) = int_{-h}^{h} s^n erf(alpha s + beta) ds
//!   J(m) = int_{-h}^{h} s^m exp(-(alpha s + beta)^2) ds
//! E follows from J by one integration by parts, so both live here.
//!
//! Two regimes keep the evaluation well conditioned:
//! - |alpha| h <= 0.5: Taylor series around the interval center. The recurrence would
//!   divide by alpha^2 and cancel catastrophically for slowly varying arguments.
//! - |alpha| h > 0.5: integration-by-parts recurrence in normalized sigma = s / h,
//!   whose coefficients stay bounded once the argument varies across the interval.

/// erf with the fractional accuracy of `erfc64` (about 1.2e-7 relative on erfc).
pub fn approximate_erf(x: f32) -> f32 {
    erf64(x as f64) as f32
}

fn erf64(x: f64) -> f64 {
    1.0 - erfc64(x)
}

/// Complementary error function for any real x, fractional error <= 1.2e-7
/// (Numerical Recipes rational-exponential fit). Relative accuracy in the tail is
/// what keeps the moment recurrences stable — an absolute-error erf fit drowns
/// tail differences like erf(-4) - erf(-5.6) in approximation noise.
fn erfc64(x: f64) -> f64 {
    let magnitude = x.abs();
    let t = 1.0 / (1.0 + 0.5 * magnitude);
    let tail = t
        * (-magnitude * magnitude - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    if x >= 0.0 {
        tail
    } else {
        2.0 - tail
    }
}

pub const ERF_MOMENT_COUNT: usize = 7;
pub const GAUSSIAN_MOMENT_COUNT: usize = 8;

const TAYLOR_ALPHA_H: f64 = 0.5;
const TAYLOR_TERMS: usize = 13;
const SATURATION_ARGUMENT: f64 = 5.5;
const SQRT_PI: f64 = 1.772_453_850_905_516;

/// `int_{-1}^{1} sigma^j d sigma`.
fn unit_power_moment(j: usize) -> f64 {
    if j % 2 == 1 {
        0.0
    } else {
        2.0 / (j as f64 + 1.0)
    }
}

/// Taylor coefficients `c_k = (-1)^k H_k(t0) exp(-t0^2) / k!` of `exp(-(t0 + u)^2)`
/// in u, via the physicists' Hermite recurrence.
fn gaussian_taylor_coefficients(t0: f64) -> [f64; TAYLOR_TERMS] {
    let weight = (-t0 * t0).exp();
    let mut coefficients = [0.0; TAYLOR_TERMS];
    coefficients[0] = weight;
    let mut hermite_prev = 1.0;
    let mut hermite = 2.0 * t0;
    let mut factorial = 1.0;
    let mut sign = 1.0;
    for k in 1..TAYLOR_TERMS {
        factorial *= k as f64;
        sign = -sign;
        coefficients[k] = sign * hermite * weight / factorial;
        let hermite_next = 2.0 * t0 * hermite - 2.0 * k as f64 * hermite_prev;
        hermite_prev = hermite;
        hermite = hermite_next;
    }
    coefficients
}

/// Both moment families over `[-half_width, half_width]`:
/// `(E(n) for n = 0..=6, J(m) for m = 0..=7)`.
pub fn integrate_erf_and_gaussian_powers(
    alpha: f32,
    beta: f32,
    half_width: f32,
) -> ([f32; ERF_MOMENT_COUNT], [f32; GAUSSIAN_MOMENT_COUNT]) {
    let mut erf_moments = [0.0f32; ERF_MOMENT_COUNT];
    let mut gaussian_moments = [0.0f32; GAUSSIAN_MOMENT_COUNT];
    if half_width <= 0.0 {
        return (erf_moments, gaussian_moments);
    }

    let h = half_width as f64;
    let alpha_h = alpha as f64 * h;
    let beta = beta as f64;

    // Scale factors h^{j+1} shared by both families (sigma = s / h).
    let mut scale = [0.0f64; GAUSSIAN_MOMENT_COUNT];
    let mut power = h;
    for entry in scale.iter_mut() {
        *entry = power;
        power *= h;
    }

    // Saturated bridge: erf is +-1 over the whole interval and the Gaussian vanishes.
    if beta - alpha_h.abs() > SATURATION_ARGUMENT || beta + alpha_h.abs() < -SATURATION_ARGUMENT {
        let sign = if beta > 0.0 { 1.0 } else { -1.0 };
        for (n, moment) in erf_moments.iter_mut().enumerate() {
            *moment = (sign * unit_power_moment(n) * scale[n]) as f32;
        }
        return (erf_moments, gaussian_moments);
    }

    let mut erf_sigma = [0.0f64; ERF_MOMENT_COUNT];
    let mut gaussian_sigma = [0.0f64; GAUSSIAN_MOMENT_COUNT];

    if alpha_h.abs() <= TAYLOR_ALPHA_H {
        // erf(alpha_h sigma + beta) = erf(beta)
        //   + (2/sqrt(pi)) sum_{k>=1} c_{k-1} alpha_h^k / k * sigma^k,
        // exp(-(alpha_h sigma + beta)^2) = sum_k c_k alpha_h^k sigma^k,
        // with c_k the Gaussian Taylor coefficients at beta (erf' = (2/sqrt(pi)) e^{-t^2}).
        let gaussian_taylor = gaussian_taylor_coefficients(beta);

        let mut erf_series = [0.0f64; TAYLOR_TERMS];
        let mut gaussian_series = [0.0f64; TAYLOR_TERMS];
        erf_series[0] = erf64(beta);
        gaussian_series[0] = gaussian_taylor[0];
        let mut alpha_power = 1.0;
        for k in 1..TAYLOR_TERMS {
            alpha_power *= alpha_h;
            erf_series[k] = (2.0 / SQRT_PI) * gaussian_taylor[k - 1] * alpha_power / k as f64;
            gaussian_series[k] = gaussian_taylor[k] * alpha_power;
        }

        for (n, target) in erf_sigma.iter_mut().enumerate() {
            *target = erf_series
                .iter()
                .enumerate()
                .map(|(k, coefficient)| coefficient * unit_power_moment(n + k))
                .sum();
        }
        for (m, target) in gaussian_sigma.iter_mut().enumerate() {
            *target = gaussian_series
                .iter()
                .enumerate()
                .map(|(k, coefficient)| coefficient * unit_power_moment(m + k))
                .sum();
        }
    } else {
        let t_hi = alpha_h + beta;
        let t_lo = -alpha_h + beta;
        let gauge_hi = (-t_hi * t_hi).exp();
        let gauge_lo = (-t_lo * t_lo).exp();
        // erf_hi - erf_lo == erfc(t_lo) - erfc(t_hi): the erfc form keeps relative
        // accuracy in the tails where the direct difference of erf values cancels.
        let erf_difference = erfc64(t_lo) - erfc64(t_hi);
        let erf_sum = 2.0 - erfc64(t_hi) - erfc64(t_lo);

        gaussian_sigma[0] = SQRT_PI / (2.0 * alpha_h) * erf_difference;
        let inv_two_alpha_sq = 1.0 / (2.0 * alpha_h * alpha_h);
        let beta_over_alpha = beta / alpha_h;
        for j in 1..GAUSSIAN_MOMENT_COUNT {
            // sigma^{j-1} at the bounds: 1 at sigma = 1, (-1)^{j-1} at sigma = -1.
            let boundary = if (j - 1) % 2 == 0 {
                gauge_hi - gauge_lo
            } else {
                gauge_hi + gauge_lo
            };
            let two_back = if j >= 2 { gaussian_sigma[j - 2] } else { 0.0 };
            gaussian_sigma[j] = -inv_two_alpha_sq * boundary
                + (j as f64 - 1.0) * inv_two_alpha_sq * two_back
                - beta_over_alpha * gaussian_sigma[j - 1];
        }

        // E(n) by parts: [sigma^{n+1} erf]_{-1}^{1} feeds the boundary, J(n+1) the rest.
        for (n, target) in erf_sigma.iter_mut().enumerate() {
            let boundary = if (n + 1) % 2 == 0 {
                erf_difference
            } else {
                erf_sum
            };
            *target =
                (boundary - (2.0 / SQRT_PI) * alpha_h * gaussian_sigma[n + 1]) / (n as f64 + 1.0);
        }
    }

    for (n, value) in erf_sigma.iter().enumerate() {
        erf_moments[n] = (value * scale[n]) as f32;
    }
    for (m, value) in gaussian_sigma.iter().enumerate() {
        gaussian_moments[m] = (value * scale[m]) as f32;
    }
    (erf_moments, gaussian_moments)
}

/// `int_{-half_width}^{half_width} s^n erf(alpha s + beta) ds` for n = 0..=6.
pub fn integrate_erf_powers(alpha: f32, beta: f32, half_width: f32) -> [f32; ERF_MOMENT_COUNT] {
    integrate_erf_and_gaussian_powers(alpha, beta, half_width).0
}

/// `int_{-half_width}^{half_width} s^m exp(-(alpha s + beta)^2) ds` for m = 0..=7.
pub fn integrate_gaussian_linear_powers(
    alpha: f32,
    beta: f32,
    half_width: f32,
) -> [f32; GAUSSIAN_MOMENT_COUNT] {
    integrate_erf_and_gaussian_powers(alpha, beta, half_width).1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_moments(
        alpha: f64,
        beta: f64,
        half_width: f64,
    ) -> ([f64; ERF_MOMENT_COUNT], [f64; GAUSSIAN_MOMENT_COUNT]) {
        let steps = 200_000;
        let ds = 2.0 * half_width / steps as f64;
        let mut erf_reference = [0.0f64; ERF_MOMENT_COUNT];
        let mut gaussian_reference = [0.0f64; GAUSSIAN_MOMENT_COUNT];
        for i in 0..steps {
            let s = -half_width + (i as f64 + 0.5) * ds;
            let t = alpha * s + beta;
            let erf_value = erf64(t);
            let gaussian_value = (-t * t).exp();
            let mut s_power = 1.0;
            for n in 0..GAUSSIAN_MOMENT_COUNT {
                if n < ERF_MOMENT_COUNT {
                    erf_reference[n] += s_power * erf_value * ds;
                }
                gaussian_reference[n] += s_power * gaussian_value * ds;
                s_power *= s;
            }
        }
        (erf_reference, gaussian_reference)
    }

    /// Alpha sweep across both regimes (Taylor <= 0.5 / recurrence > 0.5 in alpha*h),
    /// including saturation and sign flips, against f64 quadrature.
    #[test]
    fn test_moments_match_quadrature_across_alpha_sweep() {
        let mut checked = 0;
        for alpha in [
            0.0f32, 1e-4, 0.03, 0.4, 0.9, 4.0, 16.0, 64.0, 1e3, -2.5, -40.0,
        ] {
            for beta in [-9.0f32, -4.8, -1.2, -0.3, 0.0, 0.6, 2.0, 5.2, 10.0] {
                for half_width in [0.05f32, 0.4, 1.0] {
                    let (erf_got, gaussian_got) =
                        integrate_erf_and_gaussian_powers(alpha, beta, half_width);
                    let (erf_expected, gaussian_expected) =
                        reference_moments(alpha as f64, beta as f64, half_width as f64);

                    let h = half_width as f64;
                    let mut envelope = h;
                    for n in 0..GAUSSIAN_MOMENT_COUNT {
                        // |integrand| <= 1, so h^{n+1} bounds the moment scale. The floor
                        // is the erfc fit's 1.2e-7 fractional error amplified by the
                        // recurrence (~1e3 worst case); the bridge fit that consumes these
                        // targets ~5e-3, so 1e-4 of the envelope keeps a 50x margin.
                        let tolerance = 1e-4 * envelope;
                        if n < ERF_MOMENT_COUNT {
                            assert!(
                                (erf_got[n] as f64 - erf_expected[n]).abs() < tolerance,
                                "E({n}) alpha={alpha} beta={beta} h={half_width}: \
                                 got {}, expected {}",
                                erf_got[n],
                                erf_expected[n]
                            );
                        }
                        assert!(
                            (gaussian_got[n] as f64 - gaussian_expected[n]).abs() < tolerance,
                            "J({n}) alpha={alpha} beta={beta} h={half_width}: \
                             got {}, expected {}",
                            gaussian_got[n],
                            gaussian_expected[n]
                        );
                        envelope *= h;
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 200);
    }

    /// The two regimes must agree where they meet (alpha * h crossing 0.5).
    #[test]
    fn test_regime_boundary_is_continuous() {
        for beta in [-2.0f32, 0.0, 1.5, 4.0] {
            let half_width = 1.0f32;
            let below = integrate_erf_and_gaussian_powers(0.499, beta, half_width);
            let above = integrate_erf_and_gaussian_powers(0.501, beta, half_width);
            for n in 0..ERF_MOMENT_COUNT {
                assert!(
                    (below.0[n] - above.0[n]).abs() < 2e-3 * half_width + 1e-6,
                    "E({n}) beta={beta}: {} vs {}",
                    below.0[n],
                    above.0[n]
                );
            }
        }
    }

    #[test]
    fn test_saturated_bridge_reduces_to_power_moments() {
        let (erf_moments, gaussian_moments) = integrate_erf_and_gaussian_powers(2.0, 40.0, 0.5);
        let mut envelope = 0.5f64;
        for n in 0..ERF_MOMENT_COUNT {
            let expected = unit_power_moment(n) * envelope;
            assert!(
                (erf_moments[n] as f64 - expected).abs() < 1e-7,
                "E({n}): {} vs {expected}",
                erf_moments[n]
            );
            envelope *= 0.5;
        }
        assert!(gaussian_moments.iter().all(|&j| j == 0.0));
    }

    #[test]
    fn test_odd_symmetry_in_beta() {
        // erf is odd: E(n)(alpha, -beta) = -(-1)^n E(n)(alpha, beta).
        let (positive, _) = integrate_erf_and_gaussian_powers(3.0, 0.8, 0.7);
        let (negative, _) = integrate_erf_and_gaussian_powers(3.0, -0.8, 0.7);
        for n in 0..ERF_MOMENT_COUNT {
            let expected = if n % 2 == 0 {
                -positive[n]
            } else {
                positive[n]
            };
            assert!(
                (negative[n] - expected).abs() < 1e-6,
                "E({n}): {} vs {expected}",
                negative[n]
            );
        }
    }

    #[test]
    fn test_approximate_erf_matches_known_values() {
        for (x, expected) in [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (2.0, 0.995_322_3),
            (-1.5, -0.966_105_1),
        ] {
            assert!(
                (approximate_erf(x) - expected).abs() < 2e-6,
                "erf({x}) = {}, expected {expected}",
                approximate_erf(x)
            );
        }
    }

    #[test]
    fn test_empty_interval_is_zero() {
        let (erf_moments, gaussian_moments) = integrate_erf_and_gaussian_powers(1.0, 0.0, 0.0);
        assert!(erf_moments.iter().all(|&value| value == 0.0));
        assert!(gaussian_moments.iter().all(|&value| value == 0.0));
    }
}
