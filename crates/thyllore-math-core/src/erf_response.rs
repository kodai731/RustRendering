//! Additive erf-bridge model of the erosion threshold response.
//!
//! The styled flame field applies `smoothstep(edge_low, edge_high, x)` to the
//! argument `x = dSmooth - erosion`. The closed-form integrator replaces that
//! pointwise nonlinearity with the additive split (sharp term + localized odd
//! corrections, all globally smooth — no window, no joints):
//!   phi(x) = 0.5 (1 + erf(xi)) + c1 psi(xi) + c2 psi(xi / 2),
//!   xi = kappa (x - center),  psi(t) = t exp(-t^2).
//! `kappa = 3.6 / width` is the one-term erf fit of a smoothstep (max err 0.0205
//! independent of width); the two odd Gaussian corrections absorb most of the
//! smoothstep-minus-erf residual.
//!
//! Unresolved band noise of std `sigma` smooths every term inside its own family
//! (the mean-variance rule absorbed into the bridge): with `tau = kappa sigma`,
//!   E[erf(kappa (y - Z))]      = erf(kappa_eff y),      kappa_eff = kappa r1,
//!   E[psi(kappa (y - Z))]      = r1^2 psi(kappa_eff y), r1 = 1/sqrt(1 + 2 tau^2),
//!   E[psi(kappa (y - Z) / 2)]  = r2^2 psi(kappa r2 y / 2), r2 = 1/sqrt(1 + tau^2/2),
//! so sigma -> 0 restores the fitted sharpness and large sigma flattens the
//! response, exactly as the mean-variance program requires.
//! Mirrored in shaders/include/flame_erosion_response.glsl.

use crate::erf_moments::{erf64, integrate_erf_and_gaussian_powers};
use crate::smooth_step::smooth_step;
const FIT_SAMPLE_COUNT: usize = 257;
const FIT_XI_RANGE: f64 = 6.0;
const KAPPA_WIDTH_FACTOR: f64 = 3.6;
const MIN_EDGE_WIDTH: f64 = 1e-3;

/// Bridge model of `smoothstep(edge_low, edge_high, x)`; packs into one vec4.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErfResponseModel {
    pub center: f32,
    pub kappa: f32,
    pub gaussian_weights: [f32; 2],
}

impl ErfResponseModel {
    pub fn pack(&self) -> [f32; 4] {
        [
            self.center,
            self.kappa,
            self.gaussian_weights[0],
            self.gaussian_weights[1],
        ]
    }
}

fn psi(t: f64) -> f64 {
    t * (-t * t).exp()
}

/// Fit the additive erf-bridge response to `smoothstep(edge_low, edge_high, x)`.
/// kappa is fixed by the width rule; the two odd Gaussian weights are the least
/// squares solution over a grid covering the transition and both tails.
pub fn fit_erf_response(edge_low: f32, edge_high: f32) -> ErfResponseModel {
    let low = edge_low as f64;
    let width = (edge_high as f64 - low).max(MIN_EDGE_WIDTH);
    let center = low + 0.5 * width;
    let kappa = KAPPA_WIDTH_FACTOR / width;

    let mut normal = [[0.0f64; 2]; 2];
    let mut rhs = [0.0f64; 2];
    for sample in 0..FIT_SAMPLE_COUNT {
        let xi = -FIT_XI_RANGE + 2.0 * FIT_XI_RANGE * sample as f64 / (FIT_SAMPLE_COUNT - 1) as f64;
        let x = center + xi / kappa;
        let residual = smooth_step((x - low) / width) - 0.5 * (1.0 + erf64(xi));
        let basis = [psi(xi), psi(0.5 * xi)];
        for row in 0..2 {
            rhs[row] += basis[row] * residual;
            for column in 0..2 {
                normal[row][column] += basis[row] * basis[column];
            }
        }
    }
    let determinant = normal[0][0] * normal[1][1] - normal[0][1] * normal[1][0];
    assert!(
        determinant.abs() > 1e-14,
        "degenerate erf response fit basis"
    );
    let weights = [
        (normal[1][1] * rhs[0] - normal[0][1] * rhs[1]) / determinant,
        (normal[0][0] * rhs[1] - normal[1][0] * rhs[0]) / determinant,
    ];

    ErfResponseModel {
        center: center as f32,
        kappa: kappa as f32,
        gaussian_weights: [weights[0] as f32, weights[1] as f32],
    }
}

/// Per-family smoothing constants for unresolved noise of std `sigma` in x units.
#[derive(Clone, Copy, Debug)]
pub struct SmoothedErfResponse {
    pub center: f32,
    pub kappa_eff: f32,
    pub weight1: f32,
    pub kappa1: f32,
    pub weight2: f32,
    pub kappa2: f32,
}

pub fn smooth_erf_response(model: &ErfResponseModel, sigma: f32) -> SmoothedErfResponse {
    let tau_squared = (model.kappa * sigma) * (model.kappa * sigma);
    let r1 = 1.0 / (1.0 + 2.0 * tau_squared).sqrt();
    let r2 = 1.0 / (1.0 + 0.5 * tau_squared).sqrt();
    SmoothedErfResponse {
        center: model.center,
        kappa_eff: model.kappa * r1,
        weight1: model.gaussian_weights[0] * r1 * r1,
        kappa1: model.kappa * r1,
        weight2: model.gaussian_weights[1] * r2 * r2,
        kappa2: 0.5 * model.kappa * r2,
    }
}

/// Pointwise smoothed response `E[phi(x - Z)]`, `Z ~ N(0, sigma^2)`.
pub fn evaluate_erf_response(model: &ErfResponseModel, x: f32, sigma: f32) -> f32 {
    let smoothed = smooth_erf_response(model, sigma);
    let y = (x - smoothed.center) as f64;
    let value = 0.5 * (1.0 + erf64(smoothed.kappa_eff as f64 * y))
        + smoothed.weight1 as f64 * psi(smoothed.kappa1 as f64 * y)
        + smoothed.weight2 as f64 * psi(smoothed.kappa2 as f64 * y);
    value as f32
}

/// Integral and first moment of the smoothed response along a linear argument:
/// `int_{s0}^{s1} phi_sigma(x0 + x1 s) ds` and the same with weight `s`.
/// Every term reduces to the E/J moment families over the piece.
pub fn integrate_erf_response_linear(
    model: &ErfResponseModel,
    sigma: f32,
    x0: f32,
    x1: f32,
    s0: f32,
    s1: f32,
) -> (f32, f32) {
    if s1 <= s0 {
        return (0.0, 0.0);
    }
    let smoothed = smooth_erf_response(model, sigma);
    let mid = 0.5 * (s0 + s1);
    let half_width = 0.5 * (s1 - s0);
    let y0 = x0 - smoothed.center + x1 * mid;

    let length = s1 - s0;
    let mut integral = 0.5 * length;
    let mut first_moment = 0.5 * length * mid;

    let alpha1 = smoothed.kappa_eff * x1;
    let beta1 = smoothed.kappa_eff * y0;
    let (erf_moments, gaussian_moments) =
        integrate_erf_and_gaussian_powers(alpha1, beta1, half_width);
    integral += 0.5 * erf_moments[0];
    first_moment += 0.5 * (mid * erf_moments[0] + erf_moments[1]);
    // psi(alpha s + beta) = (alpha s + beta) exp(-(alpha s + beta)^2).
    let psi1 = alpha1 * gaussian_moments[1] + beta1 * gaussian_moments[0];
    let psi1_first = alpha1 * gaussian_moments[2] + beta1 * gaussian_moments[1];
    integral += smoothed.weight1 * psi1;
    first_moment += smoothed.weight1 * (mid * psi1 + psi1_first);

    let alpha2 = smoothed.kappa2 * x1;
    let beta2 = smoothed.kappa2 * (x0 - smoothed.center + x1 * mid);
    let (_, gaussian_moments2) = integrate_erf_and_gaussian_powers(alpha2, beta2, half_width);
    let psi2 = alpha2 * gaussian_moments2[1] + beta2 * gaussian_moments2[0];
    let psi2_first = alpha2 * gaussian_moments2[2] + beta2 * gaussian_moments2[1];
    integral += smoothed.weight2 * psi2;
    first_moment += smoothed.weight2 * (mid * psi2 + psi2_first);

    (integral, first_moment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smooth_step::smooth_step;

    const EDGE_LOW: f32 = 0.27;
    const EDGE_HIGH: f32 = 0.33;

    fn smoothstep(low: f64, high: f64, x: f64) -> f64 {
        smooth_step((x - low) / (high - low))
    }

    #[test]
    fn test_fit_tracks_smoothstep_to_the_correction_floor() {
        for (low, high) in [(EDGE_LOW, EDGE_HIGH), (0.1f32, 0.5f32), (0.0, 0.06)] {
            let model = fit_erf_response(low, high);
            let mut max_error = 0.0f64;
            for i in 0..=2000 {
                let x = -0.5 + 1.5 * i as f64 / 2000.0;
                let error = (evaluate_erf_response(&model, x as f32, 0.0) as f64
                    - smoothstep(low as f64, high as f64, x))
                .abs();
                max_error = max_error.max(error);
            }
            assert!(max_error < 1.2e-2, "({low}, {high}): max error {max_error}");
        }
    }

    #[test]
    fn test_fit_saturates_exactly_in_the_tails() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        assert!(evaluate_erf_response(&model, -1.0, 0.0).abs() < 1e-6);
        assert!((evaluate_erf_response(&model, 1.5, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_degenerate_edge_width_is_guarded() {
        let model = fit_erf_response(0.3, 0.3);
        assert!(model.kappa.is_finite());
        assert!(evaluate_erf_response(&model, 0.2, 0.0) < 0.01);
        assert!(evaluate_erf_response(&model, 0.4, 0.0) > 0.99);
    }

    /// The smoothed response must equal the Gaussian average of the unsmoothed
    /// response — the family-wise smoothing algebra against brute quadrature.
    #[test]
    fn test_smoothed_response_matches_gaussian_quadrature() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        for sigma in [0.01f32, 0.05, 0.15, 0.4] {
            for x in [-0.1f32, 0.2, 0.28, 0.3, 0.32, 0.5, 0.9] {
                let steps = 4001;
                let span = 6.0 * sigma as f64;
                let dz = 2.0 * span / (steps - 1) as f64;
                let mut reference = 0.0f64;
                let mut weight_total = 0.0f64;
                for i in 0..steps {
                    let z = -span + i as f64 * dz;
                    let weight = (-(z * z) / (2.0 * (sigma as f64) * (sigma as f64))).exp();
                    reference += weight * evaluate_erf_response(&model, x - z as f32, 0.0) as f64;
                    weight_total += weight;
                }
                reference /= weight_total;
                let value = evaluate_erf_response(&model, x, sigma) as f64;
                assert!(
                    (value - reference).abs() < 2e-3,
                    "sigma={sigma} x={x}: got {value}, expected {reference}"
                );
            }
        }
    }

    #[test]
    fn test_linear_integral_matches_quadrature() {
        let model = fit_erf_response(EDGE_LOW, EDGE_HIGH);
        for sigma in [0.0f32, 0.1] {
            for (x0, x1, s0, s1) in [
                (0.0f32, 1.0f32, -0.4f32, 0.7f32),
                (0.6, -0.9, 0.0, 1.0),
                (0.29, 0.02, -0.1, 0.1),
                (0.31, 0.0, -0.25, 0.25),
                (-2.0, 8.0, 0.0, 0.5),
            ] {
                let (integral, first_moment) =
                    integrate_erf_response_linear(&model, sigma, x0, x1, s0, s1);
                let steps = 40000;
                let ds = (s1 - s0) as f64 / steps as f64;
                let mut reference = 0.0f64;
                let mut reference_first = 0.0f64;
                for i in 0..steps {
                    let s = s0 as f64 + (i as f64 + 0.5) * ds;
                    let value =
                        evaluate_erf_response(&model, (x0 as f64 + x1 as f64 * s) as f32, sigma)
                            as f64;
                    reference += value * ds;
                    reference_first += s * value * ds;
                }
                let scale = (s1 - s0) as f64;
                assert!(
                    (integral as f64 - reference).abs() < 2e-3 * scale.max(1.0),
                    "sigma={sigma} x0={x0} x1={x1}: got {integral}, expected {reference}"
                );
                assert!(
                    (first_moment as f64 - reference_first).abs() < 2e-3 * scale.max(1.0),
                    "sigma={sigma} x0={x0} x1={x1} first moment: got {first_moment}, expected {reference_first}"
                );
            }
        }
    }

    #[test]
    fn test_fit_is_deterministic() {
        let first = fit_erf_response(EDGE_LOW, EDGE_HIGH).pack();
        let second = fit_erf_response(EDGE_LOW, EDGE_HIGH).pack();
        let first_bits: Vec<u32> = first.iter().map(|v| v.to_bits()).collect();
        let second_bits: Vec<u32> = second.iter().map(|v| v.to_bits()).collect();
        assert_eq!(first_bits, second_bits);
    }
}
