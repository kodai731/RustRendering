//! Compact-support radial profile primitives for the analytic flame integrals.
//!
//! The density profile is the biweight kernel `(1 - u^2)^2` on `u in [0, 1]`,
//! exactly zero outside. Along a ray the squared normalized radius is a
//! quadratic `g(s) = a s^2 + b s + c`, so the support is the interval where
//! `g(s) <= 1` and every band integral reduces to power-rule moments.

/// Biweight kernel `(1 - u^2)^2` for `u^2 <= 1`, exactly zero outside.
pub fn biweight_profile(u_squared: f32) -> f32 {
    let inside = (1.0 - u_squared).max(0.0);
    inside * inside
}

const LINEAR_COEFFICIENT_EPSILON: f32 = 1e-12;

/// Interval where `a s^2 + b s + c <= 1` clipped to `[s_min, s_max]`, or `None` when empty.
///
/// Requires `a >= 0` (squared-distance quadratics only). The roots are taken in the
/// cancellation-free Citardauq form so grazing rays (discriminant near zero) and
/// near-linear quadratics (`a` tiny against `b`) keep full precision.
pub fn solve_support_interval(
    a: f32,
    b: f32,
    c: f32,
    s_min: f32,
    s_max: f32,
) -> Option<(f32, f32)> {
    if s_max <= s_min {
        return None;
    }
    if a < LINEAR_COEFFICIENT_EPSILON {
        // Squared distance is constant over the segment scale: inside iff c <= 1.
        return (c <= 1.0).then_some((s_min, s_max));
    }

    let discriminant = b * b - 4.0 * a * (c - 1.0);
    if discriminant <= 0.0 {
        // The upward parabola never dips to 1 (or only touches it): empty support.
        return None;
    }

    let root = discriminant.sqrt();
    let q = -0.5 * (b + root.copysign(if b == 0.0 { 1.0 } else { b }));
    let s_first = q / a;
    let s_second = (c - 1.0) / q;
    let lo = s_first.min(s_second).max(s_min);
    let hi = s_first.max(s_second).min(s_max);
    (hi > lo).then_some((lo, hi))
}

/// Power-rule moments `int_{s_lo}^{s_hi} s^n ds` for `n = 0..=7`.
pub fn integrate_powers(s_lo: f32, s_hi: f32) -> [f32; 8] {
    let mut moments = [0.0f32; 8];
    let mut power_lo = 1.0;
    let mut power_hi = 1.0;
    for (n, moment) in moments.iter_mut().enumerate() {
        power_lo *= s_lo;
        power_hi *= s_hi;
        *moment = (power_hi - power_lo) / (n as f32 + 1.0);
    }
    moments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_interval(a: f64, b: f64, c: f64, s_min: f64, s_max: f64) -> Option<(f64, f64)> {
        if a < LINEAR_COEFFICIENT_EPSILON as f64 {
            return (c <= 1.0).then_some((s_min, s_max));
        }
        let discriminant = b * b - 4.0 * a * (c - 1.0);
        if discriminant <= 0.0 {
            return None;
        }
        let root = discriminant.sqrt();
        let lo = ((-b - root) / (2.0 * a)).max(s_min);
        let hi = ((-b + root) / (2.0 * a)).min(s_max);
        (hi > lo).then_some((lo, hi))
    }

    #[test]
    fn test_biweight_profile_shape() {
        assert_eq!(biweight_profile(0.0), 1.0);
        assert_eq!(biweight_profile(1.0), 0.0);
        assert_eq!(biweight_profile(4.0), 0.0);
        let mid = biweight_profile(0.5);
        assert!((mid - 0.25).abs() < 1e-7);
    }

    #[test]
    fn test_interval_matches_f64_reference_across_scales() {
        let mut checked = 0;
        for a_exp in [-8i32, -4, -1, 0, 2, 5] {
            let a = 10.0f32.powi(a_exp);
            for b in [-30.0f32, -1.0, 0.0, 0.7, 12.0] {
                for c in [-0.5f32, 0.0, 0.9, 1.1, 40.0] {
                    let got = solve_support_interval(a, b, c, -2.0, 2.0);
                    let expected = reference_interval(a as f64, b as f64, c as f64, -2.0, 2.0);
                    match (got, expected) {
                        (None, None) => {}
                        (Some((lo, hi)), Some((elo, ehi))) => {
                            let scale = (ehi - elo).max(1e-3);
                            assert!(
                                ((lo as f64 - elo).abs() < 1e-4 * scale.max(elo.abs()))
                                    && ((hi as f64 - ehi).abs() < 1e-4 * scale.max(ehi.abs())),
                                "a={a} b={b} c={c}: got ({lo}, {hi}), expected ({elo}, {ehi})"
                            );
                        }
                        _ => panic!("a={a} b={b} c={c}: got {got:?}, expected {expected:?}"),
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 100);
    }

    /// Grazing rays: the discriminant crosses zero. Slightly outside must be empty,
    /// slightly inside must return a tiny interval centered on the tangent point.
    #[test]
    fn test_grazing_ray_near_zero_discriminant() {
        let a = 250.0f32;
        let b = -40.0f32;
        // Tangency at c = 1 + b^2 / (4a) = 2.6; center s = -b / (2a) = 0.08.
        let c_tangent = 1.0 + b * b / (4.0 * a);

        assert_eq!(
            solve_support_interval(a, b, c_tangent + 1e-3, -1.0, 1.0),
            None
        );

        let (lo, hi) = solve_support_interval(a, b, c_tangent - 1e-3, -1.0, 1.0)
            .expect("slightly inside tangency must intersect");
        let center = 0.5 * (lo + hi);
        assert!(
            (center - 0.08).abs() < 1e-4,
            "center {center} should sit at 0.08"
        );
        let expected_half = (1e-3f32 / a).sqrt();
        let half = 0.5 * (hi - lo);
        assert!(
            (half - expected_half).abs() < 0.05 * expected_half,
            "half-width {half}, expected {expected_half}"
        );
    }

    /// Catastrophic cancellation case for the textbook formula: b^2 >> 4a(c-1)
    /// makes one root computed as (-b + root) lose all digits. Citardauq keeps both.
    #[test]
    fn test_near_linear_quadratic_keeps_precision() {
        let a = 1e-6f32;
        let b = 2.0f32;
        let c = 0.5f32;
        let (lo, hi) = solve_support_interval(a, b, c, -1e9, 1e9).expect("must intersect");
        // Exact small root of a s^2 + b s + (c-1) = 0 is close to -(c-1)/b = 0.25.
        assert!((hi - 0.25).abs() < 1e-4, "small root {hi} should be 0.25");
        // Large root is near -b/a = -2e6.
        assert!((lo + 2e6).abs() < 1.0, "large root {lo} should be -2e6");
    }

    #[test]
    fn test_constant_quadratic_uses_membership() {
        assert_eq!(
            solve_support_interval(0.0, 0.0, 0.5, -3.0, 4.0),
            Some((-3.0, 4.0))
        );
        assert_eq!(solve_support_interval(0.0, 0.0, 1.5, -3.0, 4.0), None);
    }

    #[test]
    fn test_interval_outside_clip_window_is_empty() {
        // Support is [-1, 1] for a=1, b=0, c=0; a window beyond it must be empty.
        assert_eq!(solve_support_interval(1.0, 0.0, 0.0, 2.0, 3.0), None);
        assert_eq!(solve_support_interval(1.0, 0.0, 0.0, 3.0, 2.0), None);
    }

    #[test]
    fn test_integrate_powers_matches_quadrature() {
        let (s_lo, s_hi) = (-0.35f32, 0.6f32);
        let moments = integrate_powers(s_lo, s_hi);
        let steps = 200000;
        let ds = (s_hi - s_lo) as f64 / steps as f64;
        for (n, moment) in moments.iter().enumerate() {
            let reference: f64 = (0..steps)
                .map(|i| {
                    let s = s_lo as f64 + (i as f64 + 0.5) * ds;
                    s.powi(n as i32) * ds
                })
                .sum();
            assert!(
                (*moment as f64 - reference).abs() < 1e-6,
                "moment {n}: got {moment}, expected {reference}"
            );
        }
    }
}
