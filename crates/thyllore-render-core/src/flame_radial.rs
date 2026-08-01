use crate::flame_shell::FLAME_SHELL_BASE_RADIUS;
use thyllore_math_core::{
    biweight_profile, evaluate_chebyshev, integrate_powers, solve_support_interval, ChebyshevSeries,
};

// Mirror of shaders/include/flame_radial_integral.glsl; the accuracy tests below cover both.
//
// The radial density is the compact-support biweight kernel
//   rho(p) = F(h) * (1 - u^2)^2,  u = |p.xz| / (S * R(h)),  zero for u >= 1.
// Along a ray u^2(s) is a quadratic g(s), so each height band is the exact
// polynomial integral of (F0 + F1 s + F2 s^2) * (1 - g(s))^2 over the interval
// where g(s) <= 1 — power-rule moments only, no tail and no pedestal.

pub const FLAME_RADIAL_BAND_COUNT: usize = 6;
const MIN_DIR_Y: f32 = 1e-4;
const MIN_HEIGHT_SPAN: f32 = 1e-5;
/// Exponent variation across a band below which the constant-exponent moments are used.
const FLAT_EXPONENT: f32 = 2e-2;
/// Maximum support radius in R(h) units, bounded by the shell proxy headroom.
pub const FLAME_RADIAL_RMAX: f32 = crate::flame_shell::FLAME_SHELL_SUPPORT_HEADROOM;
/// Height taper of the radial density.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameRadialTaper {
    pub tip_ratio: f32,
    pub power: f32,
}

impl FlameRadialTaper {
    pub fn from_effect(effect: &crate::flame::FlameEffect) -> Self {
        Self {
            tip_ratio: effect.radius_tip_ratio,
            power: effect.taper_power,
        }
    }
}

/// Radius `R(h)` of the radial density profile, in flame-local units.
pub fn flame_radial_radius_scale(height01: f32, taper: FlameRadialTaper) -> f32 {
    FLAME_SHELL_BASE_RADIUS * (1.0 + (taper.tip_ratio - 1.0) * height01.powf(taper.power))
}

/// Support radius `S` of the biweight profile in R(h) units. The curvature at the
/// axis matches the former Gaussian `exp(-sharpness * u^2)`, so the sharpness lever
/// keeps its direction: larger sharpness narrows the support.
pub fn flame_radial_support_radius(radial_sharpness: f32) -> f32 {
    (2.0 / radial_sharpness.max(1e-3))
        .sqrt()
        .min(FLAME_RADIAL_RMAX)
}

fn support_inv_sq(height01: f32, taper: FlameRadialTaper, radial_sharpness: f32) -> f32 {
    let scale = (flame_radial_support_radius(radial_sharpness)
        * flame_radial_radius_scale(height01, taper))
    .max(1e-4);
    1.0 / (scale * scale)
}

/// Abramowitz-Stegun 7.1.26.
pub fn approximate_erf(x: f32) -> f32 {
    let magnitude = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * magnitude);
    let series = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    let value = 1.0 - series * (-magnitude * magnitude).exp();
    if x < 0.0 {
        -value
    } else {
        value
    }
}

/// Radial density factor (1 - u^2)^2 at a flame-local point, zero outside the support.
pub fn evaluate_radial_density_factor(
    point_local: [f32; 3],
    taper: FlameRadialTaper,
    radial_sharpness: f32,
) -> f32 {
    let height = point_local[1].clamp(0.0, 1.0);
    let radius_squared = point_local[0] * point_local[0] + point_local[2] * point_local[2];
    biweight_profile(support_inv_sq(height, taper, radial_sharpness) * radius_squared)
}

/// Moments `int_{-half}^{half} s^m exp(-(a s^2 + b s + c)) ds` for m = 0, 1, 2.
/// Kept for the plume integral, which stays Gaussian.
pub fn evaluate_gaussian_moments(a: f32, b: f32, c: f32, half_width: f32) -> [f32; 3] {
    if a * half_width * half_width < FLAT_EXPONENT && b.abs() * half_width < FLAT_EXPONENT {
        let flat = (-c).exp();
        return [
            2.0 * half_width * flat,
            0.0,
            (2.0 / 3.0) * half_width * half_width * half_width * flat,
        ];
    }

    let root_a = a.sqrt();
    let center = b / (2.0 * a);
    let peak = (-(c - b * b / (4.0 * a))).exp();
    let moment0 = peak
        * 0.5
        * (std::f32::consts::PI / a).sqrt()
        * (approximate_erf(root_a * (half_width + center))
            - approximate_erf(root_a * (center - half_width)));

    let gauge_hi = (-(a * half_width * half_width + b * half_width + c)).exp();
    let gauge_lo = (-(a * half_width * half_width - b * half_width + c)).exp();
    let moment1 = (gauge_lo - gauge_hi - b * moment0) / (2.0 * a);
    let moment2 = (moment0 - b * moment1 - half_width * (gauge_hi + gauge_lo)) / (2.0 * a);
    [moment0, moment1, moment2]
}

/// Integral and first moment of (f0 + f1 s + f2 s^2) * (1 - g(s))^2 over the part of
/// [-half_width, half_width] inside the support g(s) = a s^2 + b s + c <= 1.
pub fn evaluate_biweight_band(
    a: f32,
    b: f32,
    c: f32,
    f0: f32,
    f1: f32,
    f2: f32,
    half_width: f32,
) -> (f32, f32) {
    let Some((s_lo, s_hi)) = solve_support_interval(a, b, c, -half_width, half_width) else {
        return (0.0, 0.0);
    };

    let m = 1.0 - c;
    let w0 = m * m;
    let w1 = -2.0 * b * m;
    let w2 = b * b - 2.0 * a * m;
    let w3 = 2.0 * a * b;
    let w4 = a * a;

    let e = [
        f0 * w0,
        f0 * w1 + f1 * w0,
        f0 * w2 + f1 * w1 + f2 * w0,
        f0 * w3 + f1 * w2 + f2 * w1,
        f0 * w4 + f1 * w3 + f2 * w2,
        f1 * w4 + f2 * w3,
        f2 * w4,
    ];

    let powers = integrate_powers(s_lo, s_hi);
    let mut integral = 0.0;
    let mut first_moment = 0.0;
    for (n, coefficient) in e.iter().enumerate() {
        integral += coefficient * powers[n];
        first_moment += coefficient * powers[n + 1];
    }
    (integral, first_moment)
}

fn build_height_series(height_coefficients: &[[f32; 4]; 2]) -> ChebyshevSeries {
    ChebyshevSeries::new(
        height_coefficients.iter().flatten().copied().collect(),
        (0.0, 1.0),
    )
}

fn integrate_along_ray(
    origin: [f32; 3],
    direction: [f32; 3],
    t_near: f32,
    t_far: f32,
    inv_sq: f32,
    height: &ChebyshevSeries,
) -> f32 {
    let t_center = 0.5 * (t_near + t_far);
    let point_xz = [
        origin[0] + t_center * direction[0],
        origin[2] + t_center * direction[2],
    ];
    let half_width = 0.5 * (t_far - t_near);
    let a = inv_sq * (direction[0] * direction[0] + direction[2] * direction[2]);
    let b = 2.0 * inv_sq * (point_xz[0] * direction[0] + point_xz[1] * direction[2]);
    let c = inv_sq * (point_xz[0] * point_xz[0] + point_xz[1] * point_xz[1]);
    let height_at_center = (origin[1] + t_center * direction[1]).clamp(0.0, 1.0);
    let falloff = evaluate_chebyshev(height, height_at_center);
    let (integral, _) = evaluate_biweight_band(a, b, c, falloff, 0.0, 0.0, half_width);
    integral.max(0.0)
}

/// Horizontal offset of the ray at a given height, with `q = d.xz / d.y`.
fn ray_point_at_height(origin_local: [f32; 3], q: [f32; 2], height: f32) -> [f32; 2] {
    let rise = height - origin_local[1];
    [origin_local[0] + rise * q[0], origin_local[2] + rise * q[1]]
}

/// Emission integral of `F(h) * (1 - |p.xz|^2 / (S R(h))^2)^2` over a ray segment
/// in flame-local space.
pub fn integrate_radial_emission(
    origin_local: [f32; 3],
    direction_local: [f32; 3],
    t_near: f32,
    t_far: f32,
    height_coefficients: &[[f32; 4]; 2],
    taper: FlameRadialTaper,
    radial_sharpness: f32,
) -> f32 {
    if t_far <= t_near {
        return 0.0;
    }

    let height = build_height_series(height_coefficients);

    let height_near = (origin_local[1] + t_near * direction_local[1]).clamp(0.0, 1.0);
    let height_far = (origin_local[1] + t_far * direction_local[1]).clamp(0.0, 1.0);
    let mut height_lo = height_near.min(height_far);
    let mut height_hi = height_near.max(height_far);
    if direction_local[1].abs() < MIN_DIR_Y || height_hi - height_lo < MIN_HEIGHT_SPAN {
        let mid_height =
            (origin_local[1] + 0.5 * (t_near + t_far) * direction_local[1]).clamp(0.0, 1.0);
        return integrate_along_ray(
            origin_local,
            direction_local,
            t_near,
            t_far,
            support_inv_sq(mid_height, taper, radial_sharpness),
            &height,
        );
    }

    // Only the slope is carried: monomial coefficients in h would grow as 1/d.y^2 and cancel away.
    let q = [
        direction_local[0] / direction_local[1],
        direction_local[2] / direction_local[1],
    ];
    let quadratic = q[0] * q[0] + q[1] * q[1];

    // Trim to the widest support across heights so grazing rays do not spend bands on empty range.
    let widest_radius = flame_radial_support_radius(radial_sharpness)
        * flame_radial_radius_scale(0.0, taper)
            .max(flame_radial_radius_scale(1.0, taper))
            .max(1e-4);
    if quadratic > 1e-12 {
        let support = widest_radius / quadratic.sqrt();
        let closest_approach_height =
            origin_local[1] - (origin_local[0] * q[0] + origin_local[2] * q[1]) / quadratic;
        height_lo = height_lo.max(closest_approach_height - support);
        height_hi = height_hi.min(closest_approach_height + support);
        if height_hi <= height_lo {
            return 0.0;
        }
    }

    let band_width = (height_hi - height_lo) / FLAME_RADIAL_BAND_COUNT as f32;
    let half_width = 0.5 * band_width;
    let mut total = 0.0;
    let mut falloff_lo = evaluate_chebyshev(&height, height_lo);
    for band in 0..FLAME_RADIAL_BAND_COUNT {
        let center = height_lo + (band as f32 + 0.5) * band_width;
        let falloff_mid = evaluate_chebyshev(&height, center);
        let falloff_hi = evaluate_chebyshev(&height, center + half_width);

        let inv_sq = support_inv_sq(center, taper, radial_sharpness);
        let point_xz = ray_point_at_height(origin_local, q, center);
        let slope = (falloff_hi - falloff_lo) / band_width;
        let curvature =
            2.0 * (falloff_hi + falloff_lo - 2.0 * falloff_mid) / (band_width * band_width);
        let (integral, _) = evaluate_biweight_band(
            inv_sq * quadratic,
            2.0 * inv_sq * (point_xz[0] * q[0] + point_xz[1] * q[1]),
            inv_sq * (point_xz[0] * point_xz[0] + point_xz[1] * point_xz[1]),
            falloff_mid,
            slope,
            curvature,
            half_width,
        );
        total += integral;

        falloff_lo = falloff_hi;
    }
    (total / direction_local[1].abs()).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flame::{FlameCoefficients, FlameEffect};

    const SHARPNESS: f32 = 4.0;

    fn default_taper() -> FlameRadialTaper {
        FlameRadialTaper::from_effect(&FlameEffect::default())
    }

    fn reference_integral(
        origin: [f32; 3],
        direction: [f32; 3],
        t_near: f32,
        t_far: f32,
        coefficients: &FlameCoefficients,
        taper: FlameRadialTaper,
        sharpness: f32,
    ) -> f64 {
        let height = build_height_series(&coefficients.height);
        let steps = 40000;
        let dt = (t_far - t_near) as f64 / steps as f64;
        (0..steps)
            .map(|i| {
                let t = t_near as f64 + (i as f64 + 0.5) * dt;
                let p = [
                    origin[0] as f64 + t * direction[0] as f64,
                    origin[1] as f64 + t * direction[1] as f64,
                    origin[2] as f64 + t * direction[2] as f64,
                ];
                let h = p[1].clamp(0.0, 1.0) as f32;
                let radius_squared = p[0] * p[0] + p[2] * p[2];
                let u_squared = support_inv_sq(h, taper, sharpness) as f64 * radius_squared;
                let density = (1.0 - u_squared).max(0.0).powi(2);
                evaluate_chebyshev(&height, h) as f64 * density
            })
            .sum::<f64>()
            * dt
    }

    fn normalize_direction(d: [f32; 3]) -> [f32; 3] {
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        [d[0] / len, d[1] / len, d[2] / len]
    }

    /// Segment of the ray inside the cylinder proxy, matching what the shader hands the integral.
    fn clip_to_shell_proxy(origin: [f32; 3], direction: [f32; 3]) -> Option<(f32, f32)> {
        let radius = crate::flame_shell::flame_shell_outer_radius(0.0, 1.0);
        let a = direction[0] * direction[0] + direction[2] * direction[2];
        let b = 2.0 * (origin[0] * direction[0] + origin[2] * direction[2]);
        let c = origin[0] * origin[0] + origin[2] * origin[2] - radius * radius;
        let discriminant = b * b - 4.0 * a * c;
        if a <= 0.0 || discriminant <= 0.0 {
            return None;
        }

        let root = discriminant.sqrt();
        let mut t_near = ((-b - root) / (2.0 * a)).max(0.0);
        let mut t_far = (-b + root) / (2.0 * a);
        if direction[1].abs() > 1e-9 {
            let slab_0 = -origin[1] / direction[1];
            let slab_1 = (1.0 - origin[1]) / direction[1];
            t_near = t_near.max(slab_0.min(slab_1));
            t_far = t_far.min(slab_0.max(slab_1));
        } else if origin[1] < 0.0 || origin[1] > 1.0 {
            return None;
        }
        (t_far > t_near).then_some((t_near, t_far))
    }

    /// Rays covering side, tilted and near-vertical views of the unit-local flame.
    fn representative_rays() -> Vec<([f32; 3], [f32; 3], f32, f32)> {
        let mut rays = Vec::new();
        for origin in [
            [0.0f32, 0.0, 6.7],
            [0.0, 0.5, 6.7],
            [4.7, 1.0, 4.7],
            [0.8, 2.5, 0.8],
            [5.0, 0.12, 0.5],
        ] {
            for target_x in [-0.45f32, -0.2, 0.0, 0.2, 0.45] {
                for target_y in [0.05f32, 0.3, 0.6, 0.95] {
                    let direction = normalize_direction([
                        target_x - origin[0],
                        target_y - origin[1],
                        -origin[2],
                    ]);
                    // Enter and exit the unit-local y slab, which is what the shell clamp yields.
                    if direction[1].abs() < 1e-3 {
                        continue;
                    }
                    let t_slab_0 = -origin[1] / direction[1];
                    let t_slab_1 = (1.0 - origin[1]) / direction[1];
                    let t_near = t_slab_0.min(t_slab_1).max(0.0);
                    let t_far = t_slab_0.max(t_slab_1);
                    if t_far <= t_near {
                        continue;
                    }
                    rays.push((origin, direction, t_near, t_far));
                }
            }
        }
        rays
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
    fn test_gaussian_moments_match_quadrature() {
        for (a, b, c, half) in [
            (0.0f32, 0.0f32, 0.0f32, 0.125f32),
            (0.02, 0.01, 0.3, 0.125),
            (4.0, -1.0, 0.5, 0.125),
            (600.0, -30.0, 2.0, 0.05),
        ] {
            let moments = evaluate_gaussian_moments(a, b, c, half);
            let steps = 20000;
            let ds = 2.0 * half as f64 / steps as f64;
            let mut reference = [0.0f64; 3];
            for i in 0..steps {
                let s = -half as f64 + (i as f64 + 0.5) * ds;
                let g = (-(a as f64 * s * s + b as f64 * s + c as f64)).exp();
                reference[0] += g * ds;
                reference[1] += s * g * ds;
                reference[2] += s * s * g * ds;
            }
            for m in 0..3 {
                // Each moment carries a factor of half^m, so scale the bound the same way.
                let tolerance = 1e-3 * reference[0].abs() * (half as f64).powi(m as i32).max(1e-6);
                assert!(
                    (moments[m] as f64 - reference[m]).abs() < tolerance,
                    "moment {m} for (a={a}, b={b}, c={c}): got {}, expected {}",
                    moments[m],
                    reference[m]
                );
            }
        }
    }

    #[test]
    fn test_biweight_band_matches_quadrature() {
        for (a, b, c, f0, f1, f2, half) in [
            (0.0f32, 0.0f32, 0.3f32, 1.0f32, 0.0f32, 0.0f32, 0.125f32),
            (0.5, 0.2, 0.4, 0.8, 0.3, -0.6, 0.25),
            (40.0, -6.0, 0.9, 1.0, -0.5, 2.0, 0.125),
            (900.0, -60.0, 1.5, 0.7, 0.1, 0.0, 0.05),
        ] {
            let (integral, first_moment) = evaluate_biweight_band(a, b, c, f0, f1, f2, half);
            let steps = 200000;
            let ds = 2.0 * half as f64 / steps as f64;
            let mut reference = 0.0f64;
            let mut reference_first = 0.0f64;
            for i in 0..steps {
                let s = -half as f64 + (i as f64 + 0.5) * ds;
                let g = a as f64 * s * s + b as f64 * s + c as f64;
                let density = (1.0 - g).max(0.0).powi(2) * if g <= 1.0 { 1.0 } else { 0.0 };
                let weight = (f0 as f64 + f1 as f64 * s + f2 as f64 * s * s) * density;
                reference += weight * ds;
                reference_first += s * weight * ds;
            }
            assert!(
                (integral as f64 - reference).abs() < 1e-4 * reference.abs().max(1e-4),
                "integral for (a={a}, b={b}, c={c}): got {integral}, expected {reference}"
            );
            assert!(
                (first_moment as f64 - reference_first).abs()
                    < 1e-4 * reference.abs().max(1e-4) * half as f64,
                "first moment for (a={a}, b={b}, c={c}): got {first_moment}, expected {reference_first}"
            );
        }
    }

    #[test]
    fn test_integrate_radial_emission_matches_reference_within_one_percent() {
        let coefficients = FlameCoefficients::default();
        let taper = default_taper();
        let rays = representative_rays();
        assert!(rays.len() > 50, "ray set too small: {}", rays.len());

        let references: Vec<f64> = rays
            .iter()
            .map(|(o, d, near, far)| {
                reference_integral(*o, *d, *near, *far, &coefficients, taper, SHARPNESS)
            })
            .collect();
        let peak = references.iter().fold(0.0f64, |m, r| m.max(r.abs()));

        for ((origin, direction, t_near, t_far), reference) in rays.iter().zip(&references) {
            let value = integrate_radial_emission(
                *origin,
                *direction,
                *t_near,
                *t_far,
                &coefficients.height,
                taper,
                SHARPNESS,
            ) as f64;
            assert!(
                (value - reference).abs() < 0.01 * peak,
                "ray o={origin:?} d={direction:?}: got {value}, expected {reference}"
            );
        }
    }

    /// Worst conditioning of the height parameterization, and only above MIN_DIR_Y does the
    /// along-ray fallback stop covering it. A camera above the flame base is what triggers it.
    #[test]
    fn test_near_horizontal_rays_match_reference() {
        let coefficients = FlameCoefficients::default();
        let taper = default_taper();
        let mut checked = 0;

        for camera_height in [0.0f32, 0.5, 0.9] {
            for direction_y in [3e-2f32, 1e-2, 3e-3, 1e-3, 3e-4, 1.2e-4] {
                for sign in [1.0f32, -1.0] {
                    let origin = [0.12, camera_height, 6.7];
                    let direction = normalize_direction([0.0, sign * direction_y, -1.0]);
                    let Some((t_near, t_far)) = clip_to_shell_proxy(origin, direction) else {
                        continue;
                    };

                    let reference = reference_integral(
                        origin,
                        direction,
                        t_near,
                        t_far,
                        &coefficients,
                        taper,
                        SHARPNESS,
                    );
                    let value = integrate_radial_emission(
                        origin,
                        direction,
                        t_near,
                        t_far,
                        &coefficients.height,
                        taper,
                        SHARPNESS,
                    ) as f64;
                    assert!(
                        (value - reference).abs() < 0.01 * reference.abs().max(1e-4),
                        "camera y={camera_height} d.y={}: got {value}, expected {reference}",
                        direction[1]
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 20, "ray set too small: {checked}");
    }

    /// Grazing rays that clip the support edge: the interval solve must stay accurate
    /// where the discriminant approaches zero instead of flickering to zero or spiking.
    #[test]
    fn test_grazing_rays_near_support_edge_match_reference() {
        let coefficients = FlameCoefficients::default();
        let taper = default_taper();
        let support_edge =
            flame_radial_support_radius(SHARPNESS) * flame_radial_radius_scale(0.0, taper);

        let mut checked = 0;
        for offset in [-2e-3f32, -1e-4, 0.0, 1e-4, 2e-3] {
            let x = support_edge + offset;
            let origin = [x, 0.2, 6.7];
            let direction = normalize_direction([0.0, 0.05, -1.0]);
            let Some((t_near, t_far)) = clip_to_shell_proxy(origin, direction) else {
                continue;
            };
            let reference = reference_integral(
                origin,
                direction,
                t_near,
                t_far,
                &coefficients,
                taper,
                SHARPNESS,
            );
            let value = integrate_radial_emission(
                origin,
                direction,
                t_near,
                t_far,
                &coefficients.height,
                taper,
                SHARPNESS,
            ) as f64;
            assert!(
                (value - reference).abs() < 0.01 * reference.abs().max(1e-4),
                "grazing x={x}: got {value}, expected {reference}"
            );
            checked += 1;
        }
        assert!(checked >= 5, "ray set too small: {checked}");
    }

    #[test]
    fn test_axis_ray_equals_plain_height_integral() {
        let coefficients = FlameCoefficients::default();
        let height = build_height_series(&coefficients.height);
        let steps = 20000;
        let plain: f64 = (0..steps)
            .map(|i| evaluate_chebyshev(&height, (i as f32 + 0.5) / steps as f32) as f64)
            .sum::<f64>()
            / steps as f64;

        // On-axis: u^2 = 0, so the biweight profile is exactly 1.
        let value = integrate_radial_emission(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
            1.0,
            &coefficients.height,
            default_taper(),
            SHARPNESS,
        ) as f64;
        assert!(
            (value - plain).abs() < 0.005 * plain,
            "on-axis ray got {value}, expected {plain}"
        );
    }

    #[test]
    fn test_ray_outside_support_is_exactly_zero() {
        let coefficients = FlameCoefficients::default();
        let taper = default_taper();
        let support_edge =
            flame_radial_support_radius(SHARPNESS) * flame_radial_radius_scale(0.0, taper);
        for x in [support_edge + 1e-3, support_edge * 2.0, 3.0] {
            let value = integrate_radial_emission(
                [x, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                0.0,
                1.0,
                &coefficients.height,
                taper,
                SHARPNESS,
            );
            assert_eq!(value, 0.0, "x={x} lies outside the support");
        }
    }

    #[test]
    fn test_empty_segment_returns_zero() {
        let coefficients = FlameCoefficients::default();
        for (near, far) in [(0.5f32, 0.5f32), (1.0, 0.2)] {
            let value = integrate_radial_emission(
                [0.0, 0.0, 4.0],
                [0.0, 0.2, -1.0],
                near,
                far,
                &coefficients.height,
                default_taper(),
                SHARPNESS,
            );
            assert_eq!(value, 0.0);
        }
    }

    #[test]
    fn test_radial_density_factor_falls_off_with_radius() {
        let taper = default_taper();
        // On the axis the biweight profile is exactly 1.
        assert!(
            (evaluate_radial_density_factor([0.0, 0.5, 0.0], taper, SHARPNESS) - 1.0).abs() < 1e-6
        );
        let support_edge =
            flame_radial_support_radius(SHARPNESS) * flame_radial_radius_scale(0.5, taper);
        let inner =
            evaluate_radial_density_factor([0.3 * support_edge, 0.5, 0.0], taper, SHARPNESS);
        let outer =
            evaluate_radial_density_factor([0.7 * support_edge, 0.5, 0.0], taper, SHARPNESS);
        assert!(inner > outer && outer > 0.0);
        // Outside the support the density is exactly zero.
        assert_eq!(
            evaluate_radial_density_factor([support_edge + 1e-3, 0.5, 0.0], taper, SHARPNESS),
            0.0
        );
    }

    #[test]
    fn test_support_radius_stays_inside_the_shell_proxy() {
        let taper = default_taper();
        for sharpness in [0.5f32, 1.0, 3.0, 4.0, 6.0, 8.0, 16.0] {
            for step in 0..=20 {
                let height = step as f32 / 20.0;
                let density_edge = flame_radial_support_radius(sharpness)
                    * flame_radial_radius_scale(height, taper);
                let proxy = crate::flame_shell::flame_shell_outer_radius(height, 1.0);
                assert!(
                    density_edge <= proxy,
                    "k={sharpness} h={height}: support reaches {density_edge} but the proxy cuts at {proxy}"
                );
            }
        }
    }

    #[test]
    fn test_taper_narrows_the_profile_with_height() {
        let taper = default_taper();
        let base = flame_radial_radius_scale(0.0, taper);
        let tip = flame_radial_radius_scale(1.0, taper);
        assert!((base - FLAME_SHELL_BASE_RADIUS).abs() < 1e-6);
        assert!(tip < base * 0.2, "tip {tip} should follow radius_tip_ratio");
    }

    #[test]
    fn test_integrate_radial_emission_is_deterministic() {
        let effect = FlameEffect::default();
        let first = integrate_radial_emission(
            [0.3, 0.1, 4.0],
            normalize_direction([-0.3, 0.4, -4.0]),
            0.0,
            4.2,
            &effect.coefficients.height,
            FlameRadialTaper::from_effect(&effect),
            effect.radial_sharpness,
        );
        let second = integrate_radial_emission(
            [0.3, 0.1, 4.0],
            normalize_direction([-0.3, 0.4, -4.0]),
            0.0,
            4.2,
            &effect.coefficients.height,
            FlameRadialTaper::from_effect(&effect),
            effect.radial_sharpness,
        );
        assert_eq!(first.to_bits(), second.to_bits());
    }
}
