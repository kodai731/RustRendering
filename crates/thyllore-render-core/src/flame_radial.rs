use crate::flame_shell::FLAME_SHELL_BASE_RADIUS;
use thyllore_math_core::{evaluate_chebyshev, ChebyshevSeries};

// Mirror of shaders/include/flame_radial_integral.glsl; the accuracy tests below cover both.

pub const FLAME_RADIAL_BAND_COUNT: usize = 6;
const MIN_DIR_Y: f32 = 1e-4;
const MIN_HEIGHT_SPAN: f32 = 1e-5;
/// Exponent variation across a band below which the constant-exponent moments are used.
const FLAT_EXPONENT: f32 = 2e-2;
const SUPPORT_SIGMA: f32 = 4.0;
/// Maximum radius of the flame support (pedestal subtraction).
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

/// Gaussian radius `R(h)` of the radial density, in flame-local units.
pub fn flame_radial_gaussian_scale(height01: f32, taper: FlameRadialTaper) -> f32 {
    FLAME_SHELL_BASE_RADIUS * (1.0 + (taper.tip_ratio - 1.0) * height01.powf(taper.power))
}

fn exponent_scale(height01: f32, taper: FlameRadialTaper, radial_sharpness: f32) -> f32 {
    let scale = flame_radial_gaussian_scale(height01, taper).max(1e-4);
    radial_sharpness / (scale * scale)
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

/// Radial density factor max(0, exp(-E) - exp(-Emax)) at a flame-local point.
pub fn evaluate_radial_density_factor(
    point_local: [f32; 3],
    taper: FlameRadialTaper,
    radial_sharpness: f32,
) -> f32 {
    let height = point_local[1].clamp(0.0, 1.0);
    let radius_squared = point_local[0] * point_local[0] + point_local[2] * point_local[2];
    let exponent = exponent_scale(height, taper, radial_sharpness) * radius_squared;
    let emax = radial_sharpness * FLAME_RADIAL_RMAX * FLAME_RADIAL_RMAX;
    ((-exponent).exp() - (-emax).exp()).max(0.0)
}

/// Moments `int_{-half}^{half} s^m exp(-(a s^2 + b s + c)) ds` for m = 0, 1, 2.
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
    sharpness: f32,
    raw_radial_sharpness: f32,
    height: &ChebyshevSeries,
) -> f32 {
    let t_center = 0.5 * (t_near + t_far);
    let point_xz = [
        origin[0] + t_center * direction[0],
        origin[2] + t_center * direction[2],
    ];
    let half_width = 0.5 * (t_far - t_near);
    let moments = evaluate_gaussian_moments(
        sharpness * (direction[0] * direction[0] + direction[2] * direction[2]),
        2.0 * sharpness * (point_xz[0] * direction[0] + point_xz[1] * direction[2]),
        sharpness * (point_xz[0] * point_xz[0] + point_xz[1] * point_xz[1]),
        half_width,
    );
    // Pedestal subtraction: subtract exp(-Emax) * integral over interval I where E(s) <= Emax.
    let emax = raw_radial_sharpness * FLAME_RADIAL_RMAX * FLAME_RADIAL_RMAX;
    let a = sharpness * (direction[0] * direction[0] + direction[2] * direction[2]);
    let b = 2.0 * sharpness * (point_xz[0] * direction[0] + point_xz[1] * direction[2]);
    let c = sharpness * (point_xz[0] * point_xz[0] + point_xz[1] * point_xz[1]);
    let pedestal = pedestal_interval_moments(a, b, c, emax, half_width);
    let height_at_center = (origin[1] + t_center * direction[1]).clamp(0.0, 1.0);
    let emission =
        evaluate_chebyshev(height, height_at_center) * (moments[0] - (-emax).exp() * pedestal[0]);
    emission.max(0.0)
}

/// Pedestal interval moments: [int_I(1), int_I(s), int_I(s^2)] over the interval I where
/// a*s^2 + b*s + c <= emax, clipped to [-half_width, half_width].
/// If discriminant < 0 (ray always inside support), I is the entire band.
/// If ray is outside support (no roots within band), returns zero vector.
fn pedestal_interval_moments(a: f32, b: f32, c: f32, emax: f32, half_width: f32) -> [f32; 3] {
    // If a is near zero, treat as linear: b*s + c <= emax => s <= (emax - c)/b.
    if a < 1e-12 {
        // Near-flat exponent: if center is inside support, use full band; else empty.
        if c > emax {
            return [0.0, 0.0, 0.0];
        }
        // Full band [-half_width, half_width]: I0=2h, I1=0, I2=(2/3)*h^3
        return [
            2.0 * half_width,
            0.0,
            (2.0 / 3.0) * half_width * half_width * half_width,
        ];
    }

    let discriminant = b * b - 4.0 * a * (c - emax);
    if discriminant < 0.0 {
        // Ray always inside support — I is the entire band [-half_width, half_width].
        return [
            2.0 * half_width,
            0.0,
            (2.0 / 3.0) * half_width * half_width * half_width,
        ];
    }

    let root = discriminant.sqrt();
    let mut s_lo = (-b - root) / (2.0 * a);
    let mut s_hi = (-b + root) / (2.0 * a);
    // Clip to [-half_width, half_width]
    s_lo = s_lo.max(-half_width);
    s_hi = s_hi.min(half_width);
    if s_hi <= s_lo {
        return [0.0, 0.0, 0.0];
    }
    // Polynomial moments over [s_lo, s_hi]: I0 = s_hi - s_lo, I1 = 0.5*(s_hi^2 - s_lo^2), I2 = (1/3)*(s_hi^3 - s_lo^3)
    [
        s_hi - s_lo,
        0.5 * (s_hi * s_hi - s_lo * s_lo),
        (1.0 / 3.0) * (s_hi * s_hi * s_hi - s_lo * s_lo * s_lo),
    ]
}

/// Horizontal offset of the ray at a given height, with `q = d.xz / d.y`.
fn ray_point_at_height(origin_local: [f32; 3], q: [f32; 2], height: f32) -> [f32; 2] {
    let rise = height - origin_local[1];
    [origin_local[0] + rise * q[0], origin_local[2] + rise * q[1]]
}

/// Emission integral of `F(h) * exp(-k * |p.xz|^2 / R(h)^2)` over a ray segment in flame-local space.
pub fn integrate_radial_emission(
    origin_local: [f32; 3],
    direction_local: [f32; 3],
    t_near: f32,
    t_far: f32,
    height_coefficients: &[[f32; 4]; 2],
    taper: FlameRadialTaper,
    radial_sharpness: f32,
) -> f32 {
    integrate_radial_emission_with_rmax(
        origin_local,
        direction_local,
        t_near,
        t_far,
        height_coefficients,
        taper,
        radial_sharpness,
        FLAME_RADIAL_RMAX,
    )
}

/// Emission integral with an explicit pedestal radius `r_max`.
fn integrate_radial_emission_with_rmax(
    origin_local: [f32; 3],
    direction_local: [f32; 3],
    t_near: f32,
    t_far: f32,
    height_coefficients: &[[f32; 4]; 2],
    taper: FlameRadialTaper,
    radial_sharpness: f32,
    r_max: f32,
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
            exponent_scale(mid_height, taper, radial_sharpness),
            radial_sharpness,
            &height,
        );
    }

    // Only the slope is carried: monomial coefficients in h would grow as 1/d.y^2 and cancel away.
    let q = [
        direction_local[0] / direction_local[1],
        direction_local[2] / direction_local[1],
    ];
    let quadratic = q[0] * q[0] + q[1] * q[1];

    // Trim to the Gaussian support so grazing rays do not spend bands on empty range.
    let base_scale = exponent_scale(0.0, taper, radial_sharpness);
    if quadratic * base_scale > 1e-12 {
        let support = SUPPORT_SIGMA / (quadratic * base_scale).sqrt();
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

        let k = exponent_scale(center, taper, radial_sharpness);
        let point_xz = ray_point_at_height(origin_local, q, center);
        let moments = evaluate_gaussian_moments(
            k * quadratic,
            2.0 * k * (point_xz[0] * q[0] + point_xz[1] * q[1]),
            k * (point_xz[0] * point_xz[0] + point_xz[1] * point_xz[1]),
            half_width,
        );
        let slope = (falloff_hi - falloff_lo) / band_width;
        let curvature =
            2.0 * (falloff_hi + falloff_lo - 2.0 * falloff_mid) / (band_width * band_width);

        // Pedestal subtraction: subtract exp(-Emax) * [falloff_mid*I0 + slope*I1 + curvature*I2]
        // where I_m = int_I s^m ds over the interval I where a*s^2+b*s+c <= Emax, clipped to [-half_width,half_width].
        let emax = radial_sharpness * r_max * r_max;
        let a = k * quadratic;
        let b = 2.0 * k * (point_xz[0] * q[0] + point_xz[1] * q[1]);
        let c = k * (point_xz[0] * point_xz[0] + point_xz[1] * point_xz[1]);
        let pedestal = pedestal_interval_moments(a, b, c, emax, half_width);
        total += falloff_mid * moments[0] + slope * moments[1] + curvature * moments[2]
            - (-emax).exp()
                * (falloff_mid * pedestal[0] + slope * pedestal[1] + curvature * pedestal[2]);

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
        let emax = sharpness as f64 * FLAME_RADIAL_RMAX as f64 * FLAME_RADIAL_RMAX as f64;
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
                let scale = exponent_scale(h, taper, sharpness) as f64;
                let exponent = scale * radius_squared;
                let density = (-exponent).exp() - (-emax).exp();
                evaluate_chebyshev(&height, h) as f64 * density.max(0.0)
            })
            .sum::<f64>()
            * dt
    }

    /// Same quadrature without the pedestal, which is the limit the integral converges to as r_max grows.
    fn pure_gaussian_reference(
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
                let scale = exponent_scale(h, taper, sharpness) as f64;
                evaluate_chebyshev(&height, h) as f64 * (-scale * radius_squared).exp()
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
        let radius = crate::flame_shell::flame_shell_outer_radius(0.0);
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

    #[test]
    fn test_axis_ray_equals_plain_height_integral() {
        let coefficients = FlameCoefficients::default();
        let height = build_height_series(&coefficients.height);
        let steps = 20000;
        let plain: f64 = (0..steps)
            .map(|i| evaluate_chebyshev(&height, (i as f32 + 0.5) / steps as f32) as f64)
            .sum::<f64>()
            / steps as f64;

        // On-axis: radius_squared = 0, so density = exp(0) - exp(-Emax) = 1 - exp(-Emax)
        let emax = SHARPNESS as f64 * FLAME_RADIAL_RMAX as f64 * FLAME_RADIAL_RMAX as f64;
        let expected = (1.0 - (-emax).exp()) * plain;

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
            (value - expected).abs() < 0.005 * expected,
            "on-axis ray got {value}, expected {expected}"
        );
    }

    #[test]
    fn test_far_from_axis_ray_is_negligible() {
        let coefficients = FlameCoefficients::default();
        let value = integrate_radial_emission(
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
            1.0,
            &coefficients.height,
            default_taper(),
            SHARPNESS,
        );
        assert!(value < 1e-6, "expected negligible emission, got {value}");
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
        // At radius 0: exp(0) - exp(-Emax) = 1 - exp(-Emax), not 1.0
        let emax = SHARPNESS * FLAME_RADIAL_RMAX * FLAME_RADIAL_RMAX;
        let expected_center = (1.0f32 - (-emax).exp());
        assert!(
            (evaluate_radial_density_factor([0.0, 0.5, 0.0], taper, SHARPNESS) - expected_center)
                .abs()
                < 1e-6
        );
        let inner = evaluate_radial_density_factor([0.15, 0.5, 0.0], taper, SHARPNESS);
        let outer = evaluate_radial_density_factor([0.3, 0.5, 0.0], taper, SHARPNESS);
        assert!(inner > outer && outer > 0.0);
    }

    #[test]
    fn test_gaussian_radius_stays_inside_the_shell_proxy() {
        let taper = default_taper();
        for step in 0..=20 {
            let height = step as f32 / 20.0;
            let density_edge = flame_radial_gaussian_scale(height, taper)
                * (SHARPNESS.recip() * (1.0f32 / 0.01).ln()).sqrt();
            let proxy = crate::flame_shell::flame_shell_outer_radius(height);
            assert!(
                density_edge <= proxy,
                "h={height}: density reaches {density_edge} but the proxy cuts at {proxy}"
            );
        }
    }

    #[test]
    fn test_taper_narrows_the_gaussian_with_height() {
        let taper = default_taper();
        let base = flame_radial_gaussian_scale(0.0, taper);
        let tip = flame_radial_gaussian_scale(1.0, taper);
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

    /// Test ①: Match between the new closed-form density (with pedestal subtraction) and numerical integration.
    #[test]
    fn test_pedestal_density_closed_form_matches_numerical() {
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

    /// A pedestal radius far outside the support makes exp(-Emax) vanish, so the integral must
    /// fall back onto the pure Gaussian.
    #[test]
    fn test_large_rmax_converges_to_pure_gaussian() {
        let coefficients = FlameCoefficients::default();
        let taper = default_taper();
        let rays = representative_rays();
        assert!(rays.len() > 50, "ray set too small: {}", rays.len());

        let large_rmax: f32 = 1e3;
        let references: Vec<f64> = rays
            .iter()
            .map(|(o, d, near, far)| {
                pure_gaussian_reference(*o, *d, *near, *far, &coefficients, taper, SHARPNESS)
            })
            .collect();
        let peak = references.iter().fold(0.0f64, |m, r| m.max(r.abs()));

        for ((origin, direction, t_near, t_far), reference) in rays.iter().zip(&references) {
            let value = integrate_radial_emission_with_rmax(
                *origin,
                *direction,
                *t_near,
                *t_far,
                &coefficients.height,
                taper,
                SHARPNESS,
                large_rmax,
            ) as f64;
            assert!(
                (value - reference).abs() < 0.01 * peak,
                "ray o={origin:?} d={direction:?}: got {value}, expected {reference}"
            );
        }
    }
}
