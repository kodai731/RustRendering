use crate::flame_shell::FLAME_SHELL_BASE_RADIUS;
use thyllore_math_core::{approximate_erf, biweight_profile, evaluate_chebyshev, ChebyshevSeries};

// Mirror of shaders/include/flame_radial_integral.glsl; the accuracy tests below cover both.
//
// The radial density is the compact-support biweight kernel
//   rho(p) = F(h) * (1 - u^2)^2,  u = |p.xz| / (S * R(h)),  zero for u >= 1.
// Along a ray u^2(s) is a quadratic g(s), so each height band is the exact
// polynomial integral of (F0 + F1 s + F2 s^2) * (1 - g(s))^2 over the interval
// where g(s) <= 1 — power-rule moments only, no tail and no pedestal.

/// Exponent variation across a band below which the constant-exponent moments are used.
const FLAT_EXPONENT: f32 = 2e-2;
/// Maximum support radius in R(h) units, bounded by the shell proxy headroom.
pub const FLAME_RADIAL_RMAX: f32 = crate::flame_shell::FLAME_SHELL_SUPPORT_HEADROOM;
/// Height taper of the radial density.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameRadialTaper {
    pub tip_ratio: f32,
    pub power: f32,
    pub baked_series: Option<[[f32; 4]; 2]>,
}

impl FlameRadialTaper {
    pub fn from_effect(
        effect: &crate::flame::FlameEffect,
        baked: &crate::flame::FlameBaked,
    ) -> Self {
        Self {
            tip_ratio: effect.radius_tip_ratio,
            power: effect.taper_power,
            baked_series: if baked.radius.is_some() && baked.blend > 0.0 {
                Some(effect.coefficients.radius_scale)
            } else {
                None
            },
        }
    }
}

/// Radius `R(h)` of the radial density profile, in flame-local units.
pub fn flame_radial_radius_scale(height01: f32, taper: FlameRadialTaper) -> f32 {
    if let Some(series) = taper.baked_series {
        FLAME_SHELL_BASE_RADIUS
            * evaluate_chebyshev(
                &ChebyshevSeries::new(series.iter().flatten().copied().collect(), (0.0, 1.0)),
                height01,
            )
            .max(0.05)
    } else {
        FLAME_SHELL_BASE_RADIUS * (1.0 + (taper.tip_ratio - 1.0) * height01.powf(taper.power))
    }
}

/// Support radius `S` of the biweight profile in R(h) units. The curvature at the
/// axis matches the former Gaussian `exp(-sharpness * u^2)`, so the sharpness parameter
/// keeps its direction: larger sharpness narrows the support.
pub fn flame_radial_support_radius(radial_sharpness: f32, support_margin: f32) -> f32 {
    support_margin
        * (2.0 / radial_sharpness.max(1e-3))
            .sqrt()
            .min(FLAME_RADIAL_RMAX)
}

fn support_inv_sq(
    height01: f32,
    taper: FlameRadialTaper,
    radial_sharpness: f32,
    support_margin: f32,
) -> f32 {
    let scale = (flame_radial_support_radius(radial_sharpness, support_margin)
        * flame_radial_radius_scale(height01, taper))
    .max(1e-4);
    1.0 / (scale * scale)
}

/// Radial density factor (1 - u^2)^2 at a flame-local point, zero outside the support.
pub fn evaluate_radial_density_factor(
    point_local: [f32; 3],
    taper: FlameRadialTaper,
    radial_sharpness: f32,
    support_margin: f32,
) -> f32 {
    let height = point_local[1].clamp(0.0, 1.0);
    let radius_squared = point_local[0] * point_local[0] + point_local[2] * point_local[2];
    biweight_profile(
        support_inv_sq(height, taper, radial_sharpness, support_margin) * radius_squared,
    )
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

/// Envelope fade toward the support boundary, shared by the flooded-erosion
/// argument and the unresolved-noise sigma of the band integrals (mirror of
/// `flameEnvelopeFade`). `flood_fade_scale` is edge_high in the shader.
pub fn envelope_fade(d_smooth: f32, flood_fade_scale: f32) -> f32 {
    (d_smooth / flood_fade_scale.max(1e-3)).min(1.0)
}

/// Threshold argument with the flooded (negative) erosion faded by the envelope,
/// so the field stays continuous across the support boundary (mirror of
/// `flameErodedArgument`).
pub fn eroded_argument(d_smooth: f32, erosion: f32, flood_fade_scale: f32) -> f32 {
    let base = d_smooth
        - (erosion.max(0.0) + erosion.min(0.0) * envelope_fade(d_smooth, flood_fade_scale));
    base * erosion_remap_scale(erosion)
}

/// Remap scale: inverse of the remaining range after erosion, floored at 0.15.
/// Mirrors `flameErosionRemapScale` in flame_noise_field.glsl.
pub const EROSION_REMAP_STRENGTH: f32 = 0.0; // 0 = off (default look), 1 = full Nubis-style remap
pub fn erosion_remap_scale(erosion: f32) -> f32 {
    let remapped = 1.0 / (1.0 - erosion.max(0.0)).max(0.15);
    (1.0 - EROSION_REMAP_STRENGTH) * 1.0 + EROSION_REMAP_STRENGTH * remapped
}

/// Smooth (pre-threshold) ring density at an unwarped local point, following the
pub fn evaluate_ring_smooth_density(
    point_local: [f32; 3],
    height_series: &ChebyshevSeries,
    taper: FlameRadialTaper,
    radial_sharpness: f32,
    ring_major_radius: f32,
    wiggle: f32,
    support_margin: f32,
) -> f32 {
    evaluate_ring_smooth_density_displaced(
        point_local,
        height_series,
        taper,
        radial_sharpness,
        ring_major_radius,
        wiggle,
        [1.0, 1.0],
        support_margin,
    )
}

/// Mirror of `flameEmitterSmoothDensityAt` with `[heightScale, radiusScale]` boundary displacement.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_ring_smooth_density_displaced(
    point_local: [f32; 3],
    height_series: &ChebyshevSeries,
    taper: FlameRadialTaper,
    radial_sharpness: f32,
    ring_major_radius: f32,
    wiggle: f32,
    boundary_scale: [f32; 2],
    support_margin: f32,
) -> f32 {
    let height = (point_local[1].clamp(0.0, 1.0) / boundary_scale[0]).clamp(0.0, 1.0);
    let taper_radius = 1.0 + (taper.tip_ratio - 1.0) * height.powf(taper.power);
    let minor_scale = (1.0 - ring_major_radius).max(1e-3);
    let radius = (point_local[0] * point_local[0] + point_local[2] * point_local[2]).sqrt();
    let rho = (radius - ring_major_radius) / minor_scale;
    let rn = rho.abs() / (taper_radius * wiggle * boundary_scale[1]).max(1e-4);
    let u = rn / flame_radial_support_radius(radial_sharpness, support_margin);
    evaluate_chebyshev(height_series, height) * biweight_profile(u * u)
}

/// Narrow `[t_near, t_far]` to the ray's crossing of the ring's outer support
/// cylinder, conservative over height (widest taper) and contour wiggle
/// (mirror of `flameRingSupportSpan`). The outer cylinder is convex, so the
/// span is a single interval whose endpoints vary continuously with the ray —
/// no per-pixel topology switches. `None` means the ray misses the support.
#[allow(clippy::too_many_arguments)]
pub fn ring_support_span(
    origin_local: [f32; 3],
    direction_local: [f32; 3],
    t_near: f32,
    t_far: f32,
    ring_major_radius: f32,
    taper: FlameRadialTaper,
    radial_sharpness: f32,
    wiggle_trim: f32,
    support_margin: f32,
) -> Option<(f32, f32)> {
    let minor_scale = (1.0 - ring_major_radius).max(1e-3);
    let taper_max = taper.tip_ratio.max(1.0);
    let r_out = ring_major_radius
        + minor_scale
            * flame_radial_support_radius(radial_sharpness, support_margin)
            * taper_max
            * wiggle_trim;

    let a = direction_local[0] * direction_local[0] + direction_local[2] * direction_local[2];
    let b = 2.0 * (origin_local[0] * direction_local[0] + origin_local[2] * direction_local[2]);
    let c = origin_local[0] * origin_local[0] + origin_local[2] * origin_local[2] - r_out * r_out;
    if a < 1e-12 {
        return (c <= 0.0).then_some((t_near, t_far));
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant <= 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let lo = ((-b - root) / (2.0 * a)).max(t_near);
    let hi = ((-b + root) / (2.0 * a)).min(t_far);
    (hi > lo).then_some((lo, hi))
}

pub fn build_height_series(height_coefficients: &[[f32; 4]; 2]) -> ChebyshevSeries {
    ChebyshevSeries::new(
        height_coefficients.iter().flatten().copied().collect(),
        (0.0, 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flame::{FlameCoefficients, FlameEffect};

    const SHARPNESS: f32 = 4.0;

    fn default_taper() -> FlameRadialTaper {
        FlameRadialTaper::from_effect(&FlameEffect::default(), &Default::default())
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
                let u_squared = support_inv_sq(h, taper, sharpness, 1.0) as f64 * radius_squared;
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
        let radius = crate::flame_shell::flame_shell_outer_radius(0.0, 1.0, 1.0);
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
    fn test_radial_density_factor_falls_off_with_radius() {
        let taper = default_taper();
        // On the axis the biweight profile is exactly 1.
        assert!(
            (evaluate_radial_density_factor([0.0, 0.5, 0.0], taper, SHARPNESS, 1.0) - 1.0).abs()
                < 1e-6
        );
        let support_edge =
            flame_radial_support_radius(SHARPNESS, 1.0) * flame_radial_radius_scale(0.5, taper);
        let inner =
            evaluate_radial_density_factor([0.3 * support_edge, 0.5, 0.0], taper, SHARPNESS, 1.0);
        let outer =
            evaluate_radial_density_factor([0.7 * support_edge, 0.5, 0.0], taper, SHARPNESS, 1.0);
        assert!(inner > outer && outer > 0.0);
        // Outside the support the density is exactly zero.
        assert_eq!(
            evaluate_radial_density_factor([support_edge + 1e-3, 0.5, 0.0], taper, SHARPNESS, 1.0),
            0.0
        );
    }

    #[test]
    fn test_support_radius_stays_inside_the_shell_proxy() {
        let taper = default_taper();
        for sharpness in [0.5f32, 1.0, 3.0, 4.0, 6.0, 8.0, 16.0] {
            for step in 0..=20 {
                let height = step as f32 / 20.0;
                let density_edge = flame_radial_support_radius(sharpness, 1.0)
                    * flame_radial_radius_scale(height, taper);
                let proxy = crate::flame_shell::flame_shell_outer_radius(height, 1.0, 1.0);
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

    mod occupancy {
        use super::super::*;
        use super::{default_taper, SHARPNESS};
        use crate::flame::FlameCoefficients;

        const EDGE_LOW: f32 = 0.27;
        const EDGE_HIGH: f32 = 0.33;

        #[test]
        fn test_displaced_density_with_unit_scales_matches_baseline_bitwise() {
            let coefficients = FlameCoefficients::default();
            let height = build_height_series(&coefficients.height);
            let taper = default_taper();
            for p in [
                [-1.2f32, 0.1, 0.3],
                [0.75, 0.5, 0.0],
                [0.2, 0.95, -0.6],
                [1.0, 0.0, 1.0],
            ] {
                let baseline = evaluate_ring_smooth_density(p, &height, taper, 4.0, 0.75, 1.1, 1.0);
                let displaced = evaluate_ring_smooth_density_displaced(
                    p,
                    &height,
                    taper,
                    4.0,
                    0.75,
                    1.1,
                    [1.0, 1.0],
                    1.0,
                );
            }
        }

        /// The trim must be conservative: every t where the ring density is
        /// positive lies inside the returned span, and the span stays inside
        /// the query range.
        #[test]
        fn test_ring_support_span_covers_the_density() {
            let coefficients = FlameCoefficients::default();
            let height = build_height_series(&coefficients.height);
            let taper = default_taper();
            let ring_major = 0.75f32;
            let rays: Vec<([f32; 3], [f32; 3], f32, f32)> = vec![
                ([-1.4, 0.1, 0.0], [1.0, 0.05, 0.0], 0.0, 2.8), // through both walls
                ([0.75, 0.0, -1.2], [0.0, 0.2, 1.0], 0.0, 2.4), // along one wall
                ([0.0, 1.4, 0.0], [0.4, -1.0, 0.3], 0.4, 1.4),  // from above through the hole
                ([0.0, 0.05, 0.0], [1.0, 0.02, 0.0], 0.0, 0.4), // inside the hole, short
                ([0.0, -0.2, 0.0], [0.0, 1.0, 0.0], 0.0, 1.5),  // vertical through the hole
                ([0.75, -0.2, 0.0], [0.0, 1.0, 0.0], 0.0, 1.5), // vertical inside the wall
                ([-3.0, 0.5, 2.0], [1.0, 0.0, 0.0], 0.0, 6.0),  // missing the support
            ];
            for (o, d, t0, t1) in rays {
                let span = ring_support_span(o, d, t0, t1, ring_major, taper, SHARPNESS, 1.0, 1.0);
                if let Some((lo, hi)) = span {
                    assert!(lo >= t0 - 1e-5 && hi <= t1 + 1e-5, "span out of range");
                }
                let steps = 4000;
                for i in 0..steps {
                    let t = t0 + (i as f32 + 0.5) * (t1 - t0) / steps as f32;
                    let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                    let density = evaluate_ring_smooth_density(
                        p, &height, taper, SHARPNESS, ring_major, 1.0, 1.0,
                    );
                    if density > 0.0 {
                        let covered = span.is_some_and(|(lo, hi)| t >= lo && t <= hi);
                        assert!(
                            covered,
                            "o={o:?} d={d:?} t={t}: density {density} outside trim ({span:?})"
                        );
                    }
                }
            }
        }

        /// The point of the flooded-erosion fade: with strongly negative erosion the
        /// pointwise field must fall to zero continuously at the support boundary
        /// instead of jumping from saturation to the mask cut.
        #[test]
        fn test_flooded_field_is_continuous_at_support_edge() {
            let smoothstep = |x: f32| {
                let t = ((x - EDGE_LOW) / (EDGE_HIGH - EDGE_LOW)).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            };
            for erosion in [-0.5f32, -2.0] {
                let mut previous = 0.0f32;
                for step in 0..=4000 {
                    let d_smooth = step as f32 / 4000.0;
                    let value = smoothstep(eroded_argument(d_smooth, erosion, EDGE_HIGH));
                    assert!(
                        (value - previous).abs() < 0.05,
                        "e={erosion} d={d_smooth}: jump {previous} -> {value}"
                    );
                    previous = value;
                }
                assert!(
                    smoothstep(eroded_argument(1e-4, erosion, EDGE_HIGH)) < 1e-3,
                    "field must vanish with the envelope"
                );
            }
        }
    }
}
