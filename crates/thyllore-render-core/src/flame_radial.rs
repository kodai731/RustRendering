use crate::flame_shell::FLAME_SHELL_BASE_RADIUS;
use thyllore_math_core::{
    approximate_erf, biweight_profile, chebyshev_monomial_coefficients, evaluate_chebyshev,
    integrate_erf_response_linear, integrate_powers, solve_support_interval, ChebyshevSeries,
    ErfResponseModel,
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
    pub baked_series: Option<[[f32; 4]; 2]>,
}

impl FlameRadialTaper {
    pub fn from_effect(effect: &crate::flame::FlameEffect) -> Self {
        Self {
            tip_ratio: effect.radius_tip_ratio,
            power: effect.taper_power,
            baked_series: if effect.baked_radius.is_some() && effect.baked_blend > 0.0 {
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
    d_smooth - (erosion.max(0.0) + erosion.min(0.0) * envelope_fade(d_smooth, flood_fade_scale))
}

/// Fixed fbm node spacing per band (mirror of `FLAME_OCCUPANCY_EDGE_SEGMENTS`).
pub const FLAME_OCCUPANCY_EDGE_SEGMENTS: usize = 2;
/// Fixed sub-pieces per node piece for the argument re-linearization.
pub const FLAME_OCCUPANCY_SUB_PIECES: usize = 3;

/// One linear-argument sub-piece of the occupancy band model: response argument
/// `x(s)` linear from `x0` at `s0` to `x1` at `s1`, smoothed with `sigma_eff`.
#[derive(Clone, Copy, Debug)]
pub struct OccupancyPiece {
    pub s0: f32,
    pub s1: f32,
    pub x0: f32,
    pub x1: f32,
    pub sigma_eff: f32,
    /// Un-eroded density at the sub-piece ends (linear between them); the
    /// occupancy is the mixture `(1 - beta) phi(x) + beta phi(density)`.
    pub plain0: f32,
    pub plain1: f32,
}

/// The occupancy band model shared by the closed form and its accuracy tests:
/// pieces cut at fixed band fractions plus the field-fixed `extra_nodes` (never
/// at support-adaptive positions — those move with kinked derivatives as the
/// ray sweeps and image fold lines through the fbm), the argument linearized on
/// `FLAME_OCCUPANCY_SUB_PIECES` fixed sub-pieces of each piece clipped to the
/// support, with density AND erosion evaluated exactly at the sub-nodes (an
/// interpolated erosion renders the expectation of the field — a diffuse cap
/// with a coherent transparent gap at the displacement transition band — where
/// the true field is the sublevel-set geometry of the erosion itself).
/// `sigma_at(s0, s1)` gives the residual erosion fluctuation over one sub-piece,
/// faded per sub-piece by the sub-node envelope fades. `carve_residual` is the
/// fraction of the medium turbulence cannot carve (mixture with the un-eroded
/// threshold response), which makes a fully carved hole impossible.
/// Mirror of `flameOccupancyBandExact` in shaders/include/flame_radial_integral.glsl.
#[allow(clippy::too_many_arguments)]
pub fn occupancy_band_pieces(
    sigma_at: impl Fn(f32, f32) -> f32,
    a: f32,
    b: f32,
    c: f32,
    density_at: impl Fn(f32) -> f32,
    half_width: f32,
    erosion_at: impl Fn(f32) -> f32,
    flood_fade_scale: f32,
    carve_residual: f32,
    extra_nodes: &[f32],
) -> Vec<OccupancyPiece> {
    let Some((clip_lo, clip_hi)) = solve_support_interval(a, b, c, -half_width, half_width) else {
        return Vec::new();
    };

    let mut nodes: Vec<f32> = vec![-half_width];
    for k in 1..FLAME_OCCUPANCY_EDGE_SEGMENTS {
        nodes.push(-half_width + 2.0 * half_width * k as f32 / FLAME_OCCUPANCY_EDGE_SEGMENTS as f32);
    }
    for &cut in extra_nodes {
        if cut > -half_width + 1e-6 && cut < half_width - 1e-6 {
            nodes.push(cut);
        }
    }
    nodes.push(half_width);
    nodes.sort_by(f32::total_cmp);
    nodes.dedup_by(|a, b| (*a - *b).abs() < 1e-7);

    let mut pieces = Vec::new();
    for i in 1..nodes.len() {
        let piece_lo = nodes[i - 1].max(clip_lo);
        let piece_hi = nodes[i].min(clip_hi);
        let clipped = piece_hi - piece_lo;
        if clipped < 1e-7 {
            continue;
        }
        let sub_span = clipped / FLAME_OCCUPANCY_SUB_PIECES as f32;
        let piece_sigma = sigma_at(piece_lo, piece_lo + sub_span);
        let sub = FLAME_OCCUPANCY_SUB_PIECES;
        let mut argument = [0.0f32; FLAME_OCCUPANCY_SUB_PIECES + 1];
        let mut fade = [0.0f32; FLAME_OCCUPANCY_SUB_PIECES + 1];
        let mut plain = [0.0f32; FLAME_OCCUPANCY_SUB_PIECES + 1];
        for j in 0..=sub {
            let s = piece_lo + clipped * j as f32 / sub as f32;
            let erosion = erosion_at(s);
            let density = density_at(s);
            argument[j] = eroded_argument(density, erosion, flood_fade_scale);
            fade[j] = envelope_fade(density, flood_fade_scale);
            plain[j] = density;
        }
        let sub_span = clipped / sub as f32;
        for j in 1..=sub {
            pieces.push(OccupancyPiece {
                s0: piece_lo + (j - 1) as f32 * sub_span,
                s1: piece_lo + j as f32 * sub_span,
                x0: argument[j - 1],
                x1: argument[j],
                sigma_eff: piece_sigma * 0.5 * (fade[j - 1] + fade[j]),
                plain0: plain[j - 1],
                plain1: plain[j],
            });
        }
    }
    pieces
}

/// Occupancy integral and first moment of the smoothed threshold response over
/// the support part of `[-half_width, half_width]` — the closed-form sum of the
/// `occupancy_band_pieces` model.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_occupancy_band(
    model: &ErfResponseModel,
    sigma_at: impl Fn(f32, f32) -> f32,
    a: f32,
    b: f32,
    c: f32,
    density_at: impl Fn(f32) -> f32,
    half_width: f32,
    erosion_at: impl Fn(f32) -> f32,
    flood_fade_scale: f32,
    carve_residual: f32,
    extra_nodes: &[f32],
) -> (f32, f32) {
    let mut total = (0.0f32, 0.0f32);
    for piece in occupancy_band_pieces(
        sigma_at,
        a,
        b,
        c,
        density_at,
        half_width,
        erosion_at,
        flood_fade_scale,
        carve_residual,
        extra_nodes,
    ) {
        let span = piece.s1 - piece.s0;
        if span < 1e-7 {
            continue;
        }
        let slope = (piece.x1 - piece.x0) / span;
        let (integral, first_moment) = integrate_erf_response_linear(
            model,
            piece.sigma_eff,
            piece.x0 - slope * piece.s0,
            slope,
            piece.s0,
            piece.s1,
        );
        // Residual (un-carveable) fraction: the same threshold response applied
        // to the un-eroded density, which is linear on the piece like the carved
        // argument — one more closed-form evaluation, nothing sampled.
        let (integral, first_moment) = if carve_residual > 0.0 {
            let plain_slope = (piece.plain1 - piece.plain0) / span;
            let (plain_integral, plain_moment) = integrate_erf_response_linear(
                model,
                0.0,
                piece.plain0 - plain_slope * piece.s0,
                plain_slope,
                piece.s0,
                piece.s1,
            );
            (
                integral + carve_residual * (plain_integral - integral),
                first_moment + carve_residual * (plain_moment - first_moment),
            )
        } else {
            (integral, first_moment)
        };
        total.0 += integral;
        total.1 += first_moment;
    }
    total
}

/// Exact integral and first moment about s = 0 of `F(h0 + h_slope s) * (1 - g(s))^2`
/// over `[s0, s1]` inside the support, with F given as monomial coefficients in
/// `u = 2h - 1`. The envelope composes with the affine h(s) into a single monomial
/// in the piece-centered variable, so the integral is exact power-rule moments —
/// no within-band approximation (mirror of `flameExactPieceEmission` without the
/// boundary displacement / capFade regimes, which only exist with boundary_amp != 0).
fn exact_piece_emission(
    a: f32,
    b: f32,
    c: f32,
    h0: f32,
    h_slope: f32,
    monomial: &[f32; 8],
    s0: f32,
    s1: f32,
) -> (f32, f32) {
    let w = 0.5 * (s1 - s0);
    if w <= 1e-7 {
        return (0.0, 0.0);
    }
    let s_mid = 0.5 * (s0 + s1);
    let h_mid = h0 + h_slope * s_mid;
    let u_scale = 2.0 * h_slope;
    let u_bias = 2.0 * h_mid - 1.0;
    let mut envelope = [0.0f32; 8];
    for k in (0..8).rev() {
        for i in (1..8).rev() {
            envelope[i] = envelope[i] * u_bias + envelope[i - 1] * u_scale;
        }
        envelope[0] = envelope[0] * u_bias + monomial[k];
    }

    let gb = 2.0 * a * s_mid + b;
    let gc = (a * s_mid + b) * s_mid + c;
    let m = 1.0 - gc;
    let w0 = m * m;
    let w1 = -2.0 * gb * m;
    let w2 = gb * gb - 2.0 * a * m;
    let w3 = 2.0 * a * gb;
    let w4 = a * a;
    let mut e = [0.0f32; 12];
    for i in 0..8 {
        e[i] += envelope[i] * w0;
        e[i + 1] += envelope[i] * w1;
        e[i + 2] += envelope[i] * w2;
        e[i + 3] += envelope[i] * w3;
        e[i + 4] += envelope[i] * w4;
    }

    let mut integral = 0.0f32;
    let mut first = 0.0f32;
    let mut w_pow = w;
    for (n, coefficient) in e.iter().enumerate() {
        if n % 2 == 0 {
            integral += coefficient * 2.0 * w_pow / (n as f32 + 1.0);
        } else {
            first += coefficient * 2.0 * w_pow * w / (n as f32 + 2.0);
        }
        w_pow *= w;
    }
    (integral, s_mid * integral + first)
}

/// Exact band emission over the support part of `[-half_width, half_width]`
/// (mirror of `flameExactBandEmission`; a single piece because without boundary
/// displacement the envelope is one global polynomial — no capFade, no tip clamp).
pub fn evaluate_exact_band_emission(
    a: f32,
    b: f32,
    c: f32,
    h0: f32,
    h_slope: f32,
    monomial: &[f32; 8],
    half_width: f32,
) -> (f32, f32) {
    let Some((s_lo, s_hi)) = solve_support_interval(a, b, c, -half_width, half_width) else {
        return (0.0, 0.0);
    };
    exact_piece_emission(a, b, c, h0, h_slope, monomial, s_lo, s_hi)
}

/// Monomial coefficients (in u = 2h - 1) of the packed height envelope.
pub fn height_envelope_monomial(height_coefficients: &[[f32; 4]; 2]) -> [f32; 8] {
    let series = build_height_series(height_coefficients);
    let mut monomial = [0.0f32; 8];
    for (slot, value) in monomial
        .iter_mut()
        .zip(chebyshev_monomial_coefficients(&series))
    {
        *slot = value;
    }
    monomial
}

pub const FLAME_OCCUPANCY_NODE_SEGMENTS: usize = 4;

/// Smooth (pre-threshold) ring density at an unwarped local point, following the
/// mode-3 field conventions minus the domain warp (`flameEmitterSmoothDensity`).
pub fn evaluate_ring_smooth_density(
    point_local: [f32; 3],
    height_series: &ChebyshevSeries,
    taper: FlameRadialTaper,
    radial_sharpness: f32,
    ring_major_radius: f32,
    wiggle: f32,
) -> f32 {
    evaluate_ring_smooth_density_displaced(
        point_local,
        height_series,
        taper,
        radial_sharpness,
        ring_major_radius,
        wiggle,
        [1.0, 1.0],
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
) -> f32 {
    let height = (point_local[1].clamp(0.0, 1.0) / boundary_scale[0]).clamp(0.0, 1.0);
    let taper_radius = 1.0 + (taper.tip_ratio - 1.0) * height.powf(taper.power);
    let minor_scale = (1.0 - ring_major_radius).max(1e-3);
    let radius = (point_local[0] * point_local[0] + point_local[2] * point_local[2]).sqrt();
    let rho = (radius - ring_major_radius) / minor_scale;
    let rn = rho.abs() / (taper_radius * wiggle * boundary_scale[1]).max(1e-4);
    let u = rn / flame_radial_support_radius(radial_sharpness);
    evaluate_chebyshev(height_series, height) * biweight_profile(u * u)
}

/// Occupancy integral and first moment in t of the smoothed threshold response over
/// one t band, argument piecewise-linear between the segment nodes. Mirror of
/// `flameOccupancyNodeBand`: fbm passes the per-node erosion with the per-segment
/// residual sigma via `sigma_at(t_prev, t)`, the kernel basis passes the exact
/// per-node erosion with sigma = 0.
pub fn evaluate_occupancy_node_band(
    model: &ErfResponseModel,
    sigma_at: impl Fn(f32, f32) -> f32,
    t0: f32,
    t1: f32,
    erosion_at: impl Fn(f32) -> f32,
    flood_fade_scale: f32,
    density_at: impl Fn(f32) -> f32,
) -> (f32, f32) {
    let dt = (t1 - t0) / FLAME_OCCUPANCY_NODE_SEGMENTS as f32;
    if dt <= 0.0 {
        return (0.0, 0.0);
    }
    let mut total = (0.0, 0.0);
    let mut t_prev = t0;
    let mut density_prev = density_at(t0);
    let mut argument_prev = eroded_argument(density_prev, erosion_at(t0), flood_fade_scale);
    for segment in 1..=FLAME_OCCUPANCY_NODE_SEGMENTS {
        let t = t0 + segment as f32 * dt;
        let density = density_at(t);
        let argument = eroded_argument(density, erosion_at(t), flood_fade_scale);
        if density_prev > 0.0 || density > 0.0 {
            let sigma_eff = sigma_at(t_prev, t)
                * 0.5
                * (envelope_fade(density_prev, flood_fade_scale)
                    + envelope_fade(density, flood_fade_scale));
            let slope = (argument - argument_prev) / dt;
            let piece = integrate_erf_response_linear(
                model,
                sigma_eff,
                argument_prev - slope * t_prev,
                slope,
                t_prev,
                t,
            );
            total.0 += piece.0;
            total.1 += piece.1;
        }
        t_prev = t;
        density_prev = density;
        argument_prev = argument;
    }
    total
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
) -> Option<(f32, f32)> {
    let minor_scale = (1.0 - ring_major_radius).max(1e-3);
    let taper_max = taper.tip_ratio.max(1.0);
    let r_out = ring_major_radius
        + minor_scale * flame_radial_support_radius(radial_sharpness) * taper_max * wiggle_trim;

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

/// Density-weighted node position of one t band (mirror of the erosion sample
/// placement in `flameOccupancyNodeBand`): the frozen erosion is sampled where
/// the band actually carries density, not at the band midpoint. Returns `None`
/// when every node density is zero (empty band).
pub fn density_weighted_node_t(t0: f32, t1: f32, density_at: impl Fn(f32) -> f32) -> Option<f32> {
    let dt = (t1 - t0) / FLAME_OCCUPANCY_NODE_SEGMENTS as f32;
    if dt <= 0.0 {
        return None;
    }
    let mut weight_sum = 0.0f32;
    let mut t_weighted = 0.0f32;
    for node in 0..=FLAME_OCCUPANCY_NODE_SEGMENTS {
        let t = t0 + node as f32 * dt;
        let density = density_at(t);
        weight_sum += density;
        t_weighted += density * t;
    }
    (weight_sum > 0.0).then(|| t_weighted / weight_sum)
}

pub(crate) fn build_height_series(height_coefficients: &[[f32; 4]; 2]) -> ChebyshevSeries {
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
    let monomial = height_envelope_monomial(height_coefficients);
    let mut total = 0.0;
    for band in 0..FLAME_RADIAL_BAND_COUNT {
        let center = height_lo + (band as f32 + 0.5) * band_width;
        let inv_sq = support_inv_sq(center, taper, radial_sharpness);
        let point_xz = ray_point_at_height(origin_local, q, center);
        let (integral, _) = evaluate_exact_band_emission(
            inv_sq * quadratic,
            2.0 * inv_sq * (point_xz[0] * q[0] + point_xz[1] * q[1]),
            inv_sq * (point_xz[0] * point_xz[0] + point_xz[1] * point_xz[1]),
            center,
            1.0,
            &monomial,
            half_width,
        );
        total += integral;
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

    mod occupancy {
        use super::super::*;
        use super::{default_taper, SHARPNESS};
        use crate::flame::FlameCoefficients;
        use thyllore_math_core::{evaluate_erf_response, fit_erf_response};

        const EDGE_LOW: f32 = 0.27;
        const EDGE_HIGH: f32 = 0.33;

        fn default_model() -> ErfResponseModel {
            fit_erf_response(EDGE_LOW, EDGE_HIGH)
        }

        /// Quadratic-envelope density of the synthetic band cases:
        /// `(f.0 + f.1 s + f.2 s^2) (1 - g(s))^2` with `g(s) = a s^2 + b s + c`.
        fn occupancy_density(a: f32, b: f32, c: f32, f: (f32, f32, f32), s: f32) -> f32 {
            let g = (a * s + b) * s + c;
            let inside = (1.0 - g).max(0.0);
            (f.0 + (f.1 + f.2 * s) * s) * inside * inside
        }

        /// Band cases (a, b, c, f, half_width, erosion) spanning interior chords,
        /// grazing chords, negative erosion flooding and threshold-straddling arguments.
        fn band_cases() -> Vec<(f32, f32, f32, (f32, f32, f32), f32, f32)> {
            vec![
                (0.5, 0.2, 0.4, (0.8, 0.3, -0.6), 0.25, 0.1),
                (40.0, -6.0, 0.9, (1.0, -0.5, 2.0), 0.125, 0.0),
                (0.0, 0.0, 0.3, (0.6, 0.0, 0.0), 0.125, 0.25),
                (900.0, -60.0, 1.5, (0.7, 0.1, 0.0), 0.05, -0.2),
                (4.0, 0.0, 0.0, (0.9, 0.0, 0.0), 0.5, 0.15),
                (0.5, 0.1, 0.2, (0.35, 0.05, 0.0), 0.3, 0.05),
            ]
        }

        /// The closed form must match quadrature of the response applied to the
        /// shared piecewise-linear argument model — this isolates the moment
        /// algebra from the linearization choice.
        #[test]
        fn test_band_matches_quadrature_of_linearized_argument() {
            let model = default_model();
            for sigma in [0.0f32, 0.05, 0.2] {
                for (a, b, c, f, half, erosion) in band_cases() {
                    let density = |s: f32| occupancy_density(a, b, c, f, s);
                    let (integral, first_moment) = evaluate_occupancy_band(
                        &model,
                        |_, _| sigma,
                        a,
                        b,
                        c,
                        density,
                        half,
                        |_| erosion,
                        EDGE_HIGH,
                        0.0,
                        &[],
                    );
                    let pieces = occupancy_band_pieces(
                        |_, _| sigma,
                        a,
                        b,
                        c,
                        density,
                        half,
                        |_| erosion,
                        EDGE_HIGH,
                        0.0,
                        &[],
                    );
                    if pieces.is_empty() {
                        assert_eq!(integral, 0.0);
                        continue;
                    }
                    let mut reference = 0.0f64;
                    let mut reference_first = 0.0f64;
                    let mut scale = 0.0f64;
                    for piece in &pieces {
                        let steps = 4000;
                        let ds = (piece.s1 - piece.s0) as f64 / steps as f64;
                        scale += (piece.s1 - piece.s0) as f64;
                        for i in 0..steps {
                            let s = piece.s0 as f64 + (i as f64 + 0.5) * ds;
                            let t = ((s - piece.s0 as f64) / (piece.s1 - piece.s0) as f64) as f32;
                            let x = piece.x0 + (piece.x1 - piece.x0) * t;
                            let value = evaluate_erf_response(&model, x, piece.sigma_eff) as f64;
                            reference += value * ds;
                            reference_first += s * value * ds;
                        }
                    }
                    assert!(
                        (integral as f64 - reference).abs() < 3e-3 * scale.max(1e-3),
                        "sigma={sigma} case=({a},{b},{c}): got {integral}, expected {reference}"
                    );
                    assert!(
                        (first_moment as f64 - reference_first).abs() < 3e-3 * scale.max(1e-3),
                        "sigma={sigma} case=({a},{b},{c}) first moment: got {first_moment}, expected {reference_first}"
                    );
                }
            }
        }

        /// Against the true styled field (exact smoothstep of the exact argument):
        /// the sigma = 0 closed form carries the erf fit floor plus the linearization
        /// error, which must stay a small fraction of the support length.
        #[test]
        fn test_band_tracks_true_smoothstep_field() {
            let model = default_model();
            for (a, b, c, f, half, erosion) in band_cases() {
                let (integral, _) = evaluate_occupancy_band(
                    &model,
                    |_, _| 0.0,
                    a,
                    b,
                    c,
                    |s| occupancy_density(a, b, c, f, s),
                    half,
                    |_| erosion,
                    EDGE_HIGH,
                    0.0,
                    &[],
                );
                let Some((s_lo, s_hi)) = solve_support_interval(a, b, c, -half, half) else {
                    continue;
                };
                let steps = 40000;
                let ds = (s_hi - s_lo) as f64 / steps as f64;
                let mut reference = 0.0f64;
                for i in 0..steps {
                    let s = s_lo as f64 + (i as f64 + 0.5) * ds;
                    let x =
                        eroded_argument(occupancy_density(a, b, c, f, s as f32), erosion, EDGE_HIGH)
                            as f64;
                    let t = ((x - EDGE_LOW as f64) / (EDGE_HIGH - EDGE_LOW) as f64).clamp(0.0, 1.0);
                    reference += t * t * (3.0 - 2.0 * t) * ds;
                }
                // Worst measured: 8.9% of the support on a grazing chord (a=900) where
                // the two-piece secant misses the argument curvature; interior chords
                // stay a few percent. This is the linearization floor, not the moments.
                let scale = (s_hi - s_lo) as f64;
                assert!(
                    (integral as f64 - reference).abs() < 0.12 * scale,
                    "case=({a},{b},{c},e={erosion}): got {integral}, expected {reference} over {scale}"
                );
            }
        }

        /// Strong erosion empties the band; strongly negative erosion (turbulence)
        /// fills the whole support — and only the support (exact membership).
        #[test]
        fn test_band_saturates_to_empty_and_full_support() {
            let model = default_model();
            let (a, b, c, f, half) = (0.5f32, 0.2f32, 0.4f32, (0.8f32, 0.0f32, 0.0f32), 0.25f32);
            let (s_lo, s_hi) = solve_support_interval(a, b, c, -half, half).unwrap();
            let density = |s: f32| occupancy_density(a, b, c, f, s);
            let (empty, _) = evaluate_occupancy_band(
                &model,
                |_, _| 0.0,
                a,
                b,
                c,
                density,
                half,
                |_| 2.0,
                EDGE_HIGH,
                0.0,
                &[],
            );
            assert!(empty.abs() < 1e-4, "eroded band should vanish: {empty}");
            let (full, _) = evaluate_occupancy_band(
                &model,
                |_, _| 0.0,
                a,
                b,
                c,
                density,
                half,
                |_| -2.0,
                EDGE_HIGH,
                0.0,
                &[],
            );
            let support = s_hi - s_lo;
            assert!(
                (full - support).abs() < 0.02 * support,
                "flooded band should fill the support: {full} vs {support}"
            );
            let (outside, _) = evaluate_occupancy_band(
                &model,
                |_, _| 0.0,
                0.0,
                0.0,
                3.0,
                |s| occupancy_density(0.0, 0.0, 3.0, f, s),
                half,
                |_| -2.0,
                EDGE_HIGH,
                0.0,
                &[],
            );
            assert_eq!(outside, 0.0, "no support, no occupancy even when flooded");
        }

        /// Larger unresolved noise flattens the response monotonically toward the
        /// half-occupied band instead of jittering.
        #[test]
        fn test_sigma_flattens_the_band_smoothly() {
            let model = default_model();
            let (a, b, c, f, half) = (4.0f32, 0.0f32, 0.0f32, (0.9f32, 0.0f32, 0.0f32), 0.5f32);
            let density = |s: f32| occupancy_density(a, b, c, f, s);
            let sharp = evaluate_occupancy_band(
                &model,
                |_, _| 0.0,
                a,
                b,
                c,
                density,
                half,
                |_| 0.0,
                EDGE_HIGH,
                0.0,
                &[],
            )
            .0;
            let soft = evaluate_occupancy_band(
                &model,
                |_, _| 0.3,
                a,
                b,
                c,
                density,
                half,
                |_| 0.0,
                EDGE_HIGH,
                0.0,
                &[],
            )
            .0;
            let (s_lo, s_hi) = solve_support_interval(a, b, c, -half, half).unwrap();
            let support = s_hi - s_lo;
            assert!(sharp > soft, "smoothing must not add occupancy here");
            assert!(soft > 0.3 * support && soft < support);
        }

        /// Ring rays for the node-based generic band: chords crossing the ring wall,
        /// running inside the trough, and missing the support entirely.
        #[test]
        fn test_node_band_matches_quadrature_on_ring_chords() {
            let model = default_model();
            let coefficients = FlameCoefficients::default();
            let height = build_height_series(&coefficients.height);
            let taper = default_taper();
            let ring_major = 0.75f32;
            let wiggle = 1.0f32;
            let rays: Vec<([f32; 3], [f32; 3], f32, f32)> = vec![
                ([-1.4, 0.1, 0.0], [1.0, 0.05, 0.0], 0.0, 2.8),
                ([0.75, 0.0, -1.2], [0.0, 0.2, 1.0], 0.0, 2.4),
                ([0.0, 1.4, 0.0], [0.4, -1.0, 0.3], 0.4, 1.4),
                ([0.0, 0.05, 0.0], [1.0, 0.02, 0.0], 0.0, 0.4),
            ];
            for sigma in [0.0f32, 0.1] {
                for erosion in [0.0f32, 0.15, -0.2] {
                    for (o, d, t0, t1) in &rays {
                        let sample = |t: f32| {
                            let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                            evaluate_ring_smooth_density(p, &height, taper, 4.0, ring_major, wiggle)
                        };
                        let (integral, first_moment) = evaluate_occupancy_node_band(
                            &model,
                            |_, _| sigma,
                            *t0,
                            *t1,
                            |_| erosion,
                            EDGE_HIGH,
                            &sample,
                        );
                        // Reference: the response applied to the node-linearized argument.
                        let dt = (t1 - t0) / FLAME_OCCUPANCY_NODE_SEGMENTS as f32;
                        let steps = 40000;
                        let ds = (t1 - t0) as f64 / steps as f64;
                        let mut reference = 0.0f64;
                        let mut reference_first = 0.0f64;
                        for i in 0..steps {
                            let t = *t0 as f64 + (i as f64 + 0.5) * ds;
                            let segment = (((t - *t0 as f64) / dt as f64).floor() as usize)
                                .min(FLAME_OCCUPANCY_NODE_SEGMENTS - 1);
                            let ta = *t0 + segment as f32 * dt;
                            let da = sample(ta);
                            let db = sample(ta + dt);
                            if da <= 0.0 && db <= 0.0 {
                                continue;
                            }
                            let arg_a = eroded_argument(da, erosion, EDGE_HIGH);
                            let arg_b = eroded_argument(db, erosion, EDGE_HIGH);
                            let x = arg_a + (arg_b - arg_a) * ((t as f32 - ta) / dt);
                            let sigma_eff = sigma
                                * 0.5
                                * (envelope_fade(da, EDGE_HIGH) + envelope_fade(db, EDGE_HIGH));
                            let value = evaluate_erf_response(&model, x, sigma_eff) as f64;
                            reference += value * ds;
                            reference_first += t * value * ds;
                        }
                        let scale = (t1 - t0) as f64;
                        assert!(
                            (integral as f64 - reference).abs() < 3e-3 * scale.max(1e-3),
                            "sigma={sigma} e={erosion} o={o:?}: got {integral}, expected {reference}"
                        );
                        assert!(
                            (first_moment as f64 - reference_first).abs()
                                < 3e-3 * scale.max(1e-3) * (t1.abs().max(1.0)) as f64,
                            "sigma={sigma} e={erosion} o={o:?} first moment: got {first_moment}, expected {reference_first}"
                        );
                    }
                }
            }
        }

        /// Dumped user rays of log/flame/wall_probe_1785662819.json (near-wall ring pose).
        #[test]
        fn test_node_band_matches_quadrature_on_dumped_wall_pose_rays() {
            let model = default_model();
            let height = build_height_series(&[
                [0.353_472_62, -0.206_110_25, -0.374_551_45, 0.258_819_37],
                [
                    -0.004_593_815,
                    -0.058_476_381,
                    0.059_848_253,
                    -0.038_838_845,
                ],
            ]);
            let taper = default_taper();
            let ring_major = 0.75f32;
            let origin = [-0.018_554f32, 0.509_862, 0.9567];
            let directions: [[f32; 3]; 5] = [
                [0.088_952, -0.108_642, -0.990_093],
                [0.085_707, 0.287_378, -0.953_975],
                [-0.561_305, 0.223_091, -0.796_973],
                [0.671_719, -0.371_655, -0.640_832],
                [0.078_993, -0.469_792, -0.879_236],
            ];
            const BAND_COUNT: usize = 6;

            for direction in &directions {
                let slab_hi = if direction[1].abs() < 1e-6 {
                    6.0
                } else {
                    let a = (0.0 - origin[1]) / direction[1];
                    let b = (1.0 - origin[1]) / direction[1];
                    a.max(b).min(6.0)
                };
                let Some((span_lo, span_hi)) = ring_support_span(
                    origin, *direction, 0.0, slab_hi, ring_major, taper, SHARPNESS, 1.3,
                ) else {
                    panic!("dumped ray {direction:?} must hit the ring support");
                };

                for (boundary_scale, sigma, erosion) in [
                    ([1.0f32, 1.0], 0.144f32, -0.35f32),
                    ([1.0, 1.0], 0.144, 0.4),
                    ([1.06, 0.94], 0.144, 0.0),
                    ([1.06, 0.94], 0.0, 0.4),
                ] {
                    let sample = |t: f32| {
                        let p = [
                            origin[0] + t * direction[0],
                            origin[1] + t * direction[1],
                            origin[2] + t * direction[2],
                        ];
                        evaluate_ring_smooth_density_displaced(
                            p,
                            &height,
                            taper,
                            SHARPNESS,
                            ring_major,
                            1.0,
                            boundary_scale,
                        )
                    };

                    let band_width = (span_hi - span_lo) / BAND_COUNT as f32;
                    for band in 0..BAND_COUNT {
                        let t0 = span_lo + band as f32 * band_width;
                        let t1 = t0 + band_width;
                        let (integral, first_moment) = evaluate_occupancy_node_band(
                            &model,
                            |_, _| sigma,
                            t0,
                            t1,
                            |_| erosion,
                            EDGE_HIGH,
                            &sample,
                        );

                        let dt = (t1 - t0) / FLAME_OCCUPANCY_NODE_SEGMENTS as f32;
                        let steps = 40000;
                        let ds = (t1 - t0) as f64 / steps as f64;
                        let mut reference = 0.0f64;
                        let mut reference_first = 0.0f64;
                        for i in 0..steps {
                            let t = t0 as f64 + (i as f64 + 0.5) * ds;
                            let segment = (((t - t0 as f64) / dt as f64).floor() as usize)
                                .min(FLAME_OCCUPANCY_NODE_SEGMENTS - 1);
                            let ta = t0 + segment as f32 * dt;
                            let da = sample(ta);
                            let db = sample(ta + dt);
                            if da <= 0.0 && db <= 0.0 {
                                continue;
                            }
                            let arg_a = eroded_argument(da, erosion, EDGE_HIGH);
                            let arg_b = eroded_argument(db, erosion, EDGE_HIGH);
                            let x = arg_a + (arg_b - arg_a) * ((t as f32 - ta) / dt);
                            let sigma_eff = sigma
                                * 0.5
                                * (envelope_fade(da, EDGE_HIGH) + envelope_fade(db, EDGE_HIGH));
                            let value = evaluate_erf_response(&model, x, sigma_eff) as f64;
                            reference += value * ds;
                            reference_first += t * value * ds;
                        }

                        let scale = (t1 - t0) as f64;
                        assert!(
                            (integral as f64 - reference).abs() < 3e-3 * scale.max(1e-3),
                            "dir={direction:?} band={band} sigma={sigma} e={erosion} \
                             scale={boundary_scale:?}: got {integral}, expected {reference}"
                        );
                        assert!(
                            (first_moment as f64 - reference_first).abs()
                                < 3e-3 * scale.max(1e-3) * (t1.abs().max(1.0)) as f64,
                            "dir={direction:?} band={band} first moment: got {first_moment}, \
                             expected {reference_first}"
                        );
                    }
                }
            }
        }

        #[test]
        fn test_node_band_with_per_node_erosion_matches_quadrature() {
            let model = default_model();
            let coefficients = FlameCoefficients::default();
            let height = build_height_series(&coefficients.height);
            let taper = default_taper();
            let (o, d, t0, t1) = ([-1.4f32, 0.1, 0.0], [1.0f32, 0.05, 0.0], 0.0f32, 2.8f32);
            let sample = |t: f32| {
                let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                evaluate_ring_smooth_density(p, &height, taper, 4.0, 0.75, 1.0)
            };
            let erosion_at = |t: f32| 0.25 * (1.0 - (t - 1.2) * (t - 1.2) / 0.09).max(0.0) - 0.05;
            let (integral, _) =
                evaluate_occupancy_node_band(&model, |_, _| 0.0, t0, t1, erosion_at, EDGE_HIGH, sample);
            let dt = (t1 - t0) / FLAME_OCCUPANCY_NODE_SEGMENTS as f32;
            let steps = 40000;
            let ds = (t1 - t0) as f64 / steps as f64;
            let mut reference = 0.0f64;
            for i in 0..steps {
                let t = t0 as f64 + (i as f64 + 0.5) * ds;
                let segment = (((t - t0 as f64) / dt as f64).floor() as usize)
                    .min(FLAME_OCCUPANCY_NODE_SEGMENTS - 1);
                let ta = t0 + segment as f32 * dt;
                let da = sample(ta);
                let db = sample(ta + dt);
                if da <= 0.0 && db <= 0.0 {
                    continue;
                }
                let arg_a = eroded_argument(da, erosion_at(ta), EDGE_HIGH);
                let arg_b = eroded_argument(db, erosion_at(ta + dt), EDGE_HIGH);
                let x = arg_a + (arg_b - arg_a) * ((t as f32 - ta) / dt);
                reference += evaluate_erf_response(&model, x, 0.0) as f64;
            }
            reference *= ds;
            assert!(
                (integral as f64 - reference).abs() < 3e-3 * (t1 - t0) as f64,
                "got {integral}, expected {reference}"
            );
        }

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
                let baseline = evaluate_ring_smooth_density(p, &height, taper, 4.0, 0.75, 1.1);
                let displaced = evaluate_ring_smooth_density_displaced(
                    p,
                    &height,
                    taper,
                    4.0,
                    0.75,
                    1.1,
                    [1.0, 1.0],
                );
                assert_eq!(baseline.to_bits(), displaced.to_bits(), "p={p:?}");
            }
        }

        #[test]
        fn test_node_band_matches_quadrature_with_boundary_displacement() {
            let model = default_model();
            let coefficients = FlameCoefficients::default();
            let height = build_height_series(&coefficients.height);
            let taper = default_taper();
            let boundary_at = |p: [f32; 3]| {
                [
                    1.0 + 0.35 * (1.7 * p[0] + 0.9 * p[2]).sin(),
                    1.0 + 0.18 * (1.3 * p[2] - 0.7 * p[0]).cos(),
                ]
            };
            let (o, d, t0, t1) = ([-1.4f32, 0.1, 0.0], [1.0f32, 0.05, 0.0], 0.0f32, 2.8f32);
            let sample = |t: f32| {
                let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                evaluate_ring_smooth_density_displaced(
                    p,
                    &height,
                    taper,
                    4.0,
                    0.75,
                    1.0,
                    boundary_at(p),
                )
            };
            let (integral, _) =
                evaluate_occupancy_node_band(&model, |_, _| 0.0, t0, t1, |_| 0.1, EDGE_HIGH, sample);
            let dt = (t1 - t0) / FLAME_OCCUPANCY_NODE_SEGMENTS as f32;
            let steps = 40000;
            let ds = (t1 - t0) as f64 / steps as f64;
            let mut reference = 0.0f64;
            for i in 0..steps {
                let t = t0 as f64 + (i as f64 + 0.5) * ds;
                let segment = (((t - t0 as f64) / dt as f64).floor() as usize)
                    .min(FLAME_OCCUPANCY_NODE_SEGMENTS - 1);
                let ta = t0 + segment as f32 * dt;
                let da = sample(ta);
                let db = sample(ta + dt);
                if da <= 0.0 && db <= 0.0 {
                    continue;
                }
                let arg_a = eroded_argument(da, 0.1, EDGE_HIGH);
                let arg_b = eroded_argument(db, 0.1, EDGE_HIGH);
                let x = arg_a + (arg_b - arg_a) * ((t as f32 - ta) / dt);
                reference += evaluate_erf_response(&model, x, 0.0) as f64;
            }
            reference *= ds;
            assert!(
                (integral as f64 - reference).abs() < 3e-3 * (t1 - t0) as f64,
                "got {integral}, expected {reference}"
            );
        }

        /// Against the true ring smoothstep field: node linearization plus fit floor
        /// stays a small fraction of the chord.
        #[test]
        fn test_node_band_tracks_true_ring_field() {
            let model = default_model();
            let coefficients = FlameCoefficients::default();
            let height = build_height_series(&coefficients.height);
            let taper = default_taper();
            let (o, d, t0, t1) = ([-1.4f32, 0.1, 0.0], [1.0f32, 0.05, 0.0], 0.0f32, 2.8f32);
            let sample = |t: f32| {
                let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                evaluate_ring_smooth_density(p, &height, taper, 4.0, 0.75, 1.0)
            };
            // The production path splits the segment into FLAME_RADIAL_BAND_COUNT bands
            // before node sampling; a single band would leave ring walls under-resolved.
            let band_width = (t1 - t0) / FLAME_RADIAL_BAND_COUNT as f32;
            let mut integral = 0.0f32;
            for band in 0..FLAME_RADIAL_BAND_COUNT {
                let band_start = t0 + band as f32 * band_width;
                integral += evaluate_occupancy_node_band(
                    &model,
                    |_, _| 0.0,
                    band_start,
                    band_start + band_width,
                    |_| 0.0,
                    EDGE_HIGH,
                    sample,
                )
                .0;
            }
            let steps = 40000;
            let ds = (t1 - t0) as f64 / steps as f64;
            let mut reference = 0.0f64;
            for i in 0..steps {
                let t = t0 as f64 + (i as f64 + 0.5) * ds;
                let d_smooth = sample(t as f32) as f64;
                if d_smooth <= 0.0 {
                    continue;
                }
                let t01 =
                    ((d_smooth - EDGE_LOW as f64) / (EDGE_HIGH - EDGE_LOW) as f64).clamp(0.0, 1.0);
                reference += t01 * t01 * (3.0 - 2.0 * t01) * ds;
            }
            assert!(
                (integral as f64 - reference).abs() < 0.12 * (t1 - t0) as f64,
                "got {integral}, expected {reference}"
            );
        }

        #[test]
        fn test_node_band_outside_support_is_zero() {
            let model = default_model();
            let (integral, first_moment) =
                evaluate_occupancy_node_band(&model, |_, _| 0.0, 0.0, 1.0, |_| -2.0, EDGE_HIGH, |_| 0.0);
            assert_eq!(integral, 0.0);
            assert_eq!(first_moment, 0.0);
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
                let span = ring_support_span(o, d, t0, t1, ring_major, taper, SHARPNESS, 1.0);
                if let Some((lo, hi)) = span {
                    assert!(lo >= t0 - 1e-5 && hi <= t1 + 1e-5, "span out of range");
                }
                let steps = 4000;
                for i in 0..steps {
                    let t = t0 + (i as f32 + 0.5) * (t1 - t0) / steps as f32;
                    let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                    let density =
                        evaluate_ring_smooth_density(p, &height, taper, SHARPNESS, ring_major, 1.0);
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

        #[test]
        fn test_density_weighted_node_t_follows_the_mass() {
            let uniform = density_weighted_node_t(0.0, 1.0, |_| 0.5).unwrap();
            assert!((uniform - 0.5).abs() < 1e-6, "uniform density -> midpoint");
            let concentrated =
                density_weighted_node_t(0.0, 1.0, |t| if t > 0.9 { 1.0 } else { 0.0 }).unwrap();
            assert!((concentrated - 1.0).abs() < 1e-6, "mass at the end node");
            assert!(density_weighted_node_t(0.0, 1.0, |_| 0.0).is_none());
            assert!(density_weighted_node_t(1.0, 0.0, |_| 1.0).is_none());
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

        /// The point of the sigma boundary fade: where the envelope sits far below
        /// the threshold (top of the flame, outskirts of the support), large
        /// unresolved noise must not manufacture occupancy — a flat band sigma left
        /// phi_sigma with a ~0.1-0.2 floor there, sliced off at the clipped support
        /// surface as a flat swirling ceiling.
        #[test]
        fn test_large_sigma_leaves_no_floor_where_envelope_is_low() {
            let model = default_model();
            let (a, b, c, half) = (4.0f32, 0.0f32, 0.0f32, 0.5f32);
            let f = (0.05f32, 0.0f32, 0.0f32);
            let (s_lo, s_hi) = solve_support_interval(a, b, c, -half, half).unwrap();
            let support = s_hi - s_lo;
            for sigma in [0.24f32, 0.3] {
                let (integral, _) = evaluate_occupancy_band(
                    &model,
                    |_, _| sigma,
                    a,
                    b,
                    c,
                    |s| occupancy_density(a, b, c, f, s),
                    half,
                    |_| 0.0,
                    EDGE_HIGH,
                    0.0,
                    &[],
                );
                assert!(
                    integral < 0.01 * support,
                    "sigma={sigma}: low-envelope band must stay empty, got {integral} over {support}"
                );

                let peak = 0.05f32;
                let (node_integral, _) = evaluate_occupancy_node_band(
                    &model,
                    |_, _| sigma,
                    0.0,
                    1.0,
                    |_| 0.0,
                    EDGE_HIGH,
                    |t: f32| peak * (1.0 - (2.0 * t - 1.0) * (2.0 * t - 1.0)).max(0.0),
                );
                assert!(
                    node_integral < 0.01,
                    "sigma={sigma}: low-envelope node band must stay empty, got {node_integral}"
                );
            }
        }

        /// The point of the residual floor: however hard the turbulence carves
        /// (erosion far above the envelope everywhere), a span that still has
        /// smooth density must keep a positive occupancy — a fully zero span is
        /// a hard hole in the Beer-Lambert composite. Without the floor the same
        /// band integrates to exactly zero.
        #[test]
        fn test_residual_prevents_a_fully_carved_hole() {
            let model = default_model();
            let (a, b, c, f, half) = (0.5f32, 0.2f32, 0.4f32, (0.8f32, 0.0f32, 0.0f32), 0.25f32);
            let density = |s: f32| occupancy_density(a, b, c, f, s);
            let (s_lo, s_hi) = solve_support_interval(a, b, c, -half, half).unwrap();
            let support = s_hi - s_lo;
            for erosion in [2.0f32, 10.0] {
                let bare = evaluate_occupancy_band(
                    &model,
                    |_, _| 0.0,
                    a,
                    b,
                    c,
                    density,
                    half,
                    |_| erosion,
                    EDGE_HIGH,
                    0.0,
                    &[],
                )
                .0;
                assert!(bare.abs() < 1e-4, "without the floor this is a hole: {bare}");
                for strength in [0.05f32, 0.12, 0.3] {
                    let floored = evaluate_occupancy_band(
                        &model,
                        |_, _| 0.0,
                        a,
                        b,
                        c,
                        density,
                        half,
                        |_| erosion,
                        EDGE_HIGH,
                        strength,
                        &[],
                    )
                    .0;
                    // The floor is strength * smoothstep(0, edge_high, density),
                    // so the occupancy sits between 0 and strength * support.
                    // The residual is beta times the un-eroded occupancy, so it
                    // sits strictly between zero and beta * support.
                    assert!(
                        floored > 0.05 * strength * support,
                        "strength={strength} e={erosion}: residual must survive, got {floored}"
                    );
                    assert!(
                        floored < strength * support * 1.01,
                        "strength={strength}: residual must not exceed its own ceiling, got {floored}"
                    );
                }
            }
        }

        /// The residual must vanish where the un-eroded density is itself below
        /// the threshold, so the silhouette and the support boundary keep their
        /// exact geometry (a floor on the raw density would haze the support).
        #[test]
        fn test_residual_vanishes_outside_the_support_and_below_threshold() {
            let model = default_model();
            let f = (0.8f32, 0.0f32, 0.0f32);
            let outside = evaluate_occupancy_band(
                &model,
                |_, _| 0.0,
                0.0,
                0.0,
                3.0,
                |s| occupancy_density(0.0, 0.0, 3.0, f, s),
                0.25,
                |_| 0.0,
                EDGE_HIGH,
                0.3,
                &[],
            );
            assert_eq!(outside.0, 0.0, "no support, no residual medium");

            // Sub-threshold envelope: the residual adds nothing visible.
            let faint = (0.05f32, 0.0f32, 0.0f32);
            let (a, b, c, half) = (4.0f32, 0.0f32, 0.0f32, 0.5f32);
            let (s_lo, s_hi) = solve_support_interval(a, b, c, -half, half).unwrap();
            let value = evaluate_occupancy_band(
                &model,
                |_, _| 0.0,
                a,
                b,
                c,
                |s| occupancy_density(a, b, c, faint, s),
                half,
                |_| 5.0,
                EDGE_HIGH,
                0.3,
                &[],
            )
            .0;
            assert!(
                value < 0.01 * (s_hi - s_lo),
                "sub-threshold density must stay dark, got {value}"
            );
        }

        /// Without erosion the mixture is the identity for any beta — the
        /// noise-free look is preserved exactly.
        #[test]
        fn test_residual_is_identity_without_erosion() {
            let model = default_model();
            let (a, b, c, f, half) = (0.5f32, 0.2f32, 0.4f32, (0.8f32, 0.3f32, -0.6f32), 0.25f32);
            let density = |s: f32| occupancy_density(a, b, c, f, s);
            let base = evaluate_occupancy_band(
                &model,
                |_, _| 0.0,
                a,
                b,
                c,
                density,
                half,
                |_| 0.0,
                EDGE_HIGH,
                0.0,
                &[],
            );
            for strength in [0.05f32, 0.12, 0.3, 1.0] {
                let mixed = evaluate_occupancy_band(
                    &model,
                    |_, _| 0.0,
                    a,
                    b,
                    c,
                    density,
                    half,
                    |_| 0.0,
                    EDGE_HIGH,
                    strength,
                    &[],
                );
                assert!(
                    (mixed.0 - base.0).abs() < 1e-6,
                    "beta={strength}: erosion-free field must be unchanged, {} vs {}",
                    mixed.0,
                    base.0
                );
            }
        }

        #[test]
        fn test_band_is_deterministic() {
            let model = default_model();
            let density = |s: f32| occupancy_density(0.5, 0.2, 0.4, (0.8, 0.3, -0.6), s);
            let first = evaluate_occupancy_band(
                &model,
                |_, _| 0.1,
                0.5,
                0.2,
                0.4,
                density,
                0.25,
                |_| 0.1,
                EDGE_HIGH,
                0.0,
                &[],
            );
            let second = evaluate_occupancy_band(
                &model,
                |_, _| 0.1,
                0.5,
                0.2,
                0.4,
                density,
                0.25,
                |_| 0.1,
                EDGE_HIGH,
                0.0,
                &[],
            );
            assert_eq!(first.0.to_bits(), second.0.to_bits());
            assert_eq!(first.1.to_bits(), second.1.to_bits());
        }
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
