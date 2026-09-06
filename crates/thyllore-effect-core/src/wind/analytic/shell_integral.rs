use crate::wind::analytic::motion::{h_top, spread_offset, streak_phase, wall_amp};
use crate::wind::WindTornadoEffect;
use cgmath::Vector3;

// Mirror of shaders/wind/include/wind_shell_field.glsl and wind_shell_integral.glsl.
//
// The density is a sum of compact-support polynomial shells in q = x^2 + z^2:
//   wall: B((q - P(h)) / W),  P(h) = (base + slope * h)^2
//   core: B(q / Pc)
//   ring: E_r(h / Hr) * B((q - Pr) / Wr)
// with B(u) = (1 - u^2)^2 on |u| < 1, scaled by the height envelope E(h).
// Along a ray q and h are quadratic and linear in the ray parameter, so every
// term is a polynomial and the optical depth of a piece between two knots is
// an exact power-rule integral in the piece-local variable sigma in [0, 1].

pub const WIND_MAX_KNOTS: usize = 16;
const POLY_TERMS: usize = 16;
const LINEAR_COEFFICIENT_EPSILON: f32 = 1e-7;
const EMPTY_INTERVAL_EPSILON: f32 = 1e-6;

type Poly = [f32; POLY_TERMS];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindShellParams {
    pub height: f32,
    pub wall_radius_base: f32,
    pub wall_radius_slope: f32,
    pub wall_width_q: f32,
    pub wall_strength: f32,
    pub core_radius_sq: f32,
    pub core_strength: f32,
    pub top_fade: f32,
    pub sigma_t: f32,
    pub h_top: f32,
    pub spread_offset: f32,
    pub ring_height: f32,
    pub ring_radius_sq: f32,
    pub ring_width_q: f32,
    pub ring_strength: f32,
    pub streak_order: f32,
    pub streak_twist: f32,
    pub streak_rise_speed: f32,
    pub streak_amplitude: f32,
    pub streak_phase: f32,
    pub streak_rise_time: f32,
}

impl WindShellParams {
    pub fn from_effect(effect: &WindTornadoEffect) -> Self {
        let t = effect.time;
        let h_top_value = h_top(t, effect.rise_initial_height, effect.rise_duration);
        let spread_offset_value = spread_offset(t, effect.spread_start, effect.spread_rate);
        let wall_strength = wall_amp(
            t,
            effect.wall_strength,
            effect.dissipate_start,
            effect.dissipate_time,
        );
        let ring_strength = wall_amp(
            t,
            effect.ring_strength,
            effect.dissipate_start,
            effect.dissipate_time,
        );
        let ring_radius_sq = effect.ring_radius * effect.ring_radius
            + spread_offset(t, effect.spread_start, effect.ring_spread_rate);
        let streak_phase_value = streak_phase(
            t,
            effect.circulation,
            effect.wall_radius_base,
            effect.spread_start,
            effect.spread_rate,
        );
        Self {
            height: effect.column_height.max(1e-3),
            wall_radius_base: effect.wall_radius_base,
            wall_radius_slope: effect.wall_radius_top - effect.wall_radius_base,
            wall_width_q: effect.wall_width_q.max(1e-4),
            wall_strength,
            core_radius_sq: effect.core_radius * effect.core_radius,
            core_strength: effect.core_strength,
            top_fade: effect.top_fade.clamp(1e-3, 1.0),
            sigma_t: effect.density,
            h_top: h_top_value.max(1e-3),
            spread_offset: spread_offset_value,
            ring_height: effect.ring_height.max(1e-3),
            ring_radius_sq,
            ring_width_q: effect.ring_width_q.max(1e-4),
            ring_strength,
            streak_order: effect.streak_order,
            streak_twist: effect.streak_twist,
            streak_rise_speed: effect.streak_rise_speed,
            streak_amplitude: effect.streak_amplitude.max(0.0),
            streak_phase: streak_phase_value,
            streak_rise_time: effect.streak_rise_speed * t,
        }
    }

    fn wall_radius(&self, h: f32) -> f32 {
        self.wall_radius_base + self.wall_radius_slope * h
    }

    fn wall_radius_sq(&self, h: f32) -> f32 {
        let radius = self.wall_radius(h);
        radius * radius + self.spread_offset
    }

    fn core_active(&self) -> bool {
        self.core_radius_sq > 1e-8 && self.core_strength > 0.0
    }

    fn ring_active(&self) -> bool {
        self.ring_strength > 0.0
    }

    pub fn ring_bounds_radius(&self) -> f32 {
        if self.ring_active() {
            (self.ring_radius_sq + self.ring_width_q).max(0.0).sqrt()
        } else {
            0.0
        }
    }

    fn ring_top_y(&self) -> f32 {
        self.ring_height * self.height
    }

    fn fade_start(&self) -> f32 {
        1.0 - self.top_fade
    }
}

pub fn wind_envelope_radius(params: &WindShellParams, h: f32) -> f32 {
    params
        .wall_radius_sq(h)
        .max(params.core_radius_sq)
        .max(0.0)
        .sqrt()
        + params.wall_width_q.sqrt()
}

pub fn wind_envelope_height(params: &WindShellParams, h: f32) -> f32 {
    if h < 0.0 || h > params.h_top {
        return 0.0;
    }
    let normalized_height = h / params.h_top;
    let fade_start = params.fade_start();
    if normalized_height <= fade_start {
        return 1.0;
    }
    let v = (normalized_height - fade_start) / params.top_fade;
    1.0 - v * v * (3.0 - 2.0 * v)
}

fn biweight(u: f32) -> f32 {
    let inside = (1.0 - u * u).max(0.0);
    inside * inside
}

fn ring_fade(v: f32) -> f32 {
    if v >= 1.0 {
        return 0.0;
    }
    1.0 - v * v * (3.0 - 2.0 * v)
}

pub fn wind_density_at(params: &WindShellParams, local: Vector3<f32>) -> f32 {
    let h = local.y / params.height;
    let envelope = wind_envelope_height(params, h);
    if envelope <= 0.0 {
        return 0.0;
    }
    let q = local.x * local.x + local.z * local.z;

    let wall =
        params.wall_strength * biweight((q - params.wall_radius_sq(h)) / params.wall_width_q);
    let core = if params.core_active() {
        params.core_strength * biweight(q / params.core_radius_sq)
    } else {
        0.0
    };
    let ring = if params.ring_active() {
        params.ring_strength
            * ring_fade(h / params.ring_height)
            * biweight((q - params.ring_radius_sq) / params.ring_width_q)
    } else {
        0.0
    };
    params.sigma_t * envelope * (wall + core + ring)
}

fn clamp_ray_to_cone_frustum(
    radius_base: f32,
    radius_top: f32,
    top_y: f32,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
    bounds: (f32, f32),
) -> Option<(f32, f32)> {
    let (mut t_near, mut t_far) = bounds;
    let slope_per_unit_y = (radius_top - radius_base) / top_y;
    let m = radius_base + slope_per_unit_y * origin.y;
    let n = slope_per_unit_y * direction.y;
    let a = direction.x * direction.x + direction.z * direction.z - n * n;
    let b = 2.0 * (origin.x * direction.x + origin.z * direction.z - m * n);
    let c = origin.x * origin.x + origin.z * origin.z - m * m;

    if a.abs() < LINEAR_COEFFICIENT_EPSILON {
        if b.abs() < LINEAR_COEFFICIENT_EPSILON {
            if c > 0.0 {
                return None;
            }
        } else {
            let t_root = -c / b;
            if b > 0.0 {
                t_far = t_far.min(t_root);
            } else {
                t_near = t_near.max(t_root);
            }
        }
    } else {
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            if a > 0.0 {
                return None;
            }
        } else if a > 0.0 {
            let sqrt_discriminant = discriminant.sqrt();
            let t0 = (-b - sqrt_discriminant) / (2.0 * a);
            let t1 = (-b + sqrt_discriminant) / (2.0 * a);
            t_near = t_near.max(t0.min(t1));
            t_far = t_far.min(t0.max(t1));
        }
    }

    if direction.y.abs() < LINEAR_COEFFICIENT_EPSILON {
        if origin.y < 0.0 || origin.y > top_y {
            return None;
        }
    } else {
        let t_y0 = -origin.y / direction.y;
        let t_y1 = (top_y - origin.y) / direction.y;
        t_near = t_near.max(t_y0.min(t_y1));
        t_far = t_far.min(t_y0.max(t_y1));
    }

    (t_near <= t_far).then_some((t_near, t_far))
}

/// Clamps the ray parameter interval to the hull of the wall cone frustum and the
/// ring frustum, both intersected with their height slab. False when both are missed.
pub fn clamp_ray_to_wind_cone(
    params: &WindShellParams,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
    t_near: &mut f32,
    t_far: &mut f32,
) -> bool {
    let bounds = (*t_near, *t_far);
    let wall = clamp_ray_to_cone_frustum(
        wind_envelope_radius(params, 0.0),
        wind_envelope_radius(params, params.h_top),
        params.h_top * params.height,
        origin,
        direction,
        bounds,
    );
    let ring = if params.ring_active() {
        let radius = params.ring_bounds_radius();
        clamp_ray_to_cone_frustum(
            radius,
            radius,
            params.ring_top_y(),
            origin,
            direction,
            bounds,
        )
    } else {
        None
    };

    match (wall, ring) {
        (None, None) => false,
        (Some(interval), None) | (None, Some(interval)) => {
            *t_near = interval.0;
            *t_far = interval.1;
            true
        }
        (Some(wall_interval), Some(ring_interval)) => {
            *t_near = wall_interval.0.min(ring_interval.0);
            *t_far = wall_interval.1.max(ring_interval.1);
            true
        }
    }
}

fn push_knot(knots: &mut [f32; WIND_MAX_KNOTS], count: &mut usize, t: f32, lo: f32, hi: f32) {
    if t <= lo || t >= hi || *count >= WIND_MAX_KNOTS {
        return;
    }
    knots[*count] = t;
    *count += 1;
}

fn push_quadratic_roots(
    a: f32,
    b: f32,
    c: f32,
    lo: f32,
    hi: f32,
    knots: &mut [f32; WIND_MAX_KNOTS],
    count: &mut usize,
) {
    if a.abs() < LINEAR_COEFFICIENT_EPSILON {
        if b.abs() >= LINEAR_COEFFICIENT_EPSILON {
            push_knot(knots, count, -c / b, lo, hi);
        }
        return;
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return;
    }
    let sqrt_discriminant = discriminant.sqrt();
    push_knot(knots, count, (-b - sqrt_discriminant) / (2.0 * a), lo, hi);
    push_knot(knots, count, (-b + sqrt_discriminant) / (2.0 * a), lo, hi);
}

fn sort_knots(knots: &mut [f32; WIND_MAX_KNOTS], count: usize) {
    for i in 1..count {
        let value = knots[i];
        let mut j = i;
        while j > 0 && knots[j - 1] > value {
            knots[j] = knots[j - 1];
            j -= 1;
        }
        knots[j] = value;
    }
}

/// Ray parameters where a shell support boundary or an envelope break is crossed,
/// sorted ascending and bracketed by `t_near` / `t_far`.
pub fn wind_ray_knots(
    params: &WindShellParams,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
    t_near: f32,
    t_far: f32,
) -> ([f32; WIND_MAX_KNOTS], usize) {
    let mut knots = [0.0f32; WIND_MAX_KNOTS];
    let mut count = 0usize;
    knots[0] = t_near;
    knots[1] = t_far;
    count += 2;

    let q_a = direction.x * direction.x + direction.z * direction.z;
    let q_b = 2.0 * (origin.x * direction.x + origin.z * direction.z);
    let q_c = origin.x * origin.x + origin.z * origin.z;

    let inv_height = 1.0 / params.height;
    let radius_0 = params.wall_radius(origin.y * inv_height);
    let radius_1 = params.wall_radius_slope * direction.y * inv_height;
    let delta_a = q_a - radius_1 * radius_1;
    let delta_b = q_b - 2.0 * radius_0 * radius_1;
    let delta_c = q_c - radius_0 * radius_0 - params.spread_offset;
    for boundary in [params.wall_width_q, -params.wall_width_q] {
        push_quadratic_roots(
            delta_a,
            delta_b,
            delta_c - boundary,
            t_near,
            t_far,
            &mut knots,
            &mut count,
        );
    }

    if params.core_active() {
        push_quadratic_roots(
            q_a,
            q_b,
            q_c - params.core_radius_sq,
            t_near,
            t_far,
            &mut knots,
            &mut count,
        );
    }

    if params.ring_active() {
        for boundary in [params.ring_width_q, -params.ring_width_q] {
            push_quadratic_roots(
                q_a,
                q_b,
                q_c - params.ring_radius_sq - boundary,
                t_near,
                t_far,
                &mut knots,
                &mut count,
            );
        }
    }

    if direction.y.abs() >= LINEAR_COEFFICIENT_EPSILON {
        let fade_y = params.fade_start() * params.h_top * params.height;
        push_knot(
            &mut knots,
            &mut count,
            (fade_y - origin.y) / direction.y,
            t_near,
            t_far,
        );
        if params.ring_active() {
            push_knot(
                &mut knots,
                &mut count,
                (params.ring_top_y() - origin.y) / direction.y,
                t_near,
                t_far,
            );
        }
    }

    sort_knots(&mut knots, count);
    (knots, count)
}

fn poly_mul(a: &Poly, b: &Poly) -> Poly {
    let mut product = [0.0f32; POLY_TERMS];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            if i + j >= POLY_TERMS {
                break;
            }
            product[i + j] += ai * bj;
        }
    }
    product
}

fn poly_from_quadratic(c0: f32, c1: f32, c2: f32) -> Poly {
    let mut poly = [0.0f32; POLY_TERMS];
    poly[0] = c0;
    poly[1] = c1;
    poly[2] = c2;
    poly
}

fn biweight_poly(u: &Poly) -> Poly {
    let mut inside = poly_mul(u, u);
    for coefficient in inside.iter_mut() {
        *coefficient = -*coefficient;
    }
    inside[0] += 1.0;
    poly_mul(&inside, &inside)
}

fn ring_fade_poly(params: &WindShellParams, h0: f32, h1: f32) -> Poly {
    let mut poly = [0.0f32; POLY_TERMS];
    let inv_ring_height = 1.0 / params.ring_height;
    let v0 = h0 * inv_ring_height;
    let v1 = h1 * inv_ring_height;
    poly[0] = 1.0 - 3.0 * v0 * v0 + 2.0 * v0 * v0 * v0;
    poly[1] = -6.0 * v0 * v1 + 6.0 * v0 * v0 * v1;
    poly[2] = -3.0 * v1 * v1 + 6.0 * v0 * v1 * v1;
    poly[3] = 2.0 * v1 * v1 * v1;
    poly
}

fn envelope_poly(params: &WindShellParams, h0: f32, h1: f32, h_mid: f32) -> Poly {
    let mut envelope = [0.0f32; POLY_TERMS];
    let fade_start = params.fade_start();
    if h_mid <= fade_start {
        envelope[0] = 1.0;
        return envelope;
    }
    let v0 = (h0 - fade_start) / params.top_fade;
    let v1 = h1 / params.top_fade;
    envelope[0] = 1.0 - 3.0 * v0 * v0 + 2.0 * v0 * v0 * v0;
    envelope[1] = -6.0 * v0 * v1 + 6.0 * v0 * v0 * v1;
    envelope[2] = -3.0 * v1 * v1 + 6.0 * v0 * v1 * v1;
    envelope[3] = 2.0 * v1 * v1 * v1;
    envelope
}

pub fn wind_streak_sigma(params: &WindShellParams, local: Vector3<f32>) -> f32 {
    let angle = params.streak_order * local.z.atan2(local.x)
        - params.streak_twist * local.y
        - params.streak_phase
        + params.streak_rise_time * local.y;
    1.0 + params.streak_amplitude * angle.cos()
}

/// Exact optical depth of the ray piece [s0, s1], which must not cross a knot.
pub fn wind_piece_optical_depth(
    params: &WindShellParams,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
    s0: f32,
    s1: f32,
) -> f32 {
    let length = s1 - s0;
    if length <= EMPTY_INTERVAL_EPSILON {
        return 0.0;
    }
    let start = origin + direction * s0;
    let inv_height = 1.0 / params.height;
    let h0 = start.y * inv_height;
    let h1 = length * direction.y * inv_height;
    let h_mid = h0 + 0.5 * h1;
    if !(0.0..=params.h_top).contains(&h_mid) {
        return 0.0;
    }

    let q0 = start.x * start.x + start.z * start.z;
    let q1 = 2.0 * length * (start.x * direction.x + start.z * direction.z);
    let q2 = length * length * (direction.x * direction.x + direction.z * direction.z);

    let radius_0 = params.wall_radius(h0);
    let radius_1 = params.wall_radius_slope * h1;
    let inv_width = 1.0 / params.wall_width_q;
    let u = poly_from_quadratic(
        (q0 - radius_0 * radius_0 - params.spread_offset) * inv_width,
        (q1 - 2.0 * radius_0 * radius_1) * inv_width,
        (q2 - radius_1 * radius_1) * inv_width,
    );
    let u_mid = u[0] + 0.5 * u[1] + 0.25 * u[2];

    let mut shell = [0.0f32; POLY_TERMS];
    if u_mid.abs() < 1.0 {
        let wall = biweight_poly(&u);
        for (target, value) in shell.iter_mut().zip(wall) {
            *target += params.wall_strength * value;
        }
    }
    if params.core_active() {
        let inv_core = 1.0 / params.core_radius_sq;
        let uc = poly_from_quadratic(q0 * inv_core, q1 * inv_core, q2 * inv_core);
        let uc_mid = uc[0] + 0.5 * uc[1] + 0.25 * uc[2];
        if uc_mid < 1.0 {
            let core = biweight_poly(&uc);
            for (target, value) in shell.iter_mut().zip(core) {
                *target += params.core_strength * value;
            }
        }
    }

    if params.ring_active() && h_mid < params.ring_height {
        let inv_ring_width = 1.0 / params.ring_width_q;
        let ur = poly_from_quadratic(
            (q0 - params.ring_radius_sq) * inv_ring_width,
            q1 * inv_ring_width,
            q2 * inv_ring_width,
        );
        let ur_mid = ur[0] + 0.5 * ur[1] + 0.25 * ur[2];
        if ur_mid.abs() < 1.0 {
            let ring = poly_mul(&ring_fade_poly(params, h0, h1), &biweight_poly(&ur));
            for (target, value) in shell.iter_mut().zip(ring) {
                *target += params.ring_strength * value;
            }
        }
    }

    let inv_h_top = 1.0 / params.h_top;
    let envelope = envelope_poly(params, h0 * inv_h_top, h1 * inv_h_top, h_mid * inv_h_top);
    let mut density = poly_mul(&envelope, &shell);

    if params.streak_amplitude > 0.0 {
        let sigma_0 = wind_streak_sigma(params, start);
        let sigma_1 = wind_streak_sigma(params, start + direction * length);

        let mut streak_poly = [0.0f32; POLY_TERMS];
        streak_poly[0] = sigma_0;
        streak_poly[1] = sigma_1 - sigma_0;

        density = poly_mul(&density, &streak_poly);
    }

    let mut moment_sum = 0.0f32;
    for (n, coefficient) in density.iter().enumerate() {
        moment_sum += coefficient / (n as f32 + 1.0);
    }
    (length * params.sigma_t * moment_sum).max(0.0)
}

pub fn wind_optical_depth(
    params: &WindShellParams,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
    t_near: f32,
    t_far: f32,
) -> f32 {
    if t_far <= t_near {
        return 0.0;
    }
    let (knots, count) = wind_ray_knots(params, origin, direction, t_near, t_far);
    let mut total = 0.0f32;
    for i in 1..count {
        total += wind_piece_optical_depth(params, origin, direction, knots[i - 1], knots[i]);
    }
    total
}
