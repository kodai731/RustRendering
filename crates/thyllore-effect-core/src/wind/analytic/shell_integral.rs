use crate::wind::WindTornadoEffect;
use cgmath::Vector3;

// Mirror of shaders/wind/include/wind_shell_field.glsl and wind_shell_integral.glsl.
//
// The density is a sum of compact-support polynomial shells in q = x^2 + z^2:
//   wall: B((q - P(h)) / W),  P(h) = (base + slope * h)^2
//   core: B(q / Pc)
// with B(u) = (1 - u^2)^2 on |u| < 1, scaled by the height envelope E(h).
// Along a ray q and h are quadratic and linear in the ray parameter, so every
// term is a polynomial and the optical depth of a piece between two knots is
// an exact power-rule integral in the piece-local variable sigma in [0, 1].

pub const WIND_MAX_KNOTS: usize = 12;
const POLY_TERMS: usize = 12;
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
}

impl WindShellParams {
    pub fn from_effect(effect: &WindTornadoEffect) -> Self {
        Self {
            height: effect.column_height.max(1e-3),
            wall_radius_base: effect.wall_radius_base,
            wall_radius_slope: effect.wall_radius_top - effect.wall_radius_base,
            wall_width_q: effect.wall_width_q.max(1e-4),
            wall_strength: effect.wall_strength,
            core_radius_sq: effect.core_radius * effect.core_radius,
            core_strength: effect.core_strength,
            top_fade: effect.top_fade.clamp(1e-3, 1.0),
            sigma_t: effect.density,
        }
    }

    fn wall_radius(&self, h: f32) -> f32 {
        self.wall_radius_base + self.wall_radius_slope * h
    }

    fn core_active(&self) -> bool {
        self.core_radius_sq > 1e-8 && self.core_strength > 0.0
    }

    fn fade_start(&self) -> f32 {
        1.0 - self.top_fade
    }
}

pub fn wind_envelope_radius(params: &WindShellParams, h: f32) -> f32 {
    params
        .wall_radius(h)
        .max(params.core_radius_sq.sqrt())
        .max(0.0)
        + params.wall_width_q.sqrt()
}

pub fn wind_envelope_height(params: &WindShellParams, h: f32) -> f32 {
    if !(0.0..=1.0).contains(&h) {
        return 0.0;
    }
    let fade_start = params.fade_start();
    if h <= fade_start {
        return 1.0;
    }
    let v = (h - fade_start) / params.top_fade;
    1.0 - v * v * (3.0 - 2.0 * v)
}

fn biweight(u: f32) -> f32 {
    let inside = (1.0 - u * u).max(0.0);
    inside * inside
}

pub fn wind_density_at(params: &WindShellParams, local: Vector3<f32>) -> f32 {
    let h = local.y / params.height;
    let envelope = wind_envelope_height(params, h);
    if envelope <= 0.0 {
        return 0.0;
    }
    let q = local.x * local.x + local.z * local.z;

    let wall_radius = params.wall_radius(h);
    let wall =
        params.wall_strength * biweight((q - wall_radius * wall_radius) / params.wall_width_q);
    let core = if params.core_active() {
        params.core_strength * biweight(q / params.core_radius_sq)
    } else {
        0.0
    };
    params.sigma_t * envelope * (wall + core)
}

/// Clamps the ray parameter interval to the cone frustum enclosing every shell
/// (radius linear in height between the envelope radii at the base and the top)
/// intersected with the height slab. Returns false when the interval is empty.
pub fn clamp_ray_to_wind_cone(
    params: &WindShellParams,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
    t_near: &mut f32,
    t_far: &mut f32,
) -> bool {
    let radius_base = wind_envelope_radius(params, 0.0);
    let radius_top = wind_envelope_radius(params, 1.0);
    let slope_per_unit_y = (radius_top - radius_base) / params.height;
    let m = radius_base + slope_per_unit_y * origin.y;
    let n = slope_per_unit_y * direction.y;
    let a = direction.x * direction.x + direction.z * direction.z - n * n;
    let b = 2.0 * (origin.x * direction.x + origin.z * direction.z - m * n);
    let c = origin.x * origin.x + origin.z * origin.z - m * m;

    if a.abs() < LINEAR_COEFFICIENT_EPSILON {
        if b.abs() < LINEAR_COEFFICIENT_EPSILON {
            if c > 0.0 {
                return false;
            }
        } else {
            let t_root = -c / b;
            if b > 0.0 {
                *t_far = t_far.min(t_root);
            } else {
                *t_near = t_near.max(t_root);
            }
        }
    } else {
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            if a > 0.0 {
                return false;
            }
        } else if a > 0.0 {
            let sqrt_discriminant = discriminant.sqrt();
            let t0 = (-b - sqrt_discriminant) / (2.0 * a);
            let t1 = (-b + sqrt_discriminant) / (2.0 * a);
            *t_near = t_near.max(t0.min(t1));
            *t_far = t_far.min(t0.max(t1));
        }
    }

    if direction.y.abs() < LINEAR_COEFFICIENT_EPSILON {
        if origin.y < 0.0 || origin.y > params.height {
            return false;
        }
    } else {
        let t_y0 = -origin.y / direction.y;
        let t_y1 = (params.height - origin.y) / direction.y;
        *t_near = t_near.max(t_y0.min(t_y1));
        *t_far = t_far.min(t_y0.max(t_y1));
    }

    *t_near <= *t_far
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
    let delta_c = q_c - radius_0 * radius_0;
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

    if direction.y.abs() >= LINEAR_COEFFICIENT_EPSILON {
        let fade_y = params.fade_start() * params.height;
        push_knot(
            &mut knots,
            &mut count,
            (fade_y - origin.y) / direction.y,
            t_near,
            t_far,
        );
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
    if !(0.0..=1.0).contains(&h_mid) {
        return 0.0;
    }

    let q0 = start.x * start.x + start.z * start.z;
    let q1 = 2.0 * length * (start.x * direction.x + start.z * direction.z);
    let q2 = length * length * (direction.x * direction.x + direction.z * direction.z);

    let radius_0 = params.wall_radius(h0);
    let radius_1 = params.wall_radius_slope * h1;
    let inv_width = 1.0 / params.wall_width_q;
    let u = poly_from_quadratic(
        (q0 - radius_0 * radius_0) * inv_width,
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

    let density = poly_mul(&envelope_poly(params, h0, h1, h_mid), &shell);
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
