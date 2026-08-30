use cgmath::{InnerSpace, Vector3};

#[derive(Clone, Copy, Debug)]
pub struct TorusHits {
    pub roots: [f32; 4],
    pub count: u8,
    pub fallback_used: bool,
}

pub fn torus_implicit(p: Vector3<f32>, r_hat: f32) -> f32 {
    let mag_sq = p.dot(p);
    (mag_sq + 1.0 - r_hat * r_hat).powi(2) - 4.0 * (p.x * p.x + p.z * p.z)
}

pub fn torus_gradient(p: Vector3<f32>, r_hat: f32) -> Vector3<f32> {
    let mag_sq = p.dot(p);
    let factor = 4.0 * (mag_sq + 1.0 - r_hat * r_hat);
    Vector3::new(
        factor * p.x - 8.0 * p.x,
        factor * p.y,
        factor * p.z - 8.0 * p.z,
    )
}

pub fn water_local_bounds(major_radius: f32, minor_radius: f32) -> (Vector3<f32>, Vector3<f32>) {
    let radius = major_radius + minor_radius;
    let min = Vector3::new(-radius, -minor_radius, -radius);
    let max = Vector3::new(radius, minor_radius, radius);
    (min, max)
}

pub fn water_local_bounds_corners(major_radius: f32, minor_radius: f32) -> [Vector3<f32>; 8] {
    let (min, max) = water_local_bounds(major_radius, minor_radius);
    let mut corners = [Vector3::new(0.0, 0.0, 0.0); 8];
    for (index, corner) in corners.iter_mut().enumerate() {
        corner.x = if index & 1 == 0 { min.x } else { max.x };
        corner.y = if index & 2 == 0 { min.y } else { max.y };
        corner.z = if index & 4 == 0 { min.z } else { max.z };
    }
    corners
}

fn quartic_coefficients(origin: Vector3<f64>, dir: Vector3<f64>, r_hat: f64) -> [f64; 5] {
    let ox = origin.x;
    let oy = origin.y;
    let oz = origin.z;
    let dx = dir.x;
    let dy = dir.y;
    let dz = dir.z;

    let coeff_a = dx * dx + dy * dy + dz * dz;
    let coeff_b = 2.0 * (ox * dx + oy * dy + oz * dz);
    let coeff_c = ox * ox + oy * oy + oz * oz;
    let coeff_d = coeff_c + 1.0 - r_hat * r_hat;

    let a4 = coeff_a * coeff_a;
    let a3 = 2.0 * coeff_a * coeff_b;
    let a2 = 2.0 * coeff_a * coeff_d + coeff_b * coeff_b - 4.0 * coeff_a + 4.0 * dy * dy;
    let a1 = 2.0 * coeff_b * coeff_d - 4.0 * coeff_b + 8.0 * oy * dy;
    let a0 = coeff_d * coeff_d - 4.0 * coeff_c + 4.0 * oy * oy;

    [a4, a3, a2, a1, a0]
}

fn eval_quartic(coeffs: &[f64; 5], t: f64) -> f64 {
    coeffs[4] + t * (coeffs[3] + t * (coeffs[2] + t * (coeffs[1] + t * coeffs[0])))
}

fn eval_quartic_derivative(coeffs: &[f64; 5], t: f64) -> f64 {
    coeffs[3] + t * (2.0 * coeffs[2] + t * (3.0 * coeffs[1] + t * 4.0 * coeffs[0]))
}

const EQUATION_EPSILON: f64 = 1e-9;

fn solve_quadratic(c: [f64; 3]) -> Vec<f64> {
    let p = c[1] / (2.0 * c[2]);
    let q = c[0] / c[2];
    let disc = p * p - q;
    if disc < -EQUATION_EPSILON {
        return Vec::new();
    }
    if disc.abs() < EQUATION_EPSILON {
        return vec![-p];
    }
    let sqrt_disc = disc.sqrt();
    vec![-p - sqrt_disc, -p + sqrt_disc]
}

fn solve_cubic(c: [f64; 4]) -> Vec<f64> {
    let cubic_a = c[2] / c[3];
    let cubic_b = c[1] / c[3];
    let cubic_c = c[0] / c[3];
    let sq_a = cubic_a * cubic_a;
    let p = (1.0 / 3.0) * (-(1.0 / 3.0) * sq_a + cubic_b);
    let q =
        (1.0 / 2.0) * ((2.0 / 27.0) * cubic_a * sq_a - (1.0 / 3.0) * cubic_a * cubic_b + cubic_c);
    let cb_p = p * p * p;
    let cubic_d = q * q + cb_p;

    let mut roots: Vec<f64>;

    if cubic_d.abs() < EQUATION_EPSILON {
        if q.abs() < EQUATION_EPSILON {
            roots = vec![0.0];
        } else {
            let u = (-q).cbrt();
            roots = vec![2.0 * u, -u];
        }
    } else if cubic_d < 0.0 {
        let phi = (1.0 / 3.0) * (-q / (-cb_p).sqrt()).acos();
        let t = 2.0 * (-p).sqrt();
        roots = vec![
            t * phi.cos(),
            -t * (phi + std::f64::consts::PI / 3.0).cos(),
            -t * (phi - std::f64::consts::PI / 3.0).cos(),
        ];
    } else {
        let sqrt_disc = cubic_d.sqrt();
        let u = (sqrt_disc - q).cbrt();
        let v = -(sqrt_disc + q).cbrt();
        roots = vec![u + v];
    }

    for root in &mut roots {
        *root -= cubic_a / 3.0;
    }
    roots
}

fn select_resolvent_root(cubic_roots: &[f64], p: f64, r: f64) -> f64 {
    let decomposition_margin = |z: f64| (z * z - r).min(2.0 * z - p);
    cubic_roots
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |best, z| {
            if decomposition_margin(z) > decomposition_margin(best) {
                z
            } else {
                best
            }
        })
}

fn non_negative_sqrt(value: f64) -> Option<f64> {
    if value > 0.0 {
        Some(value.sqrt())
    } else if value > -EQUATION_EPSILON {
        Some(0.0)
    } else {
        None
    }
}

fn solve_quartic(c: [f64; 5]) -> Vec<f64> {
    let quartic_a = c[3] / c[4];
    let quartic_b = c[2] / c[4];
    let quartic_c = c[1] / c[4];
    let quartic_d = c[0] / c[4];
    let quartic_sq_a = quartic_a * quartic_a;
    let p = -(3.0 / 8.0) * quartic_sq_a + quartic_b;
    let q =
        (1.0 / 8.0) * quartic_sq_a * quartic_a - (1.0 / 2.0) * quartic_a * quartic_b + quartic_c;
    let r = -(3.0 / 256.0) * quartic_sq_a * quartic_sq_a + (1.0 / 16.0) * quartic_sq_a * quartic_b
        - (1.0 / 4.0) * quartic_a * quartic_c
        + quartic_d;

    let mut roots: Vec<f64>;

    if r.abs() < EQUATION_EPSILON {
        let cubic_roots = solve_cubic([q, p, 0.0, 1.0]);
        roots = [0.0].into_iter().chain(cubic_roots).collect();
    } else if q.abs() < EQUATION_EPSILON {
        roots = solve_quadratic([r, p, 1.0])
            .into_iter()
            .filter(|square| *square >= 0.0)
            .flat_map(|square| {
                let root = square.sqrt();
                [root, -root]
            })
            .collect();
    } else {
        let cubic_roots = solve_cubic([r * p / 2.0 - q * q / 8.0, -r, -p / 2.0, 1.0]);
        let z = select_resolvent_root(&cubic_roots, p, r);

        let (u, v) = match (non_negative_sqrt(z * z - r), non_negative_sqrt(2.0 * z - p)) {
            (Some(u), Some(v)) => (u, v),
            _ => return Vec::new(),
        };

        let sign = if q < 0.0 { -v } else { v };
        let mut r1 = solve_quadratic([z - u, sign, 1.0]);
        let sign2 = if q < 0.0 { v } else { -v };
        let r2 = solve_quadratic([z + u, sign2, 1.0]);
        r1.extend(r2);
        roots = r1;
    }

    for root in &mut roots {
        *root -= quartic_a / 4.0;
    }
    roots
}

fn solve_quartic_ferrari(coefficients_high_to_low: [f64; 5]) -> ([f64; 4], usize) {
    let mut ascending = coefficients_high_to_low;
    ascending.reverse();
    let roots = solve_quartic(ascending);
    let mut out = [0.0; 4];
    let count = roots.len();
    for (i, r) in roots.into_iter().enumerate() {
        out[i] = r;
    }
    (out, count)
}

fn newton_refine(root: &mut f64, coeffs: &[f64; 5], iterations: usize) {
    for _ in 0..iterations {
        let f = eval_quartic(coeffs, *root);
        let df = eval_quartic_derivative(coeffs, *root);
        let correction = f / (df + 1e-30);
        if correction.abs() < 1e-14 {
            break;
        }
        *root -= correction;
    }
}

fn sphere_tracing(origin: Vector3<f64>, dir: Vector3<f64>, r_hat: f64) -> Option<f64> {
    let mut t = 0.0f64;
    for _ in 0..128 {
        let px = origin.x + dir.x * t;
        let py = origin.y + dir.y * t;
        let pz = origin.z + dir.z * t;
        let xz_mag = (px * px + pz * pz).sqrt();
        let sdf = ((xz_mag - 1.0) * (xz_mag - 1.0) + py * py).sqrt() - r_hat;
        if sdf.abs() < 1e-6 {
            return Some(t);
        }
        t += sdf.min(1.0);
        if t > 1e6 {
            break;
        }
    }
    None
}

pub fn intersect_torus(
    origin: Vector3<f32>,
    dir: Vector3<f32>,
    major_radius: f32,
    minor_radius: f32,
) -> TorusHits {
    let r_hat = minor_radius / major_radius;
    let bounding_radius: f64 = 1.0 + r_hat as f64;

    let o_norm = Vector3::new(
        origin.x as f64 / major_radius as f64,
        origin.y as f64 / major_radius as f64,
        origin.z as f64 / major_radius as f64,
    );
    let d_unit = dir.normalize();
    let d_norm = Vector3::new(d_unit.x as f64, d_unit.y as f64, d_unit.z as f64);

    let o_mag_sq = o_norm.dot(o_norm);
    let oc = o_norm.dot(d_norm);
    let disc = oc * oc - (o_mag_sq - bounding_radius * bounding_radius);
    if disc < 0.0 && o_mag_sq > bounding_radius * bounding_radius {
        return TorusHits {
            roots: [0.0; 4],
            count: 0,
            fallback_used: false,
        };
    }

    let coeffs = quartic_coefficients(o_norm, d_norm, r_hat as f64);
    let (mut roots, count) = solve_quartic_ferrari(coeffs);

    for i in 0..count {
        newton_refine(&mut roots[i], &coeffs, 2);
    }

    let mut valid_count: usize = 0;
    for i in 0..count {
        if roots[i] > 1e-6 {
            roots[valid_count] = roots[i];
            valid_count += 1;
        }
    }

    let mut fallback_used = false;
    if valid_count == 0 {
        if let Some(fallback_t) = sphere_tracing(o_norm, d_norm, r_hat as f64) {
            roots[0] = fallback_t;
            valid_count = 1;
            fallback_used = true;
        }
    }

    for i in 1..valid_count {
        let mut j = i;
        while j > 0 && roots[j - 1] > roots[j] {
            roots.swap(j - 1, j);
            j -= 1;
        }
    }

    let mut final_roots = [0.0f32; 4];
    for i in 0..valid_count {
        final_roots[i] = (roots[i] * major_radius as f64) as f32;
    }

    TorusHits {
        roots: final_roots,
        count: valid_count as u8,
        fallback_used,
    }
}
