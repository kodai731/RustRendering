use cgmath::Vector2;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

const GRID_N: usize = 64;
const CHEB_ORDER: usize = 8;

pub const LAPLACE_BELTRAMI_MODE_COUNT: usize = 4;
pub const LAPLACE_BELTRAMI_SLOTS_PER_MODE: usize = 5;

#[derive(Debug, Clone, Copy)]
pub struct LaplaceBeltramiMode {
    pub m: i32,
    pub lambda: f64,
    pub phi_cheb: [f32; CHEB_ORDER],
    pub dphi_cheb: [f32; CHEB_ORDER],
}

pub fn eval_cheb(coeffs: &[f32; CHEB_ORDER], t: f32) -> f32 {
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;

    for k in (1..CHEB_ORDER).rev() {
        let b0 = coeffs[k] + 2.0 * t * b1 - b2;
        b2 = b1;
        b1 = b0;
    }

    coeffs[0] + t * b1 - b2
}

/// Mirror of GLSL `evaluateChebyshev8`: `x01` lies in [0,1] and maps to the Clenshaw variable 2*x01-1.
fn eval_cheb8(lo: [f32; 4], hi: [f32; 4], x01: f32) -> f32 {
    let mut coeffs = [0.0f32; CHEB_ORDER];
    coeffs[..4].copy_from_slice(&lo);
    coeffs[4..].copy_from_slice(&hi);

    eval_cheb(&coeffs, 2.0 * x01 - 1.0)
}

/// Mirror of GLSL `waterLbHeightAndGradient`, evaluating the packed Laplace-Beltrami modes at (u, v).
/// Returns (h, h_u, h_v).
pub fn water_laplace_beltrami_height_and_gradient(
    uv: Vector2<f32>,
    time: f32,
    flow_rate: Vector2<f32>,
    laplace_beltrami_modes: &[[f32; 4];
         LAPLACE_BELTRAMI_MODE_COUNT * LAPLACE_BELTRAMI_SLOTS_PER_MODE],
) -> (f32, f32, f32) {
    let mut h = 0.0f32;
    let mut h_u = 0.0f32;
    let mut h_v = 0.0f32;

    for k in 0..LAPLACE_BELTRAMI_MODE_COUNT {
        let slot = LAPLACE_BELTRAMI_SLOTS_PER_MODE * k;
        let [m, omega, amplitude, phase] = laplace_beltrami_modes[slot];
        if amplitude <= 0.0 {
            continue;
        }

        let phase_prime = m * (uv.x + flow_rate.x * time) - omega * time + phase;
        let v_advected = (uv.y + flow_rate.y * time).rem_euclid(2.0 * std::f32::consts::PI);
        let t = (v_advected - std::f32::consts::PI) / std::f32::consts::PI;

        let phi = eval_cheb8(
            laplace_beltrami_modes[slot + 1],
            laplace_beltrami_modes[slot + 2],
            0.5 * t + 0.5,
        );
        let dphi = eval_cheb8(
            laplace_beltrami_modes[slot + 3],
            laplace_beltrami_modes[slot + 4],
            0.5 * t + 0.5,
        );

        h += amplitude * phase_prime.cos() * phi;
        h_u -= amplitude * m * phase_prime.sin() * phi;
        h_v += amplitude * phase_prime.cos() * dphi;
    }

    (h, h_u, h_v)
}

#[derive(Default)]
pub struct Cache {
    data: Mutex<HashMap<u32, [LaplaceBeltramiMode; 4]>>,
}

impl Cache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_compute(&self, major_radius: f32, minor_radius: f32) -> [LaplaceBeltramiMode; 4] {
        let key = ((major_radius / minor_radius) * 1000.0) as u32;

        let mut map = self.data.lock().unwrap();
        if let Some(modes) = map.get(&key) {
            return *modes;
        }

        let modes = compute_laplace_beltrami_modes(major_radius, minor_radius);
        map.insert(key, modes);
        modes
    }
}

pub static CACHE: LazyLock<Cache> = LazyLock::new(Cache::new);

pub fn compute_laplace_beltrami_modes_cached(
    major_radius: f32,
    minor_radius: f32,
) -> [LaplaceBeltramiMode; 4] {
    CACHE.get_or_compute(major_radius, minor_radius)
}

fn build_operator(m: i32, r_major: f64, r_minor: f64, n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let h = 2.0 * std::f64::consts::PI / (n as f64);
    let mut a = vec![vec![0.0f64; n]; n];
    let mut b = vec![0.0f64; n];

    for i in 0..n {
        let v_i = i as f64 * h;
        let rho_i = r_major + r_minor * v_i.cos();
        let rho_ip_half = r_major + r_minor * (v_i + 0.5 * h).cos();
        let rho_im_half = r_major + r_minor * (v_i - 0.5 * h).cos();

        a[i][i] = (rho_ip_half + rho_im_half) / (h * h)
            + (m as f64) * (m as f64) * r_minor * r_minor / rho_i;
        a[i][(i + 1) % n] = -rho_ip_half / (h * h);
        a[i][(i + n - 1) % n] = -rho_im_half / (h * h);
        b[i] = r_minor * r_minor * rho_i;
    }

    (a, b)
}

fn find_largest_off_diagonal(a: &[Vec<f64>], n: usize) -> (usize, usize, f64) {
    let mut p = 0usize;
    let mut q = 1usize;
    let mut max_abs = 0.0f64;

    for i in 0..n {
        for j in (i + 1)..n {
            if a[i][j].abs() > max_abs {
                max_abs = a[i][j].abs();
                p = i;
                q = j;
            }
        }
    }

    (p, q, max_abs)
}

fn apply_jacobi_rotation(a: &mut [Vec<f64>], v: &mut [Vec<f64>], n: usize, p: usize, q: usize) {
    let a_pq = a[p][q];
    let theta = 0.5 * f64::atan2(2.0 * a_pq, a[q][q] - a[p][p]);
    let c = theta.cos();
    let s = theta.sin();

    let a_pp = a[p][p];
    let a_qq = a[q][q];
    a[p][p] = c * c * a_pp - 2.0 * s * c * a_pq + s * s * a_qq;
    a[q][q] = s * s * a_pp + 2.0 * s * c * a_pq + c * c * a_qq;
    a[p][q] = 0.0;
    a[q][p] = 0.0;

    for i in 0..n {
        if i != p && i != q {
            let a_ip = a[i][p];
            let a_iq = a[i][q];
            a[i][p] = c * a_ip - s * a_iq;
            a[p][i] = a[i][p];
            a[i][q] = s * a_ip + c * a_iq;
            a[q][i] = a[i][q];
        }

        let v_ip = v[i][p];
        let v_iq = v[i][q];
        v[i][p] = c * v_ip - s * v_iq;
        v[i][q] = s * v_ip + c * v_iq;
    }
}

fn jacobi_eigen(mat: &[Vec<f64>], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut a: Vec<Vec<f64>> = mat.iter().map(|row| row.to_vec()).collect();
    let mut v = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    let rotations_per_sweep = n * (n - 1) / 2;
    'sweeps: for _ in 0..100 {
        for _ in 0..rotations_per_sweep {
            let (p, q, max_abs) = find_largest_off_diagonal(&a, n);
            if max_abs < 1e-10 {
                break 'sweeps;
            }
            apply_jacobi_rotation(&mut a, &mut v, n, p, q);
        }
    }

    let eigenvalues = (0..n).map(|i| a[i][i]).collect();
    (eigenvalues, v)
}

fn least_squares_cheb(values: &[f64], t: &[f64]) -> [f32; CHEB_ORDER] {
    let mut normal = [[0.0f64; CHEB_ORDER]; CHEB_ORDER];
    let mut rhs = [0.0f64; CHEB_ORDER];

    for (value, &t_i) in values.iter().zip(t.iter()) {
        let mut basis = [0.0f64; CHEB_ORDER];
        basis[0] = 1.0;
        basis[1] = t_i;
        for k in 2..CHEB_ORDER {
            basis[k] = 2.0 * t_i * basis[k - 1] - basis[k - 2];
        }

        for k in 0..CHEB_ORDER {
            for l in 0..CHEB_ORDER {
                normal[k][l] += basis[k] * basis[l];
            }
            rhs[k] += basis[k] * value;
        }
    }

    solve_gauss_partial_pivot(normal, rhs).map(|v| v as f32)
}

fn solve_gauss_partial_pivot(
    matrix: [[f64; CHEB_ORDER]; CHEB_ORDER],
    rhs: [f64; CHEB_ORDER],
) -> [f64; CHEB_ORDER] {
    let mut aug = [[0.0f64; CHEB_ORDER + 1]; CHEB_ORDER];
    for k in 0..CHEB_ORDER {
        aug[k][..CHEB_ORDER].copy_from_slice(&matrix[k]);
        aug[k][CHEB_ORDER] = rhs[k];
    }

    for col in 0..CHEB_ORDER {
        let pivot_row = (col..CHEB_ORDER)
            .max_by(|&a, &b| aug[a][col].abs().total_cmp(&aug[b][col].abs()))
            .unwrap_or(col);
        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            continue;
        }

        for row in (col + 1)..CHEB_ORDER {
            let factor = aug[row][col] / pivot;
            for k in col..=CHEB_ORDER {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }

    let mut x = [0.0f64; CHEB_ORDER];
    for i in (0..CHEB_ORDER).rev() {
        if aug[i][i].abs() < 1e-15 {
            continue;
        }
        let mut sum = aug[i][CHEB_ORDER];
        for j in (i + 1)..CHEB_ORDER {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    x
}

fn solve_lowest_mode(m: i32, r_major: f64, r_minor: f64, n: usize) -> (f64, Vec<f64>) {
    let (a, b) = build_operator(m, r_major, r_minor, n);
    let sqrt_b: Vec<f64> = b.iter().map(|&v| v.sqrt()).collect();

    let mut symmetric = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            symmetric[i][j] = a[i][j] / (sqrt_b[i] * sqrt_b[j]);
        }
    }

    let (eigenvalues, eigenvectors) = jacobi_eigen(&symmetric, n);
    let lowest = (0..n)
        .min_by(|&i, &j| eigenvalues[i].total_cmp(&eigenvalues[j]))
        .unwrap_or(0);

    let mut phi: Vec<f64> = (0..n)
        .map(|i| eigenvectors[i][lowest] / sqrt_b[i])
        .collect();
    let max_abs = phi.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    if max_abs > 1e-15 {
        for value in &mut phi {
            *value /= max_abs;
        }
    }

    (eigenvalues[lowest], phi)
}

fn cheb_parameters(n: usize) -> Vec<f64> {
    let h = 2.0 * std::f64::consts::PI / (n as f64);
    (0..n)
        .map(|i| (i as f64 * h - std::f64::consts::PI) / std::f64::consts::PI)
        .collect()
}

pub fn compute_laplace_beltrami_modes(
    major_radius: f32,
    minor_radius: f32,
) -> [LaplaceBeltramiMode; 4] {
    let r_major = major_radius as f64;
    let r_minor = minor_radius as f64;
    let n = GRID_N;
    let h = 2.0 * std::f64::consts::PI / (n as f64);
    let t = cheb_parameters(n);

    std::array::from_fn(|idx| {
        let m = idx as i32 + 1;
        let (lambda, phi) = solve_lowest_mode(m, r_major, r_minor, n);

        let dphi: Vec<f64> = (0..n)
            .map(|i| (phi[(i + 1) % n] - phi[(i + n - 1) % n]) / (2.0 * h))
            .collect();

        LaplaceBeltramiMode {
            m,
            lambda,
            phi_cheb: least_squares_cheb(&phi, &t),
            dphi_cheb: least_squares_cheb(&dphi, &t),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigenpairs_satisfy_discrete_generalized_problem() {
        let (r_major, r_minor) = (1.0f64, 0.3f64);
        let modes = compute_laplace_beltrami_modes(r_major as f32, r_minor as f32);

        for mode in &modes {
            let (a, b) = build_operator(mode.m, r_major, r_minor, GRID_N);
            let (_, phi) = solve_lowest_mode(mode.m, r_major, r_minor, GRID_N);

            let mut max_residual = 0.0f64;
            let mut max_rhs = 0.0f64;
            for i in 0..GRID_N {
                let lhs: f64 = (0..GRID_N).map(|j| a[i][j] * phi[j]).sum();
                let rhs = mode.lambda * b[i] * phi[i];
                max_residual = max_residual.max((lhs - rhs).abs());
                max_rhs = max_rhs.max(rhs.abs());
            }

            let relative = max_residual / max_rhs;
            assert!(
                relative < 1e-4,
                "m={} relative residual {relative:.3e}",
                mode.m
            );
        }
    }

    #[test]
    fn eigenvalues_approach_thin_torus_limit() {
        let r_major = 100.0f64;
        let modes = compute_laplace_beltrami_modes(r_major as f32, 0.3);

        for mode in &modes {
            let expected = (mode.m as f64).powi(2) / (r_major * r_major);
            let relative = (mode.lambda - expected).abs() / expected;
            assert!(
                relative < 0.05,
                "m={} lambda={:.6e} expected={expected:.6e} relative={relative:.4}",
                mode.m,
                mode.lambda
            );
        }
    }

    #[test]
    fn chebyshev_fit_reconstructs_grid_samples() {
        let (r_major, r_minor) = (1.0f64, 0.3f64);
        let modes = compute_laplace_beltrami_modes(r_major as f32, r_minor as f32);
        let t = cheb_parameters(GRID_N);

        for mode in &modes {
            let (_, phi) = solve_lowest_mode(mode.m, r_major, r_minor, GRID_N);
            let max_error = (0..GRID_N)
                .map(|i| (eval_cheb(&mode.phi_cheb, t[i] as f32) as f64 - phi[i]).abs())
                .fold(0.0f64, f64::max);

            assert!(max_error < 0.02, "m={} max error {max_error:.5}", mode.m);
        }
    }
}
