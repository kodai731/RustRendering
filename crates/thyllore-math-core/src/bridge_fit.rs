//! C0 domain-split fit: bounded Chebyshev intervals joined by an erf bridge.
//!
//! The bake-side single source of truth for the sharpness-preserving integration
//! design (P3): a target with one sharp transition at `center` is approximated as
//!   [domain.0, window.0]  outer Chebyshev (Gauss-Lobatto nodes, exact endpoint values)
//!   [window.0, window.1]  augmented erf bridge
//!   [window.1, domain.1]  outer Chebyshev
//! with the joint values matched *constructively*: the bridge is fit by least squares
//! inside the null space of the two joint equality constraints, so the C0 property is
//! exact by construction and independent of fit quality.
//!
//! Window rule: half-width `delta = 4 / sharpness` (the effective erf width), clamped
//! so both outer intervals stay non-empty.
//! Bridge basis in normalized coordinates `v = (x - center) / delta`, `kv = sharpness * delta * v`:
//!   { 1, v, v^2, erf(kv), v exp(-kv^2), v exp(-(kv/2)^2) }
//! (constant + linear + curvature, the sharp step, and two localized Gaussian bumps
//! that absorb the smoothstep-minus-erf residual).

use crate::chebyshev::{
    chebyshev_endpoint_values, evaluate_chebyshev, fit_chebyshev_lobatto, ChebyshevSeries,
};
use crate::erf_moments::erf64;

pub const BRIDGE_BASIS_COUNT: usize = 6;
const BRIDGE_SAMPLE_COUNT: usize = 65;
const NULL_SPACE_DIMENSION: usize = BRIDGE_BASIS_COUNT - 2;

/// Bridge window `[center - delta, center + delta]` with `delta = 4 / sharpness`,
/// clamped to keep both outer Chebyshev intervals non-empty.
pub fn bridge_window(domain: (f64, f64), center: f64, sharpness: f64) -> (f64, f64) {
    assert!(
        domain.0 < center && center < domain.1,
        "center must be interior"
    );
    assert!(sharpness > 0.0, "sharpness must be positive");
    let natural = 4.0 / sharpness;
    let delta = natural
        .min(0.5 * (center - domain.0))
        .min(0.5 * (domain.1 - center));
    (center - delta, center + delta)
}

/// Bridge basis at normalized position `v = (x - center) / delta` in [-1, 1],
/// with `k_hat = sharpness * delta`.
pub fn evaluate_bridge_basis(v: f64, k_hat: f64) -> [f64; BRIDGE_BASIS_COUNT] {
    let kv = k_hat * v;
    let half_kv = 0.5 * kv;
    [
        1.0,
        v,
        v * v,
        erf64(kv),
        v * (-kv * kv).exp(),
        v * (-half_kv * half_kv).exp(),
    ]
}

/// C0 domain-split model: two bounded Chebyshev intervals and the erf bridge between.
#[derive(Debug, Clone)]
pub struct C0BridgeModel {
    pub lower: ChebyshevSeries,
    pub upper: ChebyshevSeries,
    pub bridge_weights: [f32; BRIDGE_BASIS_COUNT],
    pub window: (f32, f32),
    pub sharpness: f32,
}

impl C0BridgeModel {
    pub fn evaluate(&self, x: f32) -> f32 {
        if x < self.window.0 {
            return evaluate_chebyshev(&self.lower, x);
        }
        if x > self.window.1 {
            return evaluate_chebyshev(&self.upper, x);
        }
        let delta = 0.5 * (self.window.1 - self.window.0) as f64;
        let center = 0.5 * (self.window.0 + self.window.1) as f64;
        let v = if delta > 0.0 {
            (x as f64 - center) / delta
        } else {
            0.0
        };
        let basis = evaluate_bridge_basis(v, self.sharpness as f64 * delta);
        let mut total = 0.0f64;
        for (weight, value) in self.bridge_weights.iter().zip(basis.iter()) {
            total += *weight as f64 * value;
        }
        total as f32
    }
}

/// Solve `matrix * x = rhs` (n x n, in place) by Gaussian elimination with partial pivoting.
fn solve_dense(matrix: &mut [[f64; NULL_SPACE_DIMENSION]], rhs: &mut [f64]) {
    let n = rhs.len();
    for column in 0..n {
        let pivot_row = (column..n)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        matrix.swap(column, pivot_row);
        rhs.swap(column, pivot_row);
        let pivot = matrix[column][column];
        assert!(
            pivot.abs() > 1e-14,
            "singular normal equations in bridge fit"
        );
        for row in (column + 1)..n {
            let factor = matrix[row][column] / pivot;
            for k in column..n {
                matrix[row][k] -= factor * matrix[column][k];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    for column in (0..n).rev() {
        let mut value = rhs[column];
        for k in (column + 1)..n {
            value -= matrix[column][k] * rhs[k];
        }
        rhs[column] = value / matrix[column][column];
    }
}

/// Fit the C0 domain-split model. The two joint constraints (bridge value equals the
/// stored outer-series value at each window end) are enforced through null-space
/// parameterization, so they hold exactly regardless of the interior fit.
pub fn fit_c0_bridge(
    target: impl Fn(f64) -> f64,
    domain: (f64, f64),
    center: f64,
    sharpness: f64,
    outer_coefficient_count: usize,
) -> C0BridgeModel {
    let window = bridge_window(domain, center, sharpness);
    let delta = 0.5 * (window.1 - window.0);
    let k_hat = sharpness * delta;

    let lower = fit_chebyshev_lobatto(&target, (domain.0, window.0), outer_coefficient_count);
    let upper = fit_chebyshev_lobatto(&target, (window.1, domain.1), outer_coefficient_count);

    // Joint targets from the stored series (not the target function): C0 must hold
    // against what the outer intervals actually evaluate to.
    let (_, lower_joint) = chebyshev_endpoint_values(&lower);
    let (upper_joint, _) = chebyshev_endpoint_values(&upper);
    let constraint_rows = [
        evaluate_bridge_basis(-1.0, k_hat),
        evaluate_bridge_basis(1.0, k_hat),
    ];
    let constraint_values = [lower_joint as f64, upper_joint as f64];

    // Particular solution w0 = C^T (C C^T)^{-1} d.
    let mut gram = [[0.0f64; 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            gram[row][column] = constraint_rows[row]
                .iter()
                .zip(constraint_rows[column].iter())
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    let determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
    assert!(determinant.abs() > 1e-14, "degenerate joint constraints");
    let multipliers = [
        (gram[1][1] * constraint_values[0] - gram[0][1] * constraint_values[1]) / determinant,
        (gram[0][0] * constraint_values[1] - gram[1][0] * constraint_values[0]) / determinant,
    ];
    let mut particular = [0.0f64; BRIDGE_BASIS_COUNT];
    for (i, entry) in particular.iter_mut().enumerate() {
        *entry = multipliers[0] * constraint_rows[0][i] + multipliers[1] * constraint_rows[1][i];
    }

    // Orthonormal null-space basis of the 2 constraint rows by Gram-Schmidt against
    // the (orthonormalized) constraint rows and previously accepted columns.
    let mut frame: Vec<[f64; BRIDGE_BASIS_COUNT]> = Vec::with_capacity(2 + NULL_SPACE_DIMENSION);
    for row in &constraint_rows {
        let mut candidate = *row;
        orthogonalize(&mut candidate, &frame);
        let norm = norm(&candidate);
        assert!(norm > 1e-12, "degenerate joint constraints");
        normalize(&mut candidate, norm);
        frame.push(candidate);
    }
    let mut null_basis: Vec<[f64; BRIDGE_BASIS_COUNT]> = Vec::with_capacity(NULL_SPACE_DIMENSION);
    for axis in 0..BRIDGE_BASIS_COUNT {
        if null_basis.len() == NULL_SPACE_DIMENSION {
            break;
        }
        let mut candidate = [0.0f64; BRIDGE_BASIS_COUNT];
        candidate[axis] = 1.0;
        orthogonalize(&mut candidate, &frame);
        let candidate_norm = norm(&candidate);
        if candidate_norm > 1e-8 {
            normalize(&mut candidate, candidate_norm);
            frame.push(candidate);
            null_basis.push(candidate);
        }
    }
    assert_eq!(
        null_basis.len(),
        NULL_SPACE_DIMENSION,
        "null space of the joint constraints must have dimension 4"
    );

    // Least squares for the null-space coordinates against the window samples.
    let mut normal = [[0.0f64; NULL_SPACE_DIMENSION]; NULL_SPACE_DIMENSION];
    let mut rhs = [0.0f64; NULL_SPACE_DIMENSION];
    for sample in 0..BRIDGE_SAMPLE_COUNT {
        let v = -1.0 + 2.0 * sample as f64 / (BRIDGE_SAMPLE_COUNT - 1) as f64;
        let x = center + delta * v;
        let basis = evaluate_bridge_basis(v, k_hat);
        let particular_value: f64 = particular
            .iter()
            .zip(basis.iter())
            .map(|(w, b)| w * b)
            .sum();
        let residual = target(x) - particular_value;
        let mut projected = [0.0f64; NULL_SPACE_DIMENSION];
        for (j, direction) in null_basis.iter().enumerate() {
            projected[j] = direction.iter().zip(basis.iter()).map(|(d, b)| d * b).sum();
        }
        for j in 0..NULL_SPACE_DIMENSION {
            rhs[j] += projected[j] * residual;
            for k in 0..NULL_SPACE_DIMENSION {
                normal[j][k] += projected[j] * projected[k];
            }
        }
    }
    solve_dense(&mut normal, &mut rhs);

    let mut bridge_weights = [0.0f32; BRIDGE_BASIS_COUNT];
    for (i, weight) in bridge_weights.iter_mut().enumerate() {
        let mut value = particular[i];
        for (j, direction) in null_basis.iter().enumerate() {
            value += rhs[j] * direction[i];
        }
        *weight = value as f32;
    }

    C0BridgeModel {
        lower,
        upper,
        bridge_weights,
        window: (window.0 as f32, window.1 as f32),
        sharpness: sharpness as f32,
    }
}

fn orthogonalize(candidate: &mut [f64; BRIDGE_BASIS_COUNT], frame: &[[f64; BRIDGE_BASIS_COUNT]]) {
    for direction in frame {
        let projection: f64 = candidate
            .iter()
            .zip(direction.iter())
            .map(|(c, d)| c * d)
            .sum();
        for (entry, d) in candidate.iter_mut().zip(direction.iter()) {
            *entry -= projection * d;
        }
    }
}

fn norm(vector: &[f64; BRIDGE_BASIS_COUNT]) -> f64 {
    vector.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn normalize(vector: &mut [f64; BRIDGE_BASIS_COUNT], norm: f64) {
    for entry in vector.iter_mut() {
        *entry /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smooth_step(x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    }

    /// Sharp smoothstep transition at `center` of width `w` over a smooth background —
    /// the target family of the scratchpad piecewise_c0 experiments.
    fn sharp_target(x: f64, center: f64, width: f64) -> f64 {
        smooth_step((x - center) / width + 0.5) + 0.25 * (4.0 * x).sin()
    }

    const CENTER: f64 = 0.42;
    const WIDTH: f64 = 0.03;
    /// One-term erf fit of a smoothstep of width w: erf(k u) with k ~ 3.6 / w
    /// (experiment A: max err 0.0205 independent of w).
    const SHARPNESS: f64 = 3.6 / WIDTH;

    fn fitted_model() -> C0BridgeModel {
        fit_c0_bridge(
            |x| sharp_target(x, CENTER, WIDTH),
            (0.0, 1.0),
            CENTER,
            SHARPNESS,
            10,
        )
    }

    fn joint_jump(model: &C0BridgeModel, x: f32) -> f32 {
        let below = model.evaluate(x - 1e-6);
        let above = model.evaluate(x + 1e-6);
        (below - above).abs()
    }

    #[test]
    fn test_joints_are_c0_to_storage_precision() {
        let model = fitted_model();
        // f32 storage bounds the jump; the f64 construction itself is exact.
        assert!(joint_jump(&model, model.window.0) < 1e-5);
        assert!(joint_jump(&model, model.window.1) < 1e-5);
    }

    /// C0 is constructive: even a badly matched bridge sharpness must not open a jump.
    #[test]
    fn test_joints_stay_c0_with_mismatched_sharpness() {
        let model = fit_c0_bridge(
            |x| sharp_target(x, CENTER, WIDTH),
            (0.0, 1.0),
            CENTER,
            0.2 * SHARPNESS,
            10,
        );
        assert!(joint_jump(&model, model.window.0) < 1e-5);
        assert!(joint_jump(&model, model.window.1) < 1e-5);
    }

    #[test]
    fn test_outer_intervals_track_the_smooth_target() {
        let model = fitted_model();
        let mut max_error = 0.0f64;
        for i in 0..500 {
            let x = i as f64 / 499.0;
            if x >= model.window.0 as f64 - 1e-9 && x <= model.window.1 as f64 + 1e-9 {
                continue;
            }
            let error = (model.evaluate(x as f32) as f64 - sharp_target(x, CENTER, WIDTH)).abs();
            max_error = max_error.max(error);
        }
        assert!(max_error < 1e-4, "outer max error {max_error}");
    }

    /// Augmented bridge (erf + Gaussian corrections) target from the scratchpad
    /// experiments: interior error under 1e-2 across the window.
    #[test]
    fn test_bridge_interior_error_is_small() {
        let model = fitted_model();
        let mut max_error = 0.0f64;
        for i in 0..=400 {
            let x =
                model.window.0 as f64 + (model.window.1 - model.window.0) as f64 * i as f64 / 400.0;
            let error = (model.evaluate(x as f32) as f64 - sharp_target(x, CENTER, WIDTH)).abs();
            max_error = max_error.max(error);
        }
        assert!(max_error < 1e-2, "interior max error {max_error}");
    }

    #[test]
    fn test_window_rule_and_clamping() {
        let (lo, hi) = bridge_window((0.0, 1.0), 0.42, 100.0);
        assert!((hi - lo - 2.0 * 4.0 / 100.0).abs() < 1e-12);
        // Center near the domain edge: the window shrinks to keep outer intervals alive.
        let (lo, hi) = bridge_window((0.0, 1.0), 0.05, 10.0);
        assert!(lo > 0.0 && hi < 1.0);
        assert!((hi - lo - 2.0 * 0.025).abs() < 1e-12);
    }

    #[test]
    fn test_fit_is_deterministic() {
        let first = fitted_model();
        let second = fitted_model();
        let first_bits: Vec<u32> = first.bridge_weights.iter().map(|w| w.to_bits()).collect();
        let second_bits: Vec<u32> = second.bridge_weights.iter().map(|w| w.to_bits()).collect();
        assert_eq!(first_bits, second_bits);
    }
}
