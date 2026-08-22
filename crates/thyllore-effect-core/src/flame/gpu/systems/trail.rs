use crate::flame::*;
use crate::flame_trail::{FlameTrailSample, FlameTrailState};
use cgmath::{Matrix3, Matrix4, Vector3};

/// Build the expanded model matrix for flame trail rendering.
/// Computes world AABB of all trail samples + effect position, then builds a rotation-free
/// expansion matrix using the same construction rules as build_flame_model_matrix.
pub fn build_flame_trail_expanded_matrix(
    effect: &FlameEffect,
    samples: &[FlameTrailSample],
) -> Matrix4<f32> {
    assert!(
        !samples.is_empty(),
        "build_flame_trail_expanded_matrix requires at least one sample"
    );

    // Compute world AABB of all samples + effect position
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut min_z = f32::MAX;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut max_z = f32::NEG_INFINITY;

    // Include effect position
    let ep = &effect.position;
    min_x = min_x.min(ep.x);
    max_x = max_x.max(ep.x);
    min_y = min_y.min(ep.y);
    max_y = max_y.max(ep.y);
    min_z = min_z.min(ep.z);
    max_z = max_z.max(ep.z);

    // Include all sample positions
    for s in samples {
        let p = &s.position;
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
        min_z = min_z.min(p[2]);
        max_z = max_z.max(p[2]);
    }
    // XZ center = AABB center
    let cx = (min_x + max_x) * 0.5;
    let cz = (min_z + max_z) * 0.5;

    // Extension radius = effect.radius + hypot(half_extent_x, half_extent_z)
    let half_extent_x = (max_x - min_x) * 0.5;
    let half_extent_z = (max_z - min_z) * 0.5;
    let extension_radius = flame_bounding_radius(effect)
        + (half_extent_x * half_extent_x + half_extent_z * half_extent_z).sqrt();

    // Extension height = effect.height + (max_y - min_y)
    let extension_height = effect.height + (max_y - min_y);

    // Base y = min_y
    let base_y = min_y;

    // Build rotation-free expansion matrix using same construction as build_flame_model_matrix
    Matrix4::from_translation(Vector3::new(cx, base_y, cz))
        * Matrix4::from_nonuniform_scale(extension_radius, extension_height, extension_radius)
}

/// Build trail UBO fields: (trailUnitInverse, trailMeta, trailCoefficients).
/// trailUnitInverse = inverse of the unit model matrix (without expansion).
/// For each sample i: localDelta_i = trailUnitInverse.linear_part * (sample.position - effect.position), w = fade weight.
/// If count is 0, trailUnitInverse = identity matrix.
pub fn build_flame_trail_ubo_fields(
    effect: &FlameEffect,
    trail: &FlameTrailState,
) -> (Matrix4<f32>, FlameTrailMeta, [[f32; 4]; 4]) {
    let count = trail.samples.len();

    if count == 0 {
        return (
            Matrix4::<f32>::from_scale(1.0),
            FlameTrailMeta {
                sample_count: 0.0,
                max_age: 0.0,
                _padding: [0.0; 2],
            },
            [[0.0; 4]; 4],
        );
    }

    // trailUnitInverse = inverse of unit model matrix (analytical: translation*scale -> inv_scale*inv_translation)
    let radius = flame_bounding_radius(effect);
    let trail_unit_inverse = Matrix4::from_translation(-effect.position)
        * Matrix4::from_nonuniform_scale(1.0 / radius, 1.0 / effect.height, 1.0 / radius);

    // Build local-space sample offsets and their normalized ages (u = age_seconds / fade_seconds)
    let linear = Matrix3::<f32>::from_cols(
        Vector3::new(
            trail_unit_inverse[0][0],
            trail_unit_inverse[1][0],
            trail_unit_inverse[2][0],
        ),
        Vector3::new(
            trail_unit_inverse[0][1],
            trail_unit_inverse[1][1],
            trail_unit_inverse[2][1],
        ),
        Vector3::new(
            trail_unit_inverse[0][2],
            trail_unit_inverse[1][2],
            trail_unit_inverse[2][2],
        ),
    );

    let mut max_u: f32 = 0.0;
    let mut ata: [[f32; 4]; 4] = [[0.0; 4]; 4]; // A^T * A (Vandermonde normal matrix)

    // Build A^T * A (independent of data, depends only on u values)
    for i in 0..count {
        let sample = &trail.samples[i];
        let u = if trail.fade_seconds > 0.0 {
            sample.age_seconds / trail.fade_seconds
        } else {
            0.0
        };
        if u > max_u {
            max_u = u;
        }

        // Vandermonde row: [1, u, u^2, u^3]
        let v = [1.0, u, u * u, u * u * u];
        for r in 0..4 {
            for c in 0..4 {
                ata[r][c] += v[r] * v[c];
            }
        }
    }

    // Build A^T * b for each axis (x, y, z) and solve least-squares
    // The system is the same A^T*A for all axes, only b changes.

    // Create augmented matrix [A^T*A | I] and row-reduce to get inverse
    let mut aug: [[f32; 8]; 4] = [[0.0; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            aug[r][c] = ata[r][c];
        }
        aug[r][r + 4] = 1.0;
    }

    // Gaussian elimination with partial pivoting
    for col in 0..4 {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..4 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }
        // Scale pivot row
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }
        for j in col..8 {
            aug[col][j] /= pivot;
        }
        // Eliminate column
        for row in 0..4 {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in col..8 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Now aug[:, 4..8] is the inverse of A^T*A
    // Compute coefficients for each axis: c = (A^T*A)^{-1} * A^T * b_axis
    let mut atb_x: [f32; 4] = [0.0; 4];
    let mut atb_y: [f32; 4] = [0.0; 4];
    let mut atb_z: [f32; 4] = [0.0; 4];
    for i in 0..count {
        let sample = &trail.samples[i];
        let u = if trail.fade_seconds > 0.0 {
            sample.age_seconds / trail.fade_seconds
        } else {
            0.0
        };
        let diff = Vector3::new(
            sample.position[0] - effect.position.x,
            sample.position[1] - effect.position.y,
            sample.position[2] - effect.position.z,
        );
        let local_delta = linear * diff;

        let v = [1.0, u, u * u, u * u * u];
        for r in 0..4 {
            atb_x[r] += v[r] * local_delta.x;
            atb_y[r] += v[r] * local_delta.y;
            atb_z[r] += v[r] * local_delta.z;
        }
    }

    // c = aug_inv * atb for each axis
    let mut coefficients: [[f32; 4]; 4] = [[0.0; 4]; 4];
    for r in 0..4 {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_z = 0.0;
        for c in 0..4 {
            sum_x += aug[r][c + 4] * atb_x[c];
            sum_y += aug[r][c + 4] * atb_y[c];
            sum_z += aug[r][c + 4] * atb_z[c];
        }
        coefficients[r] = [sum_x, sum_y, sum_z, 0.0];
    }

    let meta = FlameTrailMeta {
        sample_count: count as f32,
        max_age: max_u,
        _padding: [0.0; 2],
    };

    (trail_unit_inverse, meta, coefficients)
}
