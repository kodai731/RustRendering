/// Detokenization of GenDoP camera copilot outputs.
///
/// Each pose is encoded as 10 discrete token values in `[0, DISCRETE_BINS)` and decoded back to a
/// 4x4 homogeneous camera-to-world matrix (right-handed, glTF/OpenGL convention).
///
/// Token layout per pose: `[q0, q1, q2, q3, t0, t1, t2, fx_raw, fy_raw, scale_raw]`
/// where `q` is quaternion (w, x, y, z order), `t` is translation, and the last 3 are intrinsic
/// parameters.

/// Number of discrete bins for tokenization (matches GenDoP `discrete_bins=256`).
const DISCRETE_BINS: f32 = 256.0;

/// Scale factor for trajectory values: `temp_traj = coords_traj / (TEMP_TRAJ_SCALE * DISCRETE_BINS) - 1.0`.
const TEMP_TRAJ_SCALE: f32 = 0.5;

/// Scale factor in exponential: `scale = exp(scale_raw / DISCRETE_BINS * SCALE_FACTOR - SCALE_OFFSET)`.
const SCALE_FACTOR: f32 = 4.0;

/// Offset in exponential: `scale = exp(scale_raw / DISCRETE_BINS * SCALE_FACTOR - SCALE_OFFSET)`.
const SCALE_OFFSET: f32 = 2.0;

/// Convert a unit quaternion `(w, x, y, z)` to a 3x3 rotation matrix (row-major).
///
/// The input `q` is assumed to be in `(w, x, y, z)` order — matching how GenDoP passes
/// `torch.tensor([q0, q1, q2, q3, ...])` where `q0=w`.
///
/// Normalization is performed via `two_s = 2.0 / (w*w + x*x + y*y + z*z)` so that
/// non-unit quaternions (e.g. from inverse quantization of discrete tokens) are handled correctly.
pub fn quaternion_to_matrix(q: [f32; 4]) -> [[f32; 3]; 3] {
    let w = q[0];
    let x = q[1];
    let y = q[2];
    let z = q[3];

    let two_s = 2.0 / (w * w + x * x + y * y + z * z);

    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    [
        [
            1.0 - two_s * (yy + zz),
            two_s * (xy - wz),
            two_s * (xz + wy),
        ],
        [
            two_s * (xy + wz),
            1.0 - two_s * (xx + zz),
            two_s * (yz - wx),
        ],
        [
            two_s * (xz - wy),
            two_s * (yz + wx),
            1.0 - two_s * (xx + yy),
        ],
    ]
}

/// Detokenize a single pose from its 10 raw token values into a 4x4 homogeneous matrix.
///
/// The `raw10` array contains `[q0, q1, q2, q3, t0, t1, t2, fx_raw, fy_raw, scale_raw]`.
/// Returns the camera-to-world 4x4 matrix (row-major, right-handed).
pub fn detokenize_pose(raw10: [f32; 10]) -> [[f32; 4]; 4] {
    // Extract coords_traj (first 7) and coords_instri (last 3)
    let coords_traj = [
        raw10[0], raw10[1], raw10[2], raw10[3], raw10[4], raw10[5], raw10[6],
    ];
    let coords_instri = [raw10[7], raw10[8], raw10[9]];

    // Inverse quantization of trajectory values: temp_traj = coords_traj / (TEMP_TRAJ_SCALE * DISCRETE_BINS) - 1.0
    let temp_traj: [f32; 7] = coords_traj.map(|c| c / (TEMP_TRAJ_SCALE * DISCRETE_BINS) - 1.0);

    // Scale from exponential of scale_raw: scale = exp(scale_raw / DISCRETE_BINS * SCALE_FACTOR - SCALE_OFFSET)
    let scale_raw = coords_instri[2];
    let scale = (scale_raw / DISCRETE_BINS * SCALE_FACTOR - SCALE_OFFSET).exp();

    // Rotation matrix from quaternion (first 4 of temp_traj)
    let quat: [f32; 4] = [temp_traj[0], temp_traj[1], temp_traj[2], temp_traj[3]];
    let r = quaternion_to_matrix(quat);

    // Translation (last 3 of temp_traj), then apply scale
    let mut t = [temp_traj[4], temp_traj[5], temp_traj[6]];
    for i in 0..3 {
        t[i] *= scale;
    }

    // Construct 4x4 homogeneous matrix: [R | T; 0, 0, 0, 1] (row-major)
    [
        [r[0][0], r[0][1], r[0][2], t[0]],
        [r[1][0], r[1][1], r[1][2], t[1]],
        [r[2][0], r[2][1], r[2][2], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Detokenize a sequence of tokens into a vector of 4x4 pose matrices.
///
/// The token slice must have length divisible by 10 (each pose is 10 tokens).
/// Returns an error if the length is not a multiple of 10.
pub fn detokenize_sequence(tokens: &[f32]) -> Result<Vec<[[f32; 4]; 4]>, String> {
    if tokens.len() % 10 != 0 {
        return Err(format!(
            "token length {} is not a multiple of 10",
            tokens.len()
        ));
    }

    let mut poses = Vec::with_capacity(tokens.len() / 10);
    for chunk in tokens.chunks_exact(10) {
        let raw10: [f32; 10] = chunk.try_into().unwrap();
        poses.push(detokenize_pose(raw10));
    }

    Ok(poses)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// near-identity quaternion (w≈1, x=y=z=0) and zero translation input should produce a rotation
    /// matrix close to identity — confirms detokenize_pose handles the identity case correctly.
    #[test]
    fn test_detokenize_pose_identity() {
        // Construct tokens for near-identity: w≈1, x=y=z=0, t=(0,0,0)
        // token -> temp = token/128 - 1
        // w=1: token=256 (out of range), use 255 => temp_w = 255/128-1 ≈ 0.992
        // x=y=z=t0=t1=t2=0: token=128 => temp = 0
        // For scale, any value is fine; we just check R is identity-like and T is near-zero.
        let tokens: [f32; 10] = [
            255.0, 128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 0.0, 0.0, 0.0,
        ];

        let matrix = detokenize_pose(tokens);

        // w ≈ 0.992, x=y=z=0 => R should be very close to identity
        // T = [0, 0, 0] * scale = [0, 0, 0] (translation is zero regardless of scale)
        let eps = 1e-4;

        // Check rotation part is near identity
        assert!(
            (matrix[0][0] - 1.0).abs() < eps,
            "R[0][0] = {}",
            matrix[0][0]
        );
        assert!(matrix[0][1].abs() < eps, "R[0][1] = {}", matrix[0][1]);
        assert!(matrix[0][2].abs() < eps, "R[0][2] = {}", matrix[0][2]);
        assert!(matrix[1][0].abs() < eps, "R[1][0] = {}", matrix[1][0]);
        assert!(
            (matrix[1][1] - 1.0).abs() < eps,
            "R[1][1] = {}",
            matrix[1][1]
        );
        assert!(matrix[1][2].abs() < eps, "R[1][2] = {}", matrix[1][2]);
        assert!(matrix[2][0].abs() < eps, "R[2][0] = {}", matrix[2][0]);
        assert!(matrix[2][1].abs() < eps, "R[2][1] = {}", matrix[2][1]);
        assert!(
            (matrix[2][2] - 1.0).abs() < eps,
            "R[2][2] = {}",
            matrix[2][2]
        );

        // Check translation is zero
        assert!(matrix[0][3].abs() < eps, "T[0] = {}", matrix[0][3]);
        assert!(matrix[1][3].abs() < eps, "T[1] = {}", matrix[1][3]);
        assert!(matrix[2][3].abs() < eps, "T[2] = {}", matrix[2][3]);

        // Check bottom row
        assert_eq!(matrix[3][0], 0.0);
        assert_eq!(matrix[3][1], 0.0);
        assert_eq!(matrix[3][2], 0.0);
        assert_eq!(matrix[3][3], 1.0);
    }

    #[test]
    fn test_detokenize_sequence_length_validation() {
        let tokens: Vec<f32> = vec![0.0; 9];
        let result = detokenize_sequence(&tokens);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a multiple of 10"));
    }

    #[test]
    fn test_detokenize_sequence_empty() {
        let tokens: Vec<f32> = vec![];
        let result = detokenize_sequence(&tokens);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_detokenize_sequence_two_poses() {
        let tokens: Vec<f32> = vec![0.0; 20];
        let result = detokenize_sequence(&tokens);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_quaternion_to_matrix_non_unit() {
        // Non-unit quaternion [2.0, 0.0, 0.0, 0.0] has norm 2.0 (not 1.0).
        // After normalization via two_s = 2.0 / (4+0+0+0) = 0.5, it should produce
        // the same identity rotation matrix as the unit quaternion [1.0, 0.0, 0.0, 0.0].
        let non_unit_q: [f32; 4] = [2.0, 0.0, 0.0, 0.0];
        let unit_q: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

        let matrix_non_unit = quaternion_to_matrix(non_unit_q);
        let matrix_unit = quaternion_to_matrix(unit_q);

        // Both should produce the identity rotation matrix
        let eps = 1e-6;
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (matrix_non_unit[i][j] - matrix_unit[i][j]).abs() < eps,
                    "R[{}][{}]: non_unit={:.6}, unit={:.6}",
                    i,
                    j,
                    matrix_non_unit[i][j],
                    matrix_unit[i][j]
                );
            }
        }

        // Also verify that the non-unit quaternion result is indeed the identity matrix
        assert!((matrix_non_unit[0][0] - 1.0).abs() < eps);
        assert!(matrix_non_unit[0][1].abs() < eps);
        assert!(matrix_non_unit[0][2].abs() < eps);
        assert!(matrix_non_unit[1][0].abs() < eps);
        assert!((matrix_non_unit[1][1] - 1.0).abs() < eps);
        assert!(matrix_non_unit[1][2].abs() < eps);
        assert!(matrix_non_unit[2][0].abs() < eps);
        assert!(matrix_non_unit[2][1].abs() < eps);
        assert!((matrix_non_unit[2][2] - 1.0).abs() < eps);
    }
}
