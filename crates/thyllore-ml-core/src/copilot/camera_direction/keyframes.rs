#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraKeyParam {
    TranslationX,
    TranslationY,
    TranslationZ,
    RotationX,
    RotationY,
    RotationZ,
}

/// Extract XYZ Euler angles (roll_x, pitch_y, yaw_z) in degrees from a 3x3 rotation matrix.
///
/// Mirrors `AnimationModelTraining/scripts/camera_copilot/gendop_traj_to_thyllore.py`'s
/// `matrix_to_euler_xyz` bit-for-bit: `sy = sqrt(r[0][0]^2 + r[1][0]^2)`, falling back to
/// `x = 0` when `sy` is near zero (gimbal lock).
pub fn matrix_to_euler_xyz(r: &[[f32; 3]; 3]) -> (f32, f32, f32) {
    let sy = (r[0][0] * r[0][0] + r[1][0] * r[1][0]).sqrt();
    let eps = 1e-6;
    let (x, y, z) = if sy > eps {
        let x = r[2][1].atan2(r[2][2]);
        let y = (-r[2][0]).atan2(sy);
        let z = r[1][0].atan2(r[0][0]);
        (x, y, z)
    } else {
        let x = 0.0;
        let y = (-r[2][0]).atan2(sy);
        let z = (-r[0][1]).atan2(r[0][0]);
        (x, y, z)
    };
    (x.to_degrees(), y.to_degrees(), z.to_degrees())
}

/// Re-express GenDoP poses (camera-local: the first pose is the identity at the
/// camera's starting view) in world space by left-multiplying each with the
/// starting camera-to-world matrix.
pub fn transform_poses_to_world(
    camera_to_world: &[[f32; 4]; 4],
    poses: &[[[f32; 4]; 4]],
) -> Vec<[[f32; 4]; 4]> {
    poses
        .iter()
        .map(|pose| multiply_4x4(camera_to_world, pose))
        .collect()
}

fn multiply_4x4(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// Convert a slice of camera-to-world 4x4 poses into keyframe tuples.
///
/// For every `stride`-th frame:
///   - `time = frame_index / fps`
///   - translation is extracted from `matrix[i][3]` for i in 0..2 (x, y, z)
///   - rotation is extracted via `matrix_to_euler_xyz` on the upper-left 3x3 submatrix
///   - 6 tuples are emitted: one per parameter (TranslationX/Y/Z, RotationX/Y/Z)
pub fn poses_to_keyframe_tuples(
    poses: &[[[f32; 4]; 4]],
    fps: f32,
    stride: usize,
) -> Vec<(f32, CameraKeyParam, f32)> {
    let mut tuples = Vec::new();
    for (i, pose) in poses.iter().enumerate().step_by(stride) {
        let time = i as f32 / fps;

        let tx = pose[0][3];
        let ty = pose[1][3];
        let tz = pose[2][3];

        let r: [[f32; 3]; 3] = [
            [pose[0][0], pose[0][1], pose[0][2]],
            [pose[1][0], pose[1][1], pose[1][2]],
            [pose[2][0], pose[2][1], pose[2][2]],
        ];
        let (rx, ry, rz) = matrix_to_euler_xyz(&r);

        tuples.push((time, CameraKeyParam::TranslationX, tx));
        tuples.push((time, CameraKeyParam::TranslationY, ty));
        tuples.push((time, CameraKeyParam::TranslationZ, tz));
        tuples.push((time, CameraKeyParam::RotationX, rx));
        tuples.push((time, CameraKeyParam::RotationY, ry));
        tuples.push((time, CameraKeyParam::RotationZ, rz));
    }
    tuples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_poses_to_world_moves_a_local_step_along_the_camera_axes() {
        let camera_to_world = [
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 1.0, 0.0, 2.0],
            [-1.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let forward_step = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -3.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let world = transform_poses_to_world(&camera_to_world, &[forward_step]);

        assert_eq!(world[0][0][3], 10.0 - 3.0);
        assert_eq!(world[0][1][3], 2.0);
        assert_eq!(world[0][2][3], 5.0);
    }

    fn identity_pose() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn test_identity_matrix() {
        let pose = identity_pose();
        let r: [[f32; 3]; 3] = [
            [pose[0][0], pose[0][1], pose[0][2]],
            [pose[1][0], pose[1][1], pose[1][2]],
            [pose[2][0], pose[2][1], pose[2][2]],
        ];
        let (rx, ry, rz) = matrix_to_euler_xyz(&r);

        assert!((rx).abs() < 1e-6, "roll should be ~0, got {}", rx);
        assert!((ry).abs() < 1e-6, "pitch should be ~0, got {}", ry);
        assert!((rz).abs() < 1e-6, "yaw should be ~0, got {}", rz);

        // Translation from identity is (0, 0, 0)
        assert_eq!(pose[0][3], 0.0);
        assert_eq!(pose[1][3], 0.0);
        assert_eq!(pose[2][3], 0.0);
    }

    #[test]
    fn test_poses_to_keyframe_tuples_identity() {
        let poses = vec![identity_pose(), identity_pose(), identity_pose()];
        let tuples = poses_to_keyframe_tuples(&poses, 30.0, 1);

        // 3 frames * 6 params = 18 tuples
        assert_eq!(tuples.len(), 18);

        // All values should be ~0
        for (time, _param, value) in &tuples {
            assert!(
                (time).abs() < 1e-6
                    || (*time - 1.0 / 30.0).abs() < 1e-6
                    || (*time - 2.0 / 30.0).abs() < 1e-6
            );
            assert!((*value).abs() < 1e-6, "value should be ~0, got {}", value);
        }
    }
    #[test]
    fn test_rotation_90_degrees_x() {
        // +90 degree rotation around X axis:
        // [1,  0,  0]
        // [0,  0, -1]
        // [0,  1,  0]
        let r: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        let (rx, ry, rz) = matrix_to_euler_xyz(&r);

        assert!((rx - 90.0).abs() < 1e-4, "roll should be ~90, got {}", rx);
        assert!((ry).abs() < 1e-4, "pitch should be ~0, got {}", ry);
        assert!((rz).abs() < 1e-4, "yaw should be ~0, got {}", rz);
    }

    #[test]
    fn test_rotation_90_degrees_y() {
        // +90 degree rotation around Y axis:
        // [ 0, 0, 1]
        // [ 0, 1, 0]
        // [-1, 0, 0]
        let r: [[f32; 3]; 3] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]];
        let (rx, ry, rz) = matrix_to_euler_xyz(&r);

        assert!((rx).abs() < 1e-4, "roll should be ~0, got {}", rx);
        assert!((ry - 90.0).abs() < 1e-4, "pitch should be ~90, got {}", ry);
        assert!((rz).abs() < 1e-4, "yaw should be ~0, got {}", rz);
    }

    #[test]
    fn test_rotation_90_degrees_z() {
        // +90 degree rotation around Z axis:
        // [ 0,-1, 0]
        // [ 1, 0, 0]
        // [ 0, 0, 1]
        let r: [[f32; 3]; 3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let (rx, ry, rz) = matrix_to_euler_xyz(&r);

        assert!((rx).abs() < 1e-4, "roll should be ~0, got {}", rx);
        assert!((ry).abs() < 1e-4, "pitch should be ~0, got {}", ry);
        assert!((rz - 90.0).abs() < 1e-4, "yaw should be ~90, got {}", rz);
    }

    #[test]
    fn test_stride_filtering() {
        let poses = vec![identity_pose(); 6];
        let tuples = poses_to_keyframe_tuples(&poses, 30.0, 2);

        // stride=2: frames 0, 2, 4 -> 3 frames * 6 params = 18 tuples
        assert_eq!(tuples.len(), 18);

        // Check times: frame 0 -> time 0, frame 2 -> time 2/30, frame 4 -> time 4/30
        let times: Vec<f32> = tuples.iter().map(|(t, _, _)| *t).collect::<Vec<_>>();
        assert!((times[0] - 0.0).abs() < 1e-6);
        assert!((times[6] - 2.0 / 30.0).abs() < 1e-6);
        assert!((times[12] - 4.0 / 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_translation_extraction() {
        let mut pose = identity_pose();
        pose[0][3] = 1.5;
        pose[1][3] = -2.0;
        pose[2][3] = 3.0;
        let poses = vec![pose];

        let tuples = poses_to_keyframe_tuples(&poses, 30.0, 1);
        assert_eq!(tuples.len(), 6);

        for (time, param, value) in &tuples {
            assert!((time).abs() < 1e-6);
            match param {
                CameraKeyParam::TranslationX => assert!((*value - 1.5).abs() < 1e-6),
                CameraKeyParam::TranslationY => assert!((*value + 2.0).abs() < 1e-6),
                CameraKeyParam::TranslationZ => assert!((*value - 3.0).abs() < 1e-6),
                _ => assert!((*value).abs() < 1e-6),
            }
        }
    }

    #[test]
    fn test_gimbal_lock_pitch_90() {
        // +90 degree rotation around Y axis triggers gimbal lock (sy <= eps).
        // Matrix for Y=+90°:
        // [ 0, 0, 1]
        // [ 0, 1, 0]
        // [-1, 0, 0]
        let r: [[f32; 3]; 3] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]];
        let (rx, ry, rz) = matrix_to_euler_xyz(&r);

        // sy = sqrt(r[0][0]^2 + r[1][0]^2) = sqrt(0 + 0) = 0 <= eps -> gimbal lock branch
        // x = 0.0 (forced in gimbal lock branch)
        // y = atan2(-r[2][0], sy) = atan2(-(-1), 0) = atan2(1, 0) = +90°
        // z = atan2(-r[0][1], r[0][0]) = atan2(0, 0) -> depends on implementation
        assert!(
            (rx).abs() < 1e-4,
            "roll should be ~0 (gimbal lock), got {}",
            rx
        );
        assert!((ry - 90.0).abs() < 1e-4, "pitch should be ~90, got {}", ry);
        // z is arbitrary in gimbal lock; just check it's finite
        assert!(
            rz.is_finite(),
            "yaw should be finite in gimbal lock, got {}",
            rz
        );
    }
}
