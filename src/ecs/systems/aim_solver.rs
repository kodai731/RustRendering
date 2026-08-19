use cgmath::{InnerSpace, Quaternion, Vector3};

use crate::animation::normalize_quat;

#[derive(Clone, Copy, Debug)]
pub struct AimSolveInput {
    pub source_pos: Vector3<f32>,
    pub source_rot: Quaternion<f32>,
    pub target_pos: Vector3<f32>,
    pub aim_axis: Vector3<f32>,
    pub up_axis: Vector3<f32>,
    pub up_world: Option<Vector3<f32>>,
}

/// 世界座標系で source を target に向けた最終回転 (twist 補正込み)。
/// target と source が一致 (magnitude2 < 1e-8) なら None。
pub fn solve_aim_world_rotation(input: &AimSolveInput) -> Option<Quaternion<f32>> {
    let direction = input.target_pos - input.source_pos;

    if direction.magnitude2() < 1e-8 {
        return None;
    }

    let current_aim = rotate_vector_by_quat(input.source_rot, input.aim_axis);
    let aim_rotation = rotation_between_vectors(current_aim, direction);

    let up_world = match input.up_world {
        Some(v) => v,
        None => input.up_axis,
    };

    let rotated_up = rotate_vector_by_quat(
        aim_rotation,
        rotate_vector_by_quat(input.source_rot, input.up_axis),
    );
    let direction_normalized = direction.normalize();
    let desired_up = up_world - direction_normalized * direction_normalized.dot(up_world);
    let actual_up = rotated_up - direction_normalized * direction_normalized.dot(rotated_up);

    let final_rot = if desired_up.magnitude2() > 1e-8 && actual_up.magnitude2() > 1e-8 {
        let twist = rotation_between_vectors(actual_up.normalize(), desired_up.normalize());
        normalize_quat(quat_mul(twist, quat_mul(aim_rotation, input.source_rot)))
    } else {
        normalize_quat(quat_mul(aim_rotation, input.source_rot))
    };

    Some(final_rot)
}

pub(crate) fn rotate_vector_by_quat(q: Quaternion<f32>, v: Vector3<f32>) -> Vector3<f32> {
    let qv = Vector3::new(q.v.x, q.v.y, q.v.z);
    let uv = qv.cross(v);
    let uuv = qv.cross(uv);
    v + (uv * q.s + uuv) * 2.0
}

pub(crate) fn rotation_between_vectors(from: Vector3<f32>, to: Vector3<f32>) -> Quaternion<f32> {
    let from_n = from.normalize();
    let to_n = to.normalize();
    let dot = from_n.dot(to_n);

    if dot > 0.9999 {
        return Quaternion::new(1.0, 0.0, 0.0, 0.0);
    }

    if dot < -0.9999 {
        let perp = if from_n.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let axis = from_n.cross(perp).normalize();
        return Quaternion::new(0.0, axis.x, axis.y, axis.z);
    }

    let axis = from_n.cross(to_n);
    let s = ((1.0 + dot) * 2.0).sqrt();
    let inv_s = 1.0 / s;
    normalize_quat(Quaternion::new(
        s * 0.5,
        axis.x * inv_s,
        axis.y * inv_s,
        axis.z * inv_s,
    ))
}

pub(crate) fn quaternion_from_axis_angle(axis: Vector3<f32>, angle: f32) -> Quaternion<f32> {
    let half = angle * 0.5;
    let s = half.sin();
    let c = half.cos();
    let a = axis.normalize();
    Quaternion::new(c, a.x * s, a.y * s, a.z * s)
}

pub(crate) fn quat_mul(a: Quaternion<f32>, b: Quaternion<f32>) -> Quaternion<f32> {
    Quaternion::new(
        a.s * b.s - a.v.x * b.v.x - a.v.y * b.v.y - a.v.z * b.v.z,
        a.s * b.v.x + a.v.x * b.s + a.v.y * b.v.z - a.v.z * b.v.y,
        a.s * b.v.y - a.v.x * b.v.z + a.v.y * b.s + a.v.z * b.v.x,
        a.s * b.v.z + a.v.x * b.v.y - a.v.y * b.v.x + a.v.z * b.s,
    )
}

pub(crate) fn conjugate_quat(q: Quaternion<f32>) -> Quaternion<f32> {
    Quaternion::new(q.s, -q.v.x, -q.v.y, -q.v.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aim_at_same_axis_near_identity() {
        let input = AimSolveInput {
            source_pos: Vector3::new(0.0, 0.0, 0.0),
            source_rot: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            target_pos: Vector3::new(0.0, 0.0, 5.0),
            aim_axis: Vector3::new(0.0, 0.0, 1.0),
            up_axis: Vector3::new(0.0, 1.0, 0.0),
            up_world: None,
        };

        let rot = solve_aim_world_rotation(&input).expect("should not be None");
        let z_rotated = rotate_vector_by_quat(rot, Vector3::new(0.0, 0.0, 1.0));
        assert!(
            (z_rotated - Vector3::new(0.0, 0.0, 1.0)).magnitude() < 1e-5,
            "Z-axis rotated result should be near (0,0,1), got {:?}",
            z_rotated
        );
    }

    #[test]
    fn test_aim_at_x_axis_up_maintained() {
        let input = AimSolveInput {
            source_pos: Vector3::new(0.0, 0.0, 0.0),
            source_rot: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            target_pos: Vector3::new(5.0, 0.0, 0.0),
            aim_axis: Vector3::new(0.0, 0.0, 1.0),
            up_axis: Vector3::new(0.0, 1.0, 0.0),
            up_world: None,
        };

        let rot = solve_aim_world_rotation(&input).expect("should not be None");
        let z_rotated = rotate_vector_by_quat(rot, Vector3::new(0.0, 0.0, 1.0));
        assert!(
            (z_rotated - Vector3::new(1.0, 0.0, 0.0)).magnitude() < 1e-5,
            "Z-axis rotated result should be near (1,0,0), got {:?}",
            z_rotated
        );

        let y_rotated = rotate_vector_by_quat(rot, Vector3::new(0.0, 1.0, 0.0));
        assert!(
            (y_rotated - Vector3::new(0.0, 1.0, 0.0)).magnitude() < 1e-5,
            "Y-axis rotated result should be near (0,1,0), got {:?}",
            y_rotated
        );
    }

    #[test]
    fn test_aim_source_equals_target_returns_none() {
        let input = AimSolveInput {
            source_pos: Vector3::new(1.0, 2.0, 3.0),
            source_rot: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            target_pos: Vector3::new(1.0, 2.0, 3.0),
            aim_axis: Vector3::new(0.0, 0.0, 1.0),
            up_axis: Vector3::new(0.0, 1.0, 0.0),
            up_world: None,
        };

        assert!(
            solve_aim_world_rotation(&input).is_none(),
            "should return None when source equals target"
        );
    }
}
