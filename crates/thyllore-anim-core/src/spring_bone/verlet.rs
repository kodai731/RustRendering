use cgmath::{InnerSpace, Matrix4, Quaternion, Vector3};

use crate::{compose_transform, decompose_transform, normalize_quat};

pub fn integrate_joint(
    current_tail: Vector3<f32>,
    prev_tail: Vector3<f32>,
    drag_force: f32,
    stiffness: f32,
    gravity_dir: Vector3<f32>,
    gravity_power: f32,
    parent_world_rotation: Quaternion<f32>,
    initial_local_rotation: Quaternion<f32>,
    bone_axis: Vector3<f32>,
    bone_length: f32,
    head_position: Vector3<f32>,
    dt: f32,
) -> Vector3<f32> {
    let inertia = (current_tail - prev_tail) * (1.0 - drag_force);

    let rest_dir = rotate_vector(parent_world_rotation * initial_local_rotation, bone_axis);
    let rest_tail = head_position + rest_dir * bone_length;
    let stiffness_force = (rest_tail - current_tail) * stiffness * dt;

    let external_force = gravity_dir * gravity_power * dt;

    current_tail + inertia + stiffness_force + external_force
}

pub fn apply_length_constraint(
    head: Vector3<f32>,
    tail: Vector3<f32>,
    bone_length: f32,
) -> Vector3<f32> {
    let dir = tail - head;
    let len = dir.magnitude();

    if len < 1e-8 {
        return head + Vector3::new(0.0, bone_length, 0.0);
    }

    head + (dir / len) * bone_length
}

pub fn compute_joint_rotation(
    head_position: Vector3<f32>,
    parent_world_rotation: Quaternion<f32>,
    initial_local_rotation: Quaternion<f32>,
    bone_axis: Vector3<f32>,
    current_tail: Vector3<f32>,
) -> Quaternion<f32> {
    let rest_dir = rotate_vector(parent_world_rotation * initial_local_rotation, bone_axis);

    let actual_dir = current_tail - head_position;
    let actual_len = actual_dir.magnitude();
    if actual_len < 1e-8 {
        return initial_local_rotation;
    }
    let actual_dir = actual_dir / actual_len;

    let rotation_diff = rotation_between(rest_dir, actual_dir);

    let parent_rot_inv = conjugate(parent_world_rotation);
    normalize_quat(parent_rot_inv * rotation_diff * parent_world_rotation * initial_local_rotation)
}

pub fn extract_world_position(transform: &Matrix4<f32>) -> Vector3<f32> {
    Vector3::new(transform[3][0], transform[3][1], transform[3][2])
}

pub fn compute_tail_position(
    head_position: Vector3<f32>,
    parent_world_rotation: Quaternion<f32>,
    initial_local_rotation: Quaternion<f32>,
    bone_axis: Vector3<f32>,
    bone_length: f32,
) -> Vector3<f32> {
    let dir = rotate_vector(parent_world_rotation * initial_local_rotation, bone_axis);
    head_position + dir * bone_length
}

pub fn recompute_global_transform(
    parent_global: &Matrix4<f32>,
    local_rotation: Quaternion<f32>,
    original_local_transform: &Matrix4<f32>,
) -> Matrix4<f32> {
    let (t, _, s) = decompose_transform(original_local_transform);
    let local = compose_transform(t, local_rotation, s);
    parent_global * local
}

fn rotate_vector(q: Quaternion<f32>, v: Vector3<f32>) -> Vector3<f32> {
    let qv = Quaternion::new(0.0, v.x, v.y, v.z);
    let result = q * qv * conjugate(q);
    Vector3::new(result.v.x, result.v.y, result.v.z)
}

fn conjugate(q: Quaternion<f32>) -> Quaternion<f32> {
    Quaternion::new(q.s, -q.v.x, -q.v.y, -q.v.z)
}

fn rotation_between(from: Vector3<f32>, to: Vector3<f32>) -> Quaternion<f32> {
    let from_len = from.magnitude();
    let to_len = to.magnitude();
    if from_len < 1e-8 || to_len < 1e-8 {
        return Quaternion::new(1.0, 0.0, 0.0, 0.0);
    }

    let from_n = from / from_len;
    let to_n = to / to_len;

    let dot = from_n.x * to_n.x + from_n.y * to_n.y + from_n.z * to_n.z;

    if dot > 0.9999 {
        return Quaternion::new(1.0, 0.0, 0.0, 0.0);
    }

    if dot < -0.9999 {
        let perp = if from_n.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let axis = from_n.cross(perp);
        let axis_len = axis.magnitude();
        if axis_len < 1e-8 {
            return Quaternion::new(1.0, 0.0, 0.0, 0.0);
        }
        let axis = axis / axis_len;
        return Quaternion::new(0.0, axis.x, axis.y, axis.z);
    }

    let cross = from_n.cross(to_n);
    let w = 1.0 + dot;
    normalize_quat(Quaternion::new(w, cross.x, cross.y, cross.z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_length_constraint_preserves_length() {
        let head = Vector3::new(0.0, 0.0, 0.0);
        let tail = Vector3::new(3.0, 4.0, 0.0);
        let bone_length = 2.0;

        let result = apply_length_constraint(head, tail, bone_length);
        let dist =
            ((result - head).x.powi(2) + (result - head).y.powi(2) + (result - head).z.powi(2))
                .sqrt();

        assert!((dist - bone_length).abs() < 1e-5);
    }

    #[test]
    fn test_integrate_joint_with_no_forces() {
        let head = Vector3::new(0.0, 0.0, 0.0);
        let current = Vector3::new(0.0, 1.0, 0.0);
        let prev = Vector3::new(0.0, 1.0, 0.0);

        let result = integrate_joint(
            current,
            prev,
            1.0,
            0.0,
            Vector3::new(0.0, -1.0, 0.0),
            0.0,
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
            head,
            0.016,
        );

        assert!((result - current).magnitude() < 1e-5);
    }

    #[test]
    fn test_rotation_between_same_direction() {
        let dir = Vector3::new(0.0, 1.0, 0.0);
        let q = rotation_between(dir, dir);
        assert!((q.s - 1.0).abs() < 1e-4);
    }
}
