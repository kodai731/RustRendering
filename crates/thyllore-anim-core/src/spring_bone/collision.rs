use cgmath::Vector3;

pub struct WorldCollider {
    pub center: Vector3<f32>,
    pub radius: f32,
    pub tail: Option<Vector3<f32>>,
}

pub fn resolve_all_collisions(
    position: Vector3<f32>,
    joint_radius: f32,
    colliders: &[WorldCollider],
) -> Vector3<f32> {
    let mut resolved = position;
    for collider in colliders {
        resolved = resolve_collision(resolved, joint_radius, collider);
    }
    resolved
}

fn resolve_collision(
    position: Vector3<f32>,
    joint_radius: f32,
    collider: &WorldCollider,
) -> Vector3<f32> {
    match collider.tail {
        Some(tail) => resolve_sphere_vs_capsule(
            position,
            joint_radius,
            collider.center,
            tail,
            collider.radius,
        ),
        None => resolve_sphere_vs_sphere(position, joint_radius, collider.center, collider.radius),
    }
}

fn resolve_sphere_vs_sphere(
    joint_pos: Vector3<f32>,
    joint_radius: f32,
    sphere_center: Vector3<f32>,
    sphere_radius: f32,
) -> Vector3<f32> {
    let diff = joint_pos - sphere_center;
    let distance = (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z).sqrt();
    let min_distance = joint_radius + sphere_radius;

    if distance >= min_distance {
        return joint_pos;
    }

    if distance < 1e-8 {
        return sphere_center + Vector3::new(0.0, min_distance, 0.0);
    }

    sphere_center + (diff / distance) * min_distance
}

fn resolve_sphere_vs_capsule(
    joint_pos: Vector3<f32>,
    joint_radius: f32,
    capsule_head: Vector3<f32>,
    capsule_tail: Vector3<f32>,
    capsule_radius: f32,
) -> Vector3<f32> {
    let closest = find_closest_point_on_segment(capsule_head, capsule_tail, joint_pos);
    resolve_sphere_vs_sphere(joint_pos, joint_radius, closest, capsule_radius)
}

fn find_closest_point_on_segment(
    start: Vector3<f32>,
    end: Vector3<f32>,
    point: Vector3<f32>,
) -> Vector3<f32> {
    let seg = end - start;
    let seg_len_sq = seg.x * seg.x + seg.y * seg.y + seg.z * seg.z;

    if seg_len_sq < 1e-8 {
        return start;
    }

    let to_point = point - start;
    let t = (to_point.x * seg.x + to_point.y * seg.y + to_point.z * seg.z) / seg_len_sq;
    let t = t.clamp(0.0, 1.0);

    start + seg * t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Vector3<f32>, b: Vector3<f32>, eps: f32) -> bool {
        (a.x - b.x).abs() < eps && (a.y - b.y).abs() < eps && (a.z - b.z).abs() < eps
    }

    fn magnitude(v: Vector3<f32>) -> f32 {
        (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
    }

    #[test]
    fn test_no_collision_distant_spheres() {
        let pos = Vector3::new(5.0, 0.0, 0.0);
        let result = resolve_sphere_vs_sphere(pos, 0.1, Vector3::new(0.0, 0.0, 0.0), 0.5);
        assert!(approx_eq(result, pos, 1e-5));
    }

    #[test]
    fn test_collision_penetrating_spheres() {
        let pos = Vector3::new(0.3, 0.0, 0.0);
        let center = Vector3::new(0.0, 0.0, 0.0);
        let result = resolve_sphere_vs_sphere(pos, 0.1, center, 0.5);

        let dist = magnitude(result - center);
        assert!((dist - 0.6).abs() < 1e-5);
        assert!(result.x > 0.0);
    }

    #[test]
    fn test_no_collision_at_exact_contact() {
        let center = Vector3::new(0.0, 0.0, 0.0);
        let pos = Vector3::new(0.6, 0.0, 0.0);
        let result = resolve_sphere_vs_sphere(pos, 0.1, center, 0.5);
        assert!(approx_eq(result, pos, 1e-5));
    }

    #[test]
    fn test_capsule_collision_near_center() {
        let capsule_head = Vector3::new(0.0, 0.0, 0.0);
        let capsule_tail = Vector3::new(0.0, 2.0, 0.0);
        let pos = Vector3::new(0.3, 1.0, 0.0);

        let result = resolve_sphere_vs_capsule(pos, 0.1, capsule_head, capsule_tail, 0.5);
        let closest = find_closest_point_on_segment(capsule_head, capsule_tail, pos);
        let dist = magnitude(result - closest);
        assert!((dist - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_capsule_collision_near_endpoint() {
        let capsule_head = Vector3::new(0.0, 0.0, 0.0);
        let capsule_tail = Vector3::new(0.0, 2.0, 0.0);
        let pos = Vector3::new(0.3, -0.5, 0.0);

        let result = resolve_sphere_vs_capsule(pos, 0.1, capsule_head, capsule_tail, 0.5);
        let dist = magnitude(result - capsule_head);
        assert!((dist - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_degenerate_capsule_acts_as_sphere() {
        let center = Vector3::new(0.0, 0.0, 0.0);
        let pos = Vector3::new(0.3, 0.0, 0.0);

        let sphere_result = resolve_sphere_vs_sphere(pos, 0.1, center, 0.5);
        let capsule_result = resolve_sphere_vs_capsule(pos, 0.1, center, center, 0.5);

        assert!(approx_eq(sphere_result, capsule_result, 1e-5));
    }

    #[test]
    fn test_resolve_all_collisions_sequential() {
        let colliders = vec![
            WorldCollider {
                center: Vector3::new(-2.0, 0.0, 0.0),
                radius: 0.5,
                tail: None,
            },
            WorldCollider {
                center: Vector3::new(0.0, 0.0, 0.0),
                radius: 0.5,
                tail: None,
            },
        ];
        let pos = Vector3::new(0.3, 0.0, 0.0);

        let result = resolve_all_collisions(pos, 0.1, &colliders);

        let dist1 = magnitude(result - colliders[1].center);
        assert!((dist1 - 0.6).abs() < 1e-4);
    }

    #[test]
    fn test_complete_overlap_default_direction() {
        let center = Vector3::new(1.0, 2.0, 3.0);
        let pos = center;
        let result = resolve_sphere_vs_sphere(pos, 0.1, center, 0.5);

        let expected = center + Vector3::new(0.0, 0.6, 0.0);
        assert!(approx_eq(result, expected, 1e-5));
    }
}
