use cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector3, Vector4};

use super::torus_intersect::intersect_torus;

pub fn pick_torus(
    ray_origin: Vector3<f32>,
    ray_dir: Vector3<f32>,
    model: Matrix4<f32>,
    inverse_model: Matrix4<f32>,
    R: f32,
    r: f32,
) -> Option<f32> {
    let local_origin = inverse_model * Vector4::new(ray_origin.x, ray_origin.y, ray_origin.z, 1.0);
    let local_dir = inverse_model * Vector4::new(ray_dir.x, ray_dir.y, ray_dir.z, 0.0);

    let hits = intersect_torus(local_origin.truncate(), local_dir.truncate(), R, r);
    if hits.count == 0 {
        return None;
    }

    let t_local = hits.roots[0];
    let world_hit = model
        * Vector4::new(
            local_origin.x + local_dir.x * t_local,
            local_origin.y + local_dir.y * t_local,
            local_origin.z + local_dir.z * t_local,
            1.0,
        );

    let hit_world = world_hit.truncate();
    let dist = (hit_world - ray_origin).magnitude();
    Some(dist)
}
