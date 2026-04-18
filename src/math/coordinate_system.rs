use crate::math::matrix::Mat4;
use crate::math::vector::Vector4;
use cgmath::{vec3, Deg, InnerSpace, Matrix4, Rad, SquareMatrix, Vector2, Vector3};
use std::f32::EPSILON;

pub fn fix_coord() -> Mat4 {
    Matrix4::from_cols(
        Vector4::new(1.0, 0.0, 0.0, 0.0), // X ← X
        Vector4::new(0.0, 0.0, 1.0, 0.0), // Y ← Z
        Vector4::new(0.0, 1.0, 0.0, 0.0), // Z ← -Y
        Vector4::new(0.0, 0.0, 0.0, 1.0),
    )
}
pub unsafe fn view(
    camera_pos: cgmath::Vector3<f32>,
    direction: cgmath::Vector3<f32>,
    up: cgmath::Vector3<f32>,
) -> cgmath::Matrix4<f32> {
    // glam look_to_rh 互換 (projection_y_flip との組み合わせで正しい向きになる)
    let forward = cgmath::Vector3::normalize(direction);
    let n_x = cgmath::Vector3::normalize(cgmath::Vector3::cross(forward, up));
    let n_y = cgmath::Vector3::cross(n_x, forward);
    let n_z = -forward;

    cgmath::Matrix4::new(
        n_x.x,
        n_y.x,
        n_z.x,
        0.0,
        n_x.y,
        n_y.y,
        n_z.y,
        0.0,
        n_x.z,
        n_y.z,
        n_z.z,
        0.0,
        -cgmath::InnerSpace::dot(camera_pos, n_x),
        -cgmath::InnerSpace::dot(camera_pos, n_y),
        -cgmath::InnerSpace::dot(camera_pos, n_z),
        1.0,
    )
}

pub fn perspective(fovy: Deg<f32>, aspect: f32, near: f32, far: f32) -> Matrix4<f32> {
    let fovy_rad: Rad<f32> = fovy.into();
    let f = 1.0 / (fovy_rad.0 / 2.0).tan();

    Matrix4::new(
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        -f,
        0.0,
        0.0,
        0.0,
        0.0,
        far / (near - far),
        -1.0,
        0.0,
        0.0,
        (near * far) / (near - far),
        0.0,
    )
}

pub fn perspective_infinite_reverse(fovy: Deg<f32>, aspect: f32, near: f32) -> Matrix4<f32> {
    let fovy_rad: Rad<f32> = fovy.into();
    let f = 1.0 / (fovy_rad.0 / 2.0).tan();

    Matrix4::new(
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        -f,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        near,
        0.0,
    )
}

/// FBX Z-up → ワールド Y-up 変換（X軸周りに-90度回転）
pub fn fbx_to_world() -> Matrix4<f32> {
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    )
}

pub fn world_to_fbx() -> Matrix4<f32> {
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    )
}

/// glTF Y-up → ワールド Y-up 変換（恒等変換）
pub fn gltf_to_world() -> Matrix4<f32> {
    Matrix4::identity()
}

/// Blender Z-up → ワールド Y-up 変換（FBXと同じ）
pub fn blender_to_world() -> Matrix4<f32> {
    fbx_to_world()
}

/// ワールド座標系のY軸（上向き）
pub fn world_y_axis() -> cgmath::Vector3<f32> {
    cgmath::vec3(0.0, 1.0, 0.0)
}

/// ワールド座標系のY軸負方向（下向き）
pub fn world_y_down() -> cgmath::Vector3<f32> {
    cgmath::vec3(0.0, -1.0, 0.0)
}

pub fn screen_to_world_ray(
    screen_pos: Vector2<f32>,
    screen_size: Vector2<f32>,
    view_matrix: Mat4,
    proj_matrix: Mat4,
) -> (Vector3<f32>, Vector3<f32>) {
    let ndc_x = (2.0 * screen_pos.x) / screen_size.x - 1.0;
    let ndc_y = (2.0 * screen_pos.y) / screen_size.y - 1.0;

    let clip_near = cgmath::vec4(ndc_x, ndc_y, 1.0, 1.0);
    let clip_far = cgmath::vec4(ndc_x, ndc_y, 0.0, 1.0);

    let view_proj_inverse = match (proj_matrix * view_matrix).invert() {
        Some(inv) => inv,
        None => return (vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, -1.0)),
    };

    let world_near_4 = view_proj_inverse * clip_near;
    let world_far_4 = view_proj_inverse * clip_far;

    let ray_origin = vec3(
        world_near_4.x / world_near_4.w,
        world_near_4.y / world_near_4.w,
        world_near_4.z / world_near_4.w,
    );

    let ray_direction = if world_far_4.w.abs() < EPSILON {
        vec3(world_far_4.x, world_far_4.y, world_far_4.z).normalize()
    } else {
        let world_far = vec3(
            world_far_4.x / world_far_4.w,
            world_far_4.y / world_far_4.w,
            world_far_4.z / world_far_4.w,
        );
        (world_far - ray_origin).normalize()
    };

    (ray_origin, ray_direction)
}

pub fn ray_to_point_distance(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    point: Vector3<f32>,
) -> f32 {
    let to_point = point - ray_origin;
    let projection = to_point.dot(ray_direction);
    let closest_point = ray_origin + ray_direction * projection;
    (point - closest_point).magnitude()
}

pub fn ray_to_line_segment_distance(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    line_start: Vector3<f32>,
    line_end: Vector3<f32>,
) -> f32 {
    let line_dir = (line_end - line_start).normalize();
    let w0 = ray_origin - line_start;

    let a = ray_direction.dot(ray_direction);
    let b = ray_direction.dot(line_dir);
    let c = line_dir.dot(line_dir);
    let d = ray_direction.dot(w0);
    let e = line_dir.dot(w0);

    let denom = a * c - b * b;

    let (s, t) = if denom.abs() < EPSILON {
        (0.0, e / c)
    } else {
        let s = (b * e - c * d) / denom;
        let t = (a * e - b * d) / denom;
        (s, t)
    };

    let t_clamped = t.max(0.0).min((line_end - line_start).magnitude());

    let point_on_ray = ray_origin + ray_direction * s;
    let point_on_line = line_start + line_dir * t_clamped;

    (point_on_ray - point_on_line).magnitude()
}

pub fn ray_plane_intersection(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    plane_point: Vector3<f32>,
    plane_normal: Vector3<f32>,
) -> Option<Vector3<f32>> {
    let denom = plane_normal.dot(ray_direction);

    if denom.abs() < EPSILON {
        return None;
    }

    let t = (plane_point - ray_origin).dot(plane_normal) / denom;

    if t < 0.0 {
        return None;
    }

    Some(ray_origin + ray_direction * t)
}

pub fn get_camera_axes_from_view(view_matrix: Mat4) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>) {
    let view_inverse = match view_matrix.invert() {
        Some(inv) => inv,
        None => {
            return (
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 0.0, -1.0),
            )
        }
    };

    let camera_right = vec3(view_inverse.x.x, view_inverse.y.x, view_inverse.z.x);
    let camera_up = vec3(view_inverse.x.y, view_inverse.y.y, view_inverse.z.y);
    let camera_forward = -vec3(view_inverse.x.z, view_inverse.y.z, view_inverse.z.z);

    (camera_right, camera_up, camera_forward)
}

pub fn world_to_screen(
    world_pos: Vector3<f32>,
    screen_size: Vector2<f32>,
    view_matrix: Mat4,
    proj_matrix: Mat4,
) -> Option<Vector2<f32>> {
    let clip_pos =
        proj_matrix * view_matrix * cgmath::vec4(world_pos.x, world_pos.y, world_pos.z, 1.0);

    if clip_pos.w <= 0.0 {
        return None;
    }

    let ndc_x = clip_pos.x / clip_pos.w;
    let ndc_y = clip_pos.y / clip_pos.w;

    let screen_x = (ndc_x + 1.0) * 0.5 * screen_size.x;
    let screen_y = (ndc_y + 1.0) * 0.5 * screen_size.y;

    Some(Vector2::new(screen_x, screen_y))
}

pub fn ray_to_triangle_intersection(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    v0: Vector3<f32>,
    v1: Vector3<f32>,
    v2: Vector3<f32>,
) -> Option<f32> {
    ray_to_triangle_barycentric(ray_origin, ray_direction, v0, v1, v2).map(|(t, _, _)| t)
}

pub fn ray_to_triangle_barycentric(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    v0: Vector3<f32>,
    v1: Vector3<f32>,
    v2: Vector3<f32>,
) -> Option<(f32, f32, f32)> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = ray_direction.cross(edge2);
    let a = edge1.dot(h);

    if a.abs() < EPSILON {
        return None;
    }

    let f = 1.0 / a;
    let s = ray_origin - v0;
    let u = f * s.dot(h);

    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(edge1);
    let v = f * ray_direction.dot(q);

    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * edge2.dot(q);

    if t < EPSILON {
        return None;
    }

    Some((t, u, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{vec3, vec4, InnerSpace, Matrix4, SquareMatrix, Vector3};

    #[test]
    fn test_fbx_to_world() {
        let transform = fbx_to_world();
        let fbx_up = vec4(0.0, 0.0, 1.0, 0.0);
        let world_up = transform * fbx_up;
        assert_eq!(world_up, vec4(0.0, 1.0, 0.0, 0.0));
    }

    #[test]
    fn test_gltf_to_world() {
        assert_eq!(gltf_to_world(), Matrix4::identity());
    }

    #[test]
    fn test_world_axes() {
        assert_eq!(world_y_axis(), vec3(0.0, 1.0, 0.0));
        assert_eq!(world_y_down(), vec3(0.0, -1.0, 0.0));
    }

    #[test]
    fn test_perspective_y_flip() {
        let proj = perspective(Deg(45.0), 1.0, 0.1, 100.0);
        assert!(proj.y.y < 0.0);
    }

    #[test]
    fn test_perspective_infinite_reverse_y_flip() {
        let proj = perspective_infinite_reverse(Deg(45.0), 1.0, 0.1);
        assert!(proj.y.y < 0.0);
    }

    #[test]
    fn test_perspective_infinite_reverse_near_maps_to_one() {
        let near = 0.1;
        let proj = perspective_infinite_reverse(Deg(45.0), 1.0, near);
        let near_point = proj * vec4(0.0, 0.0, -near, 1.0);
        let ndc_z = near_point.z / near_point.w;
        assert!(
            (ndc_z - 1.0).abs() < 1e-5,
            "near should map to ndc_z=1.0, got {}",
            ndc_z
        );
    }

    #[test]
    fn test_perspective_infinite_reverse_far_maps_to_zero() {
        let proj = perspective_infinite_reverse(Deg(45.0), 1.0, 0.1);
        let far_point = proj * vec4(0.0, 0.0, -100000.0, 1.0);
        let ndc_z = far_point.z / far_point.w;
        assert!(
            ndc_z.abs() < 0.01,
            "infinity should map to ndc_z≈0.0, got {}",
            ndc_z
        );
    }

    #[test]
    fn test_right_handed() {
        let x = vec3(1.0, 0.0, 0.0);
        let y = world_y_axis();
        let z = vec3(0.0, 0.0, 1.0);
        assert!((x.cross(y) - z).magnitude() < 1e-5);
    }

    #[test]
    fn test_view_matrix() {
        unsafe {
            let camera_pos = Vector3::new(0.0, 0.0, -5.0);
            let direction = Vector3::new(0.0, 0.0, 1.0);
            let up = Vector3::new(0.0, 1.0, 0.0);

            let view_matrix = view(camera_pos, direction, up);

            assert_eq!(view_matrix.w.w, 1.0);
        }
    }

    #[test]
    fn test_fix_coord() {
        let m = fix_coord();
        assert_eq!(m.x.x, 1.0);
        assert_eq!(m.y.z, 1.0);
        assert_eq!(m.z.y, 1.0);
        assert_eq!(m.w.w, 1.0);
    }

    #[test]
    fn test_ray_triangle_intersection_hit() {
        let origin = vec3(0.0, 0.0, -1.0);
        let direction = vec3(0.0, 0.0, 1.0);
        let v0 = vec3(-1.0, -1.0, 0.0);
        let v1 = vec3(1.0, -1.0, 0.0);
        let v2 = vec3(0.0, 1.0, 0.0);

        let result = ray_to_triangle_intersection(origin, direction, v0, v1, v2);
        assert!(result.is_some());
        assert!((result.unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_ray_triangle_intersection_parallel() {
        let origin = vec3(0.0, 0.0, -1.0);
        let direction = vec3(1.0, 0.0, 0.0);
        let v0 = vec3(-1.0, -1.0, 0.0);
        let v1 = vec3(1.0, -1.0, 0.0);
        let v2 = vec3(0.0, 1.0, 0.0);

        let result = ray_to_triangle_intersection(origin, direction, v0, v1, v2);
        assert!(result.is_none());
    }

    #[test]
    fn test_ray_triangle_intersection_behind() {
        let origin = vec3(0.0, 0.0, 1.0);
        let direction = vec3(0.0, 0.0, 1.0);
        let v0 = vec3(-1.0, -1.0, 0.0);
        let v1 = vec3(1.0, -1.0, 0.0);
        let v2 = vec3(0.0, 1.0, 0.0);

        let result = ray_to_triangle_intersection(origin, direction, v0, v1, v2);
        assert!(result.is_none());
    }

    #[test]
    fn test_screen_to_world_ray_infinite_reverse_z_no_nan() {
        unsafe {
            let camera_pos = vec3(0.0, 1.0, -3.0);
            let direction = vec3(0.0, 0.0, 1.0);
            let up = vec3(0.0, 1.0, 0.0);

            let view_mat = view(camera_pos, direction, up);
            let proj_mat = perspective_infinite_reverse(Deg(45.0), 16.0 / 9.0, 0.1);

            let screen_center = Vector2::new(400.0, 300.0);
            let screen_size = Vector2::new(800.0, 600.0);

            let (origin, dir) = screen_to_world_ray(screen_center, screen_size, view_mat, proj_mat);

            assert!(!origin.x.is_nan() && !origin.y.is_nan() && !origin.z.is_nan());
            assert!(!dir.x.is_nan() && !dir.y.is_nan() && !dir.z.is_nan());
            assert!((dir.magnitude() - 1.0).abs() < 1e-4);
        }
    }
}
