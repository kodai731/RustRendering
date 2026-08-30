use cgmath::{InnerSpace, Vector3};

#[derive(Clone, Copy, Debug)]
pub struct TorusProjection {
    pub point: Vector3<f32>,
    pub normal: Vector3<f32>,
    pub u: f32,
    pub v: f32,
}

pub fn project_to_torus(p: Vector3<f32>, R: f32, r: f32) -> TorusProjection {
    let q = Vector3::new(p.x, 0.0, p.z);
    let q_mag = q.magnitude();

    let q_normalized = if q_mag < 1e-6 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        q / q_mag
    };

    let n = (p - R * q_normalized).normalize();
    let point = R * q_normalized + r * n;

    let u = q_normalized.z.atan2(q_normalized.x);
    let v = n.y.atan2(n.dot(q_normalized));

    TorusProjection {
        point,
        normal: n,
        u,
        v,
    }
}

pub fn water_surface_point(u: f32, v: f32, R: f32, r: f32) -> Vector3<f32> {
    let cos_v = v.cos();
    let sin_v = v.sin();
    let cos_u = u.cos();
    let sin_u = u.sin();

    let radius = R + r * cos_v;
    Vector3::new(radius * cos_u, r * sin_v, radius * sin_u)
}

pub fn water_surface_normal(u: f32, v: f32) -> Vector3<f32> {
    let cos_v = v.cos();
    let sin_v = v.sin();
    let cos_u = u.cos();
    let sin_u = u.sin();

    Vector3::new(cos_v * cos_u, sin_v, cos_v * sin_u)
}
