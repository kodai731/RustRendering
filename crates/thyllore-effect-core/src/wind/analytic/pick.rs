use crate::wind::analytic::shell_integral::{
    clamp_ray_to_wind_cone, wind_envelope_radius, WindShellParams,
};
use cgmath::{InnerSpace, Matrix4, Vector3};

const PICK_T_MAX: f32 = 1e4;

/// Distance along the world ray at which it enters the tornado envelope, if it does.
pub fn pick_wind(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    inverse_model: Matrix4<f32>,
    params: &WindShellParams,
) -> Option<f32> {
    let local_origin = (inverse_model * ray_origin.extend(1.0)).truncate();
    let local_direction = (inverse_model * ray_direction.extend(0.0))
        .truncate()
        .normalize();

    let mut t_near = 0.0f32;
    let mut t_far = PICK_T_MAX;
    if !clamp_ray_to_wind_cone(
        params,
        local_origin,
        local_direction,
        &mut t_near,
        &mut t_far,
    ) {
        return None;
    }
    Some(t_near.max(0.0))
}

/// Corners of the local axis-aligned box enclosing the envelope cone frustum.
pub fn wind_local_bounds_corners(params: &WindShellParams) -> [Vector3<f32>; 8] {
    let radius = wind_envelope_radius(params, 0.0)
        .max(wind_envelope_radius(params, 1.0))
        .max(params.ring_bounds_radius());
    let mut corners = [Vector3::new(0.0, 0.0, 0.0); 8];
    for (index, corner) in corners.iter_mut().enumerate() {
        corner.x = if index & 1 == 0 { -radius } else { radius };
        corner.y = if index & 2 == 0 { 0.0 } else { params.height };
        corner.z = if index & 4 == 0 { -radius } else { radius };
    }
    corners
}
