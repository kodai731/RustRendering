use std::f32::consts::PI;

use cgmath::{vec3, Deg, InnerSpace, Quaternion, Rotation3, Vector2, Vector3};

use crate::ecs::component::{ColorVertex, LineMesh};
use crate::ecs::resource::gizmo::transform_gizmo::{TransformGizmoData, TransformGizmoHandle};
use crate::ecs::resource::TransformGizmoMode;
use crate::math::{
    coordinate_system::perspective_infinite_reverse, ray_plane_intersection,
    ray_to_line_segment_distance, ray_to_point_distance, screen_to_world_ray, view,
};

const RED: [f32; 3] = [0.9, 0.1, 0.1];
const GREEN: [f32; 3] = [0.1, 0.9, 0.1];
const BLUE: [f32; 3] = [0.1, 0.1, 0.9];
const YELLOW: [f32; 3] = [1.0, 1.0, 0.3];
const WHITE: [f32; 3] = [0.8, 0.8, 0.8];
const PLANE_XY: [f32; 3] = [0.3, 0.3, 0.9];
const PLANE_XZ: [f32; 3] = [0.3, 0.9, 0.3];
const PLANE_YZ: [f32; 3] = [0.9, 0.3, 0.3];

pub fn build_translate_gizmo_meshes(
    active_handle: TransformGizmoHandle,
    line_mesh: &mut LineMesh,
    solid_mesh: &mut LineMesh,
) {
    line_mesh.vertices.clear();
    line_mesh.indices.clear();
    solid_mesh.vertices.clear();
    solid_mesh.indices.clear();

    let shaft_length = 1.0f32;
    let cone_base = 0.7;
    let cone_radius = 0.06;
    let cone_segments = 8;
    let plane_offset = 0.25;
    let plane_size = 0.15;

    let x_color = resolve_handle_color(TransformGizmoHandle::AxisX, active_handle, RED);
    let y_color = resolve_handle_color(TransformGizmoHandle::AxisY, active_handle, GREEN);
    let z_color = resolve_handle_color(TransformGizmoHandle::AxisZ, active_handle, BLUE);

    push_line(
        line_mesh,
        [0.0, 0.0, 0.0],
        [shaft_length, 0.0, 0.0],
        x_color,
    );
    push_line(
        line_mesh,
        [0.0, 0.0, 0.0],
        [0.0, shaft_length, 0.0],
        y_color,
    );
    push_line(
        line_mesh,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, shaft_length],
        z_color,
    );

    generate_cone_vertices(
        vec3(cone_base, 0.0, 0.0),
        vec3(shaft_length, 0.0, 0.0),
        cone_radius,
        cone_segments,
        x_color,
        &mut solid_mesh.vertices,
        &mut solid_mesh.indices,
    );
    generate_cone_vertices(
        vec3(0.0, cone_base, 0.0),
        vec3(0.0, shaft_length, 0.0),
        cone_radius,
        cone_segments,
        y_color,
        &mut solid_mesh.vertices,
        &mut solid_mesh.indices,
    );
    generate_cone_vertices(
        vec3(0.0, 0.0, cone_base),
        vec3(0.0, 0.0, shaft_length),
        cone_radius,
        cone_segments,
        z_color,
        &mut solid_mesh.vertices,
        &mut solid_mesh.indices,
    );

    let xy_color = resolve_handle_color(TransformGizmoHandle::PlaneXY, active_handle, PLANE_XY);
    let xz_color = resolve_handle_color(TransformGizmoHandle::PlaneXZ, active_handle, PLANE_XZ);
    let yz_color = resolve_handle_color(TransformGizmoHandle::PlaneYZ, active_handle, PLANE_YZ);

    push_quad(
        solid_mesh,
        [plane_offset, plane_offset, 0.0],
        [plane_offset + plane_size, plane_offset, 0.0],
        [plane_offset + plane_size, plane_offset + plane_size, 0.0],
        [plane_offset, plane_offset + plane_size, 0.0],
        xy_color,
    );
    push_quad(
        solid_mesh,
        [plane_offset, 0.0, plane_offset],
        [plane_offset + plane_size, 0.0, plane_offset],
        [plane_offset + plane_size, 0.0, plane_offset + plane_size],
        [plane_offset, 0.0, plane_offset + plane_size],
        xz_color,
    );
    push_quad(
        solid_mesh,
        [0.0, plane_offset, plane_offset],
        [0.0, plane_offset + plane_size, plane_offset],
        [0.0, plane_offset + plane_size, plane_offset + plane_size],
        [0.0, plane_offset, plane_offset + plane_size],
        yz_color,
    );
}

pub fn build_rotate_gizmo_meshes(
    active_handle: TransformGizmoHandle,
    camera_dir: Vector3<f32>,
    line_mesh: &mut LineMesh,
    solid_mesh: &mut LineMesh,
) {
    line_mesh.vertices.clear();
    line_mesh.indices.clear();
    solid_mesh.vertices.clear();
    solid_mesh.indices.clear();

    let ring_segments = 64;
    let radius = 0.9;

    let x_color = resolve_handle_color(TransformGizmoHandle::RingX, active_handle, RED);
    let y_color = resolve_handle_color(TransformGizmoHandle::RingY, active_handle, GREEN);
    let z_color = resolve_handle_color(TransformGizmoHandle::RingZ, active_handle, BLUE);
    let tb_color = resolve_handle_color(TransformGizmoHandle::Trackball, active_handle, WHITE);

    generate_circle_vertices(
        vec3(1.0, 0.0, 0.0),
        radius,
        ring_segments,
        x_color,
        &mut line_mesh.vertices,
        &mut line_mesh.indices,
    );
    generate_circle_vertices(
        vec3(0.0, 1.0, 0.0),
        radius,
        ring_segments,
        y_color,
        &mut line_mesh.vertices,
        &mut line_mesh.indices,
    );
    generate_circle_vertices(
        vec3(0.0, 0.0, 1.0),
        radius,
        ring_segments,
        z_color,
        &mut line_mesh.vertices,
        &mut line_mesh.indices,
    );
    generate_circle_vertices(
        -camera_dir.normalize(),
        radius * 1.1,
        ring_segments,
        tb_color,
        &mut line_mesh.vertices,
        &mut line_mesh.indices,
    );
}

pub fn build_scale_gizmo_meshes(
    active_handle: TransformGizmoHandle,
    line_mesh: &mut LineMesh,
    solid_mesh: &mut LineMesh,
) {
    line_mesh.vertices.clear();
    line_mesh.indices.clear();
    solid_mesh.vertices.clear();
    solid_mesh.indices.clear();

    let shaft_length = 1.0f32;
    let cube_half = 0.05;

    let x_color = resolve_handle_color(TransformGizmoHandle::AxisX, active_handle, RED);
    let y_color = resolve_handle_color(TransformGizmoHandle::AxisY, active_handle, GREEN);
    let z_color = resolve_handle_color(TransformGizmoHandle::AxisZ, active_handle, BLUE);
    let center_color = resolve_handle_color(TransformGizmoHandle::Center, active_handle, WHITE);

    push_line(
        line_mesh,
        [0.0, 0.0, 0.0],
        [shaft_length, 0.0, 0.0],
        x_color,
    );
    push_line(
        line_mesh,
        [0.0, 0.0, 0.0],
        [0.0, shaft_length, 0.0],
        y_color,
    );
    push_line(
        line_mesh,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, shaft_length],
        z_color,
    );

    generate_cube_vertices(
        vec3(shaft_length, 0.0, 0.0),
        cube_half,
        x_color,
        &mut solid_mesh.vertices,
        &mut solid_mesh.indices,
    );
    generate_cube_vertices(
        vec3(0.0, shaft_length, 0.0),
        cube_half,
        y_color,
        &mut solid_mesh.vertices,
        &mut solid_mesh.indices,
    );
    generate_cube_vertices(
        vec3(0.0, 0.0, shaft_length),
        cube_half,
        z_color,
        &mut solid_mesh.vertices,
        &mut solid_mesh.indices,
    );

    generate_cube_vertices(
        vec3(0.0, 0.0, 0.0),
        cube_half * 1.5,
        center_color,
        &mut solid_mesh.vertices,
        &mut solid_mesh.indices,
    );
}

pub fn transform_gizmo_try_select(
    gizmo: &TransformGizmoData,
    mode: TransformGizmoMode,
    mouse_pos: Vector2<f32>,
    camera_pos: Vector3<f32>,
    camera_dir: Vector3<f32>,
    camera_up: Vector3<f32>,
    swapchain_extent: (u32, u32),
    fov_y: Deg<f32>,
    near_plane: f32,
    gizmo_scale: f32,
) -> TransformGizmoHandle {
    let gizmo_pos = gizmo.position.position;
    let view_mat = unsafe { view(camera_pos, camera_dir, camera_up) };
    let aspect = swapchain_extent.0 as f32 / swapchain_extent.1 as f32;
    let proj = perspective_infinite_reverse(fov_y, aspect, near_plane);
    let screen_size = Vector2::new(swapchain_extent.0 as f32, swapchain_extent.1 as f32);

    let (ray_origin, ray_direction) = screen_to_world_ray(mouse_pos, screen_size, view_mat, proj);

    let distance = (gizmo_pos - camera_pos).magnitude();
    let scale_factor = distance * gizmo_scale;
    let threshold = 0.05 * scale_factor;

    match mode {
        TransformGizmoMode::Translate => select_translate_handle(
            ray_origin,
            ray_direction,
            gizmo_pos,
            scale_factor,
            threshold,
        ),
        TransformGizmoMode::Rotate => select_rotate_handle(
            ray_origin,
            ray_direction,
            gizmo_pos,
            scale_factor,
            threshold,
            camera_dir,
        ),
        TransformGizmoMode::Scale => select_scale_handle(
            ray_origin,
            ray_direction,
            gizmo_pos,
            scale_factor,
            threshold,
        ),
    }
}

fn select_translate_handle(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    gizmo_pos: Vector3<f32>,
    scale_factor: f32,
    threshold: f32,
) -> TransformGizmoHandle {
    let x_end = gizmo_pos + vec3(1.0 * scale_factor, 0.0, 0.0);
    let y_end = gizmo_pos + vec3(0.0, 1.0 * scale_factor, 0.0);
    let z_end = gizmo_pos + vec3(0.0, 0.0, 1.0 * scale_factor);

    let x_dist = ray_to_line_segment_distance(ray_origin, ray_direction, gizmo_pos, x_end);
    let y_dist = ray_to_line_segment_distance(ray_origin, ray_direction, gizmo_pos, y_end);
    let z_dist = ray_to_line_segment_distance(ray_origin, ray_direction, gizmo_pos, z_end);
    let center_dist = ray_to_point_distance(ray_origin, ray_direction, gizmo_pos);

    let plane_offset = 0.25 * scale_factor;
    let plane_size = 0.15 * scale_factor;

    let xy_hit = check_plane_handle_hit(
        ray_origin,
        ray_direction,
        gizmo_pos,
        vec3(0.0, 0.0, 1.0),
        plane_offset,
        plane_size,
        |p| (p.x, p.y),
    );
    let xz_hit = check_plane_handle_hit(
        ray_origin,
        ray_direction,
        gizmo_pos,
        vec3(0.0, 1.0, 0.0),
        plane_offset,
        plane_size,
        |p| (p.x, p.z),
    );
    let yz_hit = check_plane_handle_hit(
        ray_origin,
        ray_direction,
        gizmo_pos,
        vec3(1.0, 0.0, 0.0),
        plane_offset,
        plane_size,
        |p| (p.y, p.z),
    );

    if xy_hit {
        return TransformGizmoHandle::PlaneXY;
    }
    if xz_hit {
        return TransformGizmoHandle::PlaneXZ;
    }
    if yz_hit {
        return TransformGizmoHandle::PlaneYZ;
    }

    let mut min_dist = f32::MAX;
    let mut selected = TransformGizmoHandle::None;

    if center_dist < threshold * 2.0 {
        min_dist = center_dist;
        selected = TransformGizmoHandle::Center;
    }
    if x_dist < threshold && x_dist < min_dist {
        min_dist = x_dist;
        selected = TransformGizmoHandle::AxisX;
    }
    if y_dist < threshold && y_dist < min_dist {
        min_dist = y_dist;
        selected = TransformGizmoHandle::AxisY;
    }
    if z_dist < threshold && z_dist < min_dist {
        min_dist = z_dist;
        selected = TransformGizmoHandle::AxisZ;
    }
    let _ = min_dist;

    selected
}

fn select_rotate_handle(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    gizmo_pos: Vector3<f32>,
    scale_factor: f32,
    threshold: f32,
    camera_dir: Vector3<f32>,
) -> TransformGizmoHandle {
    let radius = 0.9 * scale_factor;

    let x_dist = compute_ray_to_circle_distance(
        ray_origin,
        ray_direction,
        gizmo_pos,
        vec3(1.0, 0.0, 0.0),
        radius,
    );
    let y_dist = compute_ray_to_circle_distance(
        ray_origin,
        ray_direction,
        gizmo_pos,
        vec3(0.0, 1.0, 0.0),
        radius,
    );
    let z_dist = compute_ray_to_circle_distance(
        ray_origin,
        ray_direction,
        gizmo_pos,
        vec3(0.0, 0.0, 1.0),
        radius,
    );
    let tb_dist = compute_ray_to_circle_distance(
        ray_origin,
        ray_direction,
        gizmo_pos,
        -camera_dir.normalize(),
        radius * 1.1,
    );

    let mut min_dist = f32::MAX;
    let mut selected = TransformGizmoHandle::None;

    if tb_dist < threshold && tb_dist < min_dist {
        min_dist = tb_dist;
        selected = TransformGizmoHandle::Trackball;
    }
    if x_dist < threshold && x_dist < min_dist {
        min_dist = x_dist;
        selected = TransformGizmoHandle::RingX;
    }
    if y_dist < threshold && y_dist < min_dist {
        min_dist = y_dist;
        selected = TransformGizmoHandle::RingY;
    }
    if z_dist < threshold && z_dist < min_dist {
        min_dist = z_dist;
        selected = TransformGizmoHandle::RingZ;
    }
    let _ = min_dist;

    selected
}

fn select_scale_handle(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    gizmo_pos: Vector3<f32>,
    scale_factor: f32,
    threshold: f32,
) -> TransformGizmoHandle {
    let x_end = gizmo_pos + vec3(1.0 * scale_factor, 0.0, 0.0);
    let y_end = gizmo_pos + vec3(0.0, 1.0 * scale_factor, 0.0);
    let z_end = gizmo_pos + vec3(0.0, 0.0, 1.0 * scale_factor);

    let x_dist = ray_to_line_segment_distance(ray_origin, ray_direction, gizmo_pos, x_end);
    let y_dist = ray_to_line_segment_distance(ray_origin, ray_direction, gizmo_pos, y_end);
    let z_dist = ray_to_line_segment_distance(ray_origin, ray_direction, gizmo_pos, z_end);
    let center_dist = ray_to_point_distance(ray_origin, ray_direction, gizmo_pos);

    let mut min_dist = f32::MAX;
    let mut selected = TransformGizmoHandle::None;

    if center_dist < threshold * 2.0 {
        min_dist = center_dist;
        selected = TransformGizmoHandle::Center;
    }
    if x_dist < threshold && x_dist < min_dist {
        min_dist = x_dist;
        selected = TransformGizmoHandle::AxisX;
    }
    if y_dist < threshold && y_dist < min_dist {
        min_dist = y_dist;
        selected = TransformGizmoHandle::AxisY;
    }
    if z_dist < threshold && z_dist < min_dist {
        min_dist = z_dist;
        selected = TransformGizmoHandle::AxisZ;
    }
    let _ = min_dist;

    selected
}

pub fn transform_gizmo_compute_drag_plane(
    handle: TransformGizmoHandle,
    gizmo_pos: Vector3<f32>,
    camera_dir: Vector3<f32>,
) -> (Vector3<f32>, Vector3<f32>) {
    let plane_normal = match handle {
        TransformGizmoHandle::AxisX => {
            compute_best_axis_drag_plane_normal(vec3(1.0, 0.0, 0.0), camera_dir)
        }
        TransformGizmoHandle::AxisY => {
            compute_best_axis_drag_plane_normal(vec3(0.0, 1.0, 0.0), camera_dir)
        }
        TransformGizmoHandle::AxisZ => {
            compute_best_axis_drag_plane_normal(vec3(0.0, 0.0, 1.0), camera_dir)
        }
        TransformGizmoHandle::PlaneXY => vec3(0.0, 0.0, 1.0),
        TransformGizmoHandle::PlaneXZ => vec3(0.0, 1.0, 0.0),
        TransformGizmoHandle::PlaneYZ => vec3(1.0, 0.0, 0.0),
        TransformGizmoHandle::Center | TransformGizmoHandle::Trackball => -camera_dir.normalize(),
        TransformGizmoHandle::RingX => vec3(1.0, 0.0, 0.0),
        TransformGizmoHandle::RingY => vec3(0.0, 1.0, 0.0),
        TransformGizmoHandle::RingZ => vec3(0.0, 0.0, 1.0),
        TransformGizmoHandle::None => -camera_dir.normalize(),
    };

    (gizmo_pos, plane_normal)
}

fn compute_best_axis_drag_plane_normal(
    axis: Vector3<f32>,
    camera_dir: Vector3<f32>,
) -> Vector3<f32> {
    let cam_norm = camera_dir.normalize();
    let tangent = axis.cross(cam_norm);
    if tangent.magnitude() < 1e-6 {
        return cam_norm;
    }
    tangent.cross(axis).normalize()
}

pub fn transform_gizmo_process_translate_drag(
    gizmo: &TransformGizmoData,
    mouse_pos: Vector2<f32>,
    camera_pos: Vector3<f32>,
    camera_dir: Vector3<f32>,
    camera_up: Vector3<f32>,
    swapchain_extent: (u32, u32),
    fov_y: Deg<f32>,
    near_plane: f32,
    snap: Option<f32>,
) -> Option<Vector3<f32>> {
    let view_mat = unsafe { view(camera_pos, camera_dir, camera_up) };
    let aspect = swapchain_extent.0 as f32 / swapchain_extent.1 as f32;
    let proj = perspective_infinite_reverse(fov_y, aspect, near_plane);
    let screen_size = Vector2::new(swapchain_extent.0 as f32, swapchain_extent.1 as f32);

    let (ray_origin, ray_direction) = screen_to_world_ray(mouse_pos, screen_size, view_mat, proj);

    let hit = ray_plane_intersection(
        ray_origin,
        ray_direction,
        gizmo.position.position,
        gizmo.drag_plane_normal,
    )?;

    let raw_delta = hit - gizmo.drag_initial_hit;

    let constrained_delta = match gizmo.active_handle {
        TransformGizmoHandle::AxisX => vec3(raw_delta.x, 0.0, 0.0),
        TransformGizmoHandle::AxisY => vec3(0.0, raw_delta.y, 0.0),
        TransformGizmoHandle::AxisZ => vec3(0.0, 0.0, raw_delta.z),
        TransformGizmoHandle::PlaneXY => vec3(raw_delta.x, raw_delta.y, 0.0),
        TransformGizmoHandle::PlaneXZ => vec3(raw_delta.x, 0.0, raw_delta.z),
        TransformGizmoHandle::PlaneYZ => vec3(0.0, raw_delta.y, raw_delta.z),
        TransformGizmoHandle::Center => raw_delta,
        TransformGizmoHandle::None
        | TransformGizmoHandle::RingX
        | TransformGizmoHandle::RingY
        | TransformGizmoHandle::RingZ
        | TransformGizmoHandle::Trackball => return None,
    };

    let snapped = if let Some(snap_val) = snap {
        vec3(
            apply_snap_value(constrained_delta.x, snap_val),
            apply_snap_value(constrained_delta.y, snap_val),
            apply_snap_value(constrained_delta.z, snap_val),
        )
    } else {
        constrained_delta
    };

    Some(gizmo.drag_start_position + snapped)
}

pub fn transform_gizmo_process_rotate_drag(
    gizmo: &TransformGizmoData,
    mouse_pos: Vector2<f32>,
    camera_pos: Vector3<f32>,
    camera_dir: Vector3<f32>,
    camera_up: Vector3<f32>,
    swapchain_extent: (u32, u32),
    fov_y: Deg<f32>,
    near_plane: f32,
    snap_degrees: Option<f32>,
) -> Option<Quaternion<f32>> {
    let view_mat = unsafe { view(camera_pos, camera_dir, camera_up) };
    let aspect = swapchain_extent.0 as f32 / swapchain_extent.1 as f32;
    let proj = perspective_infinite_reverse(fov_y, aspect, near_plane);
    let screen_size = Vector2::new(swapchain_extent.0 as f32, swapchain_extent.1 as f32);

    let (ray_origin, ray_direction) = screen_to_world_ray(mouse_pos, screen_size, view_mat, proj);

    let rotation_axis = gizmo.drag_plane_normal.normalize();

    let hit = ray_plane_intersection(
        ray_origin,
        ray_direction,
        gizmo.position.position,
        rotation_axis,
    )?;

    let current_vec = (hit - gizmo.position.position).normalize();
    let initial_vec = (gizmo.drag_initial_hit - gizmo.position.position).normalize();

    if current_vec.magnitude() < 1e-6 || initial_vec.magnitude() < 1e-6 {
        return None;
    }

    let dot = initial_vec.dot(current_vec).min(1.0).max(-1.0);
    let cross = initial_vec.cross(current_vec);
    let sign = if cross.dot(rotation_axis) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let mut angle = sign * dot.acos();

    if let Some(snap_deg) = snap_degrees {
        let snap_rad = snap_deg.to_radians();
        angle = (angle / snap_rad).round() * snap_rad;
    }

    Some(Quaternion::from_axis_angle(
        rotation_axis,
        cgmath::Rad(angle),
    ))
}

pub fn transform_gizmo_process_scale_drag(
    gizmo: &TransformGizmoData,
    mouse_pos: Vector2<f32>,
    camera_pos: Vector3<f32>,
    camera_dir: Vector3<f32>,
    camera_up: Vector3<f32>,
    swapchain_extent: (u32, u32),
    fov_y: Deg<f32>,
    near_plane: f32,
    snap: Option<f32>,
) -> Option<Vector3<f32>> {
    let view_mat = unsafe { view(camera_pos, camera_dir, camera_up) };
    let aspect = swapchain_extent.0 as f32 / swapchain_extent.1 as f32;
    let proj = perspective_infinite_reverse(fov_y, aspect, near_plane);
    let screen_size = Vector2::new(swapchain_extent.0 as f32, swapchain_extent.1 as f32);

    let (ray_origin, ray_direction) = screen_to_world_ray(mouse_pos, screen_size, view_mat, proj);

    let hit = ray_plane_intersection(
        ray_origin,
        ray_direction,
        gizmo.position.position,
        gizmo.drag_plane_normal,
    )?;

    let initial_dist = (gizmo.drag_initial_hit - gizmo.position.position).magnitude();
    if initial_dist < 1e-6 {
        return None;
    }

    let current_delta = hit - gizmo.position.position;

    let scale = match gizmo.active_handle {
        TransformGizmoHandle::AxisX => {
            let ratio =
                current_delta.x / (gizmo.drag_initial_hit.x - gizmo.position.position.x).max(1e-6);
            vec3(ratio, 1.0, 1.0)
        }
        TransformGizmoHandle::AxisY => {
            let ratio =
                current_delta.y / (gizmo.drag_initial_hit.y - gizmo.position.position.y).max(1e-6);
            vec3(1.0, ratio, 1.0)
        }
        TransformGizmoHandle::AxisZ => {
            let ratio =
                current_delta.z / (gizmo.drag_initial_hit.z - gizmo.position.position.z).max(1e-6);
            vec3(1.0, 1.0, ratio)
        }
        TransformGizmoHandle::Center => {
            let current_dist = current_delta.magnitude();
            let ratio = current_dist / initial_dist;
            vec3(ratio, ratio, ratio)
        }
        TransformGizmoHandle::None
        | TransformGizmoHandle::PlaneXY
        | TransformGizmoHandle::PlaneXZ
        | TransformGizmoHandle::PlaneYZ
        | TransformGizmoHandle::RingX
        | TransformGizmoHandle::RingY
        | TransformGizmoHandle::RingZ
        | TransformGizmoHandle::Trackball => return None,
    };

    let final_scale = if let Some(snap_val) = snap {
        vec3(
            apply_snap_value(scale.x, snap_val),
            apply_snap_value(scale.y, snap_val),
            apply_snap_value(scale.z, snap_val),
        )
    } else {
        scale
    };

    Some(vec3(
        gizmo.drag_start_scale.x * final_scale.x,
        gizmo.drag_start_scale.y * final_scale.y,
        gizmo.drag_start_scale.z * final_scale.z,
    ))
}

pub fn transform_gizmo_sync_to_bone(
    gizmo: &mut TransformGizmoData,
    active_bone_index: Option<usize>,
    cached_transforms: &[cgmath::Matrix4<f32>],
    bone_offsets: &[[f32; 3]],
    mesh_scale: f32,
) {
    match active_bone_index {
        Some(idx) if idx < cached_transforms.len() => {
            let transform = cached_transforms[idx];
            let offset = if idx < bone_offsets.len() {
                bone_offsets[idx]
            } else {
                [0.0, 0.0, 0.0]
            };
            let world_pos = transform * cgmath::Vector4::new(offset[0], offset[1], offset[2], 1.0);
            let raw_pos = vec3(world_pos.x, world_pos.y, world_pos.z);
            gizmo.position.position = vec3(
                world_pos.x * mesh_scale,
                world_pos.y * mesh_scale,
                world_pos.z * mesh_scale,
            );
            if !gizmo.visible {
                log!(
                    "[GizmoSync] bone={}, raw=({:.4},{:.4},{:.4}), scaled=({:.4},{:.4},{:.4}), mesh_scale={}",
                    idx, raw_pos.x, raw_pos.y, raw_pos.z,
                    gizmo.position.position.x, gizmo.position.position.y, gizmo.position.position.z,
                    mesh_scale,
                );
            }
            gizmo.visible = true;
            gizmo.target_bone_id = Some(idx as u32);
        }
        _ => {
            gizmo.visible = false;
            gizmo.target_bone_id = None;
        }
    }
}

pub fn transform_gizmo_sync_to_entity(
    gizmo: &mut TransformGizmoData,
    world_position: Vector3<f32>,
    entity: crate::ecs::Entity,
) {
    gizmo.position.position = world_position;
    gizmo.visible = true;
    gizmo.target_entity = Some(entity);
    gizmo.target_bone_id = None;
}

fn generate_cone_vertices(
    base_center: Vector3<f32>,
    tip: Vector3<f32>,
    radius: f32,
    segments: u32,
    color: [f32; 3],
    verts: &mut Vec<ColorVertex>,
    indices: &mut Vec<u32>,
) {
    let axis = (tip - base_center).normalize();
    let (tangent, bitangent) = compute_orthonormal_basis(axis);

    let base_idx = verts.len() as u32;
    let tip_idx = base_idx;

    verts.push(ColorVertex {
        pos: [tip.x, tip.y, tip.z],
        color,
    });

    for i in 0..segments {
        let angle = 2.0 * PI * (i as f32) / (segments as f32);
        let offset = tangent * angle.cos() * radius + bitangent * angle.sin() * radius;
        let p = base_center + offset;
        verts.push(ColorVertex {
            pos: [p.x, p.y, p.z],
            color,
        });
    }

    let base_center_idx = verts.len() as u32;
    verts.push(ColorVertex {
        pos: [base_center.x, base_center.y, base_center.z],
        color,
    });

    for i in 0..segments {
        let next = (i + 1) % segments;
        indices.push(tip_idx);
        indices.push(base_idx + 1 + i);
        indices.push(base_idx + 1 + next);

        indices.push(base_center_idx);
        indices.push(base_idx + 1 + next);
        indices.push(base_idx + 1 + i);
    }
}

fn generate_cube_vertices(
    center: Vector3<f32>,
    half_size: f32,
    color: [f32; 3],
    verts: &mut Vec<ColorVertex>,
    indices: &mut Vec<u32>,
) {
    let h = half_size;
    let base = verts.len() as u32;

    let corners = [
        [-h, -h, -h],
        [h, -h, -h],
        [h, h, -h],
        [-h, h, -h],
        [-h, -h, h],
        [h, -h, h],
        [h, h, h],
        [-h, h, h],
    ];
    for c in &corners {
        verts.push(ColorVertex {
            pos: [center.x + c[0], center.y + c[1], center.z + c[2]],
            color,
        });
    }

    let face_indices: [u32; 36] = [
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 2, 6, 7, 2, 7, 3, 0, 7, 4, 0, 3, 7,
        1, 5, 6, 1, 6, 2,
    ];
    for fi in &face_indices {
        indices.push(base + fi);
    }
}

fn generate_circle_vertices(
    normal: Vector3<f32>,
    radius: f32,
    segments: u32,
    color: [f32; 3],
    verts: &mut Vec<ColorVertex>,
    indices: &mut Vec<u32>,
) {
    let (tangent, bitangent) = compute_orthonormal_basis(normal);
    let base = verts.len() as u32;

    for i in 0..segments {
        let angle = 2.0 * PI * (i as f32) / (segments as f32);
        let p = tangent * angle.cos() * radius + bitangent * angle.sin() * radius;
        verts.push(ColorVertex {
            pos: [p.x, p.y, p.z],
            color,
        });
    }

    for i in 0..segments {
        let next = (i + 1) % segments;
        indices.push(base + i);
        indices.push(base + next);
    }
}

fn compute_orthonormal_basis(normal: Vector3<f32>) -> (Vector3<f32>, Vector3<f32>) {
    let n = normal.normalize();
    let up = if n.y.abs() < 0.99 {
        vec3(0.0, 1.0, 0.0)
    } else {
        vec3(1.0, 0.0, 0.0)
    };
    let tangent = n.cross(up).normalize();
    let bitangent = tangent.cross(n).normalize();
    (tangent, bitangent)
}

fn push_line(mesh: &mut LineMesh, start: [f32; 3], end: [f32; 3], color: [f32; 3]) {
    let base = mesh.vertices.len() as u32;
    mesh.vertices.push(ColorVertex { pos: start, color });
    mesh.vertices.push(ColorVertex { pos: end, color });
    mesh.indices.push(base);
    mesh.indices.push(base + 1);
}

fn push_quad(
    mesh: &mut LineMesh,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    d: [f32; 3],
    color: [f32; 3],
) {
    let base = mesh.vertices.len() as u32;
    mesh.vertices.push(ColorVertex { pos: a, color });
    mesh.vertices.push(ColorVertex { pos: b, color });
    mesh.vertices.push(ColorVertex { pos: c, color });
    mesh.vertices.push(ColorVertex { pos: d, color });
    indices_push_tri(&mut mesh.indices, base, base + 1, base + 2);
    indices_push_tri(&mut mesh.indices, base, base + 2, base + 3);
}

fn indices_push_tri(indices: &mut Vec<u32>, a: u32, b: u32, c: u32) {
    indices.push(a);
    indices.push(b);
    indices.push(c);
}

fn resolve_handle_color(
    handle: TransformGizmoHandle,
    active: TransformGizmoHandle,
    default_color: [f32; 3],
) -> [f32; 3] {
    if handle == active {
        YELLOW
    } else {
        default_color
    }
}

fn apply_snap_value(val: f32, snap: f32) -> f32 {
    if snap.abs() < f32::EPSILON {
        return val;
    }
    (val / snap).round() * snap
}

fn check_plane_handle_hit(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    gizmo_pos: Vector3<f32>,
    plane_normal: Vector3<f32>,
    plane_offset: f32,
    plane_size: f32,
    extract_2d: impl Fn(Vector3<f32>) -> (f32, f32),
) -> bool {
    let hit = ray_plane_intersection(ray_origin, ray_direction, gizmo_pos, plane_normal);
    match hit {
        Some(point) => {
            let local = point - gizmo_pos;
            let (u, v) = extract_2d(local);
            u >= plane_offset
                && u <= plane_offset + plane_size
                && v >= plane_offset
                && v <= plane_offset + plane_size
        }
        None => false,
    }
}

pub fn compute_ray_to_circle_distance(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    center: Vector3<f32>,
    normal: Vector3<f32>,
    radius: f32,
) -> f32 {
    let n = normal.normalize();
    let denom = n.dot(ray_direction);

    if denom.abs() < 1e-6 {
        return f32::MAX;
    }

    let t = (center - ray_origin).dot(n) / denom;
    if t < 0.0 {
        return f32::MAX;
    }

    let hit = ray_origin + ray_direction * t;
    let to_hit = hit - center;
    let dist_from_center = to_hit.magnitude();

    (dist_from_center - radius).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::resource::{CoordinateSpace, TransformGizmoState};

    #[test]
    fn test_cone_vertex_generation() {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        generate_cone_vertices(
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            0.1,
            8,
            RED,
            &mut verts,
            &mut indices,
        );
        // 1 tip + 8 base + 1 base_center = 10 vertices
        assert_eq!(verts.len(), 10);
        // 8 side triangles + 8 bottom cap triangles = 48 indices
        assert_eq!(indices.len(), 48);
    }

    #[test]
    fn test_cube_vertex_generation() {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        generate_cube_vertices(vec3(0.0, 0.0, 0.0), 0.5, RED, &mut verts, &mut indices);
        assert_eq!(verts.len(), 8);
        assert_eq!(indices.len(), 36);
    }

    #[test]
    fn test_circle_vertex_generation() {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        generate_circle_vertices(
            vec3(0.0, 1.0, 0.0),
            1.0,
            64,
            GREEN,
            &mut verts,
            &mut indices,
        );
        assert_eq!(verts.len(), 64);
        assert_eq!(indices.len(), 128); // 64 lines * 2
    }

    #[test]
    fn test_translate_mesh_build() {
        let mut line = LineMesh::default();
        let mut solid = LineMesh::default();
        build_translate_gizmo_meshes(TransformGizmoHandle::None, &mut line, &mut solid);
        assert!(line.vertices.len() >= 6); // 3 axis lines * 2 verts
        assert!(solid.vertices.len() > 0); // cones + quads
    }

    #[test]
    fn test_rotate_mesh_build() {
        let mut line = LineMesh::default();
        let mut solid = LineMesh::default();
        build_rotate_gizmo_meshes(
            TransformGizmoHandle::None,
            vec3(0.0, 0.0, -1.0),
            &mut line,
            &mut solid,
        );
        // 4 rings * 64 verts each
        assert_eq!(line.vertices.len(), 256);
    }

    #[test]
    fn test_scale_mesh_build() {
        let mut line = LineMesh::default();
        let mut solid = LineMesh::default();
        build_scale_gizmo_meshes(TransformGizmoHandle::None, &mut line, &mut solid);
        assert!(line.vertices.len() >= 6);
        // 4 cubes * 8 verts = 32
        assert_eq!(solid.vertices.len(), 32);
    }

    #[test]
    fn test_apply_snap_value() {
        assert!((apply_snap_value(0.3, 0.5) - 0.5).abs() < 1e-6);
        assert!((apply_snap_value(0.7, 0.5) - 0.5).abs() < 1e-6);
        assert!((apply_snap_value(0.8, 0.5) - 1.0).abs() < 1e-6);
        assert!((apply_snap_value(-0.3, 0.5) - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_ray_to_circle_distance() {
        // Ray pointing at the rim of a unit circle in XZ plane
        let dist = compute_ray_to_circle_distance(
            vec3(1.0, 5.0, 0.0),
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            1.0,
        );
        assert!(dist.abs() < 1e-5);
    }

    #[test]
    fn test_handle_color_highlight() {
        assert_eq!(
            resolve_handle_color(
                TransformGizmoHandle::AxisX,
                TransformGizmoHandle::AxisX,
                RED
            ),
            YELLOW
        );
        assert_eq!(
            resolve_handle_color(
                TransformGizmoHandle::AxisX,
                TransformGizmoHandle::AxisY,
                RED
            ),
            RED
        );
    }

    #[test]
    fn test_config_defaults() {
        let state = TransformGizmoState::default();
        assert_eq!(state.mode, TransformGizmoMode::Translate);
        assert_eq!(state.coordinate_space, CoordinateSpace::World);
        assert!(!state.snap_enabled);
        assert!((state.translate_snap_value - 0.5).abs() < 1e-6);
        assert!((state.rotate_snap_degrees - 15.0).abs() < 1e-6);
        assert!((state.scale_snap_value - 0.1).abs() < 1e-6);
    }
}
