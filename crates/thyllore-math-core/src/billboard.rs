use crate::matrix::Mat4;
use cgmath::{Vector2, Vector3};

pub fn calculate_billboard_click_rect(
    world_pos: Vector3<f32>,
    screen_size: Vector2<f32>,
    view_matrix: Mat4,
    proj_matrix: Mat4,
    billboard_world_size: f32,
    billboard_ndc_scale: f32,
) -> Option<[f32; 4]> {
    let clip_pos =
        proj_matrix * view_matrix * cgmath::vec4(world_pos.x, world_pos.y, world_pos.z, 1.0);

    if clip_pos.w <= 0.0 {
        return None;
    }

    let ndc_x = clip_pos.x / clip_pos.w;
    let ndc_y = clip_pos.y / clip_pos.w;

    let screen_x = (ndc_x + 1.0) * 0.5 * screen_size.x;
    let screen_y = (ndc_y + 1.0) * 0.5 * screen_size.y;

    let billboard_ndc_half_size = billboard_world_size * billboard_ndc_scale;
    let billboard_screen_half_width = billboard_ndc_half_size * screen_size.x * 0.5;
    let billboard_screen_half_height = billboard_ndc_half_size * screen_size.y * 0.5;

    Some([
        screen_x - billboard_screen_half_width,
        screen_y - billboard_screen_half_height,
        screen_x + billboard_screen_half_width,
        screen_y + billboard_screen_half_height,
    ])
}

pub fn is_point_in_rect(point: Vector2<f32>, rect: [f32; 4]) -> bool {
    point.x >= rect[0] && point.x <= rect[2] && point.y >= rect[1] && point.y <= rect[3]
}
