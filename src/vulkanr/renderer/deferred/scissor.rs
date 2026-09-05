use cgmath::{Matrix4, Vector3};
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::ecs::resource::ProjectionData;

const SCISSOR_MARGIN_PX: f32 = 2.0;

#[derive(Clone, Copy)]
struct ScreenBounds {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

pub(crate) fn full_extent_scissor(extent: vk::Extent2D) -> vk::Rect2D {
    vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(extent)
        .build()
}

/// Projects local-space bound corners to a screen scissor. Returns the full
/// extent when the projection is unavailable or a corner is behind the camera,
/// and `None` when the projected bounds are empty.
pub(crate) fn compute_bounds_scissor(
    app: &App,
    extent: vk::Extent2D,
    model: &Matrix4<f32>,
    corners: impl IntoIterator<Item = Vector3<f32>>,
) -> Option<vk::Rect2D> {
    let Some(projection) = app.data.ecs_world.get_resource::<ProjectionData>() else {
        return Some(full_extent_scissor(extent));
    };
    let model_view_proj = projection.proj * projection.view * model;

    let mut screen_bounds: Option<ScreenBounds> = None;
    for corner in corners {
        let clip = model_view_proj * cgmath::vec4(corner.x, corner.y, corner.z, 1.0);
        if clip.w <= 0.0 {
            return Some(full_extent_scissor(extent));
        }
        let screen_x = (clip.x / clip.w + 1.0) * 0.5 * extent.width as f32;
        let screen_y = (clip.y / clip.w + 1.0) * 0.5 * extent.height as f32;
        screen_bounds = Some(match screen_bounds {
            None => ScreenBounds {
                min_x: screen_x,
                min_y: screen_y,
                max_x: screen_x,
                max_y: screen_y,
            },
            Some(bounds) => ScreenBounds {
                min_x: bounds.min_x.min(screen_x),
                min_y: bounds.min_y.min(screen_y),
                max_x: bounds.max_x.max(screen_x),
                max_y: bounds.max_y.max(screen_y),
            },
        });
    }
    let ScreenBounds {
        min_x,
        min_y,
        max_x,
        max_y,
    } = screen_bounds?;

    let min_x = (min_x - SCISSOR_MARGIN_PX).clamp(0.0, extent.width as f32);
    let min_y = (min_y - SCISSOR_MARGIN_PX).clamp(0.0, extent.height as f32);
    let max_x = (max_x + SCISSOR_MARGIN_PX).clamp(0.0, extent.width as f32);
    let max_y = (max_y + SCISSOR_MARGIN_PX).clamp(0.0, extent.height as f32);
    if max_x - min_x < 1.0 || max_y - min_y < 1.0 {
        return None;
    }

    Some(
        vk::Rect2D::builder()
            .offset(vk::Offset2D {
                x: min_x as i32,
                y: min_y as i32,
            })
            .extent(vk::Extent2D {
                width: (max_x - min_x).ceil() as u32,
                height: (max_y - min_y).ceil() as u32,
            })
            .build(),
    )
}
