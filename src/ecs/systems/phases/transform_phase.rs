use anyhow::Result;
use cgmath::Vector2;

use crate::app::FrameContext;
use crate::ecs::context::EcsContext;
use crate::ecs::systems::camera_shot_step;
use crate::ecs::systems::camera_systems::{compute_camera_position, compute_camera_up};
use crate::ecs::systems::sync_active_camera_to_view;
use crate::ecs::{
    calculate_projection, gizmo_sync_position, gizmo_update_selection_color,
    gizmo_update_vertex_buffer, update_billboard_transform,
};
use crate::math::calculate_billboard_click_rect;

pub fn run_transform_phase_ecs(ctx: &mut EcsContext) {
    camera_shot_step(
        &mut ctx.camera_mut(),
        &mut ctx
            .world
            .resource_mut::<crate::ecs::systems::CameraShotMotion>(),
        ctx.delta_time,
    );

    sync_active_camera_to_view(&mut ctx.world);

    update_camera_near_plane(ctx);

    let proj_data = calculate_projection(&*ctx.camera(), ctx.swapchain_extent);

    let light_position = ctx.light_state().light_position;
    gizmo_sync_position(&mut ctx.light_gizmo_mut().position, light_position);

    {
        let mut light_gizmo = ctx.light_gizmo_mut();
        let selectable = light_gizmo.selectable.clone();
        gizmo_update_selection_color(&mut light_gizmo.mesh, &selectable);
    }

    let camera_pos = compute_camera_position(&ctx.camera());
    let camera_up = compute_camera_up(&ctx.camera());
    update_billboard_transform(
        &mut ctx.billboard_mut(),
        light_position,
        camera_pos,
        camera_up,
    );

    let screen_size = Vector2::new(ctx.swapchain_extent.0 as f32, ctx.swapchain_extent.1 as f32);
    let billboard_rect = calculate_billboard_click_rect(
        light_position,
        screen_size,
        proj_data.view,
        proj_data.proj,
        0.5,
        0.15,
    );
    if let Some(mut debug_view) = ctx
        .world
        .get_resource_mut::<crate::ecs::resource::DebugViewState>()
    {
        debug_view.billboard_click_rect = billboard_rect;
    }

    ctx.world.insert_resource(proj_data);
}

pub unsafe fn run_transform_phase_gpu(ctx: &mut FrameContext) -> Result<()> {
    let mesh = ctx.light_gizmo().mesh.clone();
    let backend = ctx.create_backend();
    gizmo_update_vertex_buffer(&mesh, &backend)?;
    Ok(())
}

fn update_camera_near_plane(ctx: &mut EcsContext) {
    let camera_distance = ctx.camera().distance;
    let mut camera = ctx.camera_mut();
    camera.near_plane = (camera_distance * 0.001).max(0.1).min(10.0);
}
