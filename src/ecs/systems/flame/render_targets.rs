use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::{App, AppData};
use crate::ecs::resource::{FlameHistorySnapshotState, FlameRenderTargets};
use crate::hooks::effect::EffectHook;
use crate::vulkanr::context::RenderTargets;
use crate::vulkanr::core::RRDevice;
use crate::vulkanr::descriptor::FlameImageBindings;
use crate::vulkanr::render::RRRender;
use crate::vulkanr::resource::FlameBuffer;

pub const FLAME_EFFECT_HOOK: EffectHook = EffectHook {
    name: "flame",
    setup: Some(setup_flame),
    prepare_frame: None,
    on_viewport_resize: Some(resize_flame_render_targets),
    destroy: Some(destroy_flame_render_targets),
};

unsafe fn create_flame_render_targets(
    instance: &Instance,
    rrdevice: &RRDevice,
    data: &mut AppData,
) -> Result<()> {
    let Some(hdr_view) = data
        .viewport
        .hdr_buffer
        .as_ref()
        .map(|hdr| hdr.color_image_view)
    else {
        return Ok(());
    };

    let buffer = FlameBuffer::new(
        instance,
        rrdevice,
        &mut data.viewport.storage,
        data.raytracing.command_pool,
        data.viewport.width,
        data.viewport.height,
        hdr_view,
    )?;

    data.ecs_world
        .insert_resource(FlameRenderTargets { buffer });
    Ok(())
}

unsafe fn setup_flame(
    instance: &Instance,
    rrdevice: &RRDevice,
    data: &mut AppData,
    rrrender: &RRRender,
) -> Result<()> {
    create_flame_render_targets(instance, rrdevice, data)?;
    let Some(flame_targets) = data
        .ecs_world
        .get_resource::<crate::ecs::resource::FlameRenderTargets>()
    else {
        log!("Flame buffer not available, skipping flame pipeline");
        return Ok(());
    };
    let flame_buffer = &flame_targets.buffer;

    let position_image_view = match data.raytracing.gbuffer {
        Some(ref gbuffer) => gbuffer.position_image_view,
        None => {
            log!("GBuffer not available, skipping flame pipeline");
            return Ok(());
        }
    };

    let position_sampler = match data.raytracing.gbuffer_sampler {
        Some(sampler) => sampler,
        None => {
            log!("GBuffer sampler not available, skipping flame pipeline");
            return Ok(());
        }
    };

    data.raytracing.create_flame_pipeline(
        instance,
        rrdevice,
        rrrender,
        &data.graphics_resources,
        flame_buffer,
        position_image_view,
        position_sampler,
        rrrender.gbuffer_depth_image_view,
    )?;

    log!("Flame pipeline created successfully");
    Ok(())
}

unsafe fn resize_flame_render_targets(app: &mut App) -> Result<()> {
    let Some(hdr_view) = app
        .data
        .viewport
        .hdr_buffer
        .as_ref()
        .map(|hdr| hdr.color_image_view)
    else {
        return Ok(());
    };
    let (width, height) = (app.data.viewport.width, app.data.viewport.height);
    let command_pool = app.data.raytracing.command_pool;
    let scene_depth_view = app
        .resource::<RenderTargets>()
        .render
        .gbuffer_depth_image_view;

    let Some(mut targets) = app.data.ecs_world.get_resource_mut::<FlameRenderTargets>() else {
        return Ok(());
    };
    targets.buffer.resize(
        &app.instance,
        &app.rrdevice,
        &mut app.data.viewport.storage,
        command_pool,
        width,
        height,
        hdr_view,
    )?;

    if let Some(descriptor) = app.data.raytracing.flame_descriptor.as_ref() {
        descriptor.update_image_views(
            &app.rrdevice,
            FlameImageBindings {
                history_image_views: targets.buffer.history_image_views,
                flame_sampler: targets.buffer.sampler,
                sdf_image_view: app.data.raytracing.flame_sdf_image_view,
                sdf_sampler: app.data.raytracing.flame_sdf_sampler,
                scene_depth_view,
            },
        )?;
    }
    drop(targets);

    if let Some(mut state) = app
        .data
        .ecs_world
        .get_resource_mut::<FlameHistorySnapshotState>()
    {
        state.previous = None;
    }
    Ok(())
}

unsafe fn destroy_flame_render_targets(app: &mut App) -> Result<()> {
    if let Some(mut targets) = app.data.ecs_world.get_resource_mut::<FlameRenderTargets>() {
        targets.buffer.destroy(&app.rrdevice.device);
    }
    Ok(())
}
