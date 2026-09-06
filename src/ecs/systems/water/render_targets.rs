use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::{App, AppData};
use crate::ecs::resource::WaterRenderTargets;
use crate::hooks::effect::EffectHook;
use crate::vulkanr::context::RenderTargets;
use crate::vulkanr::core::RRDevice;
use crate::vulkanr::render::RRRender;
use crate::vulkanr::resource::WaterBuffer;

pub const WATER_EFFECT_HOOK: EffectHook = EffectHook {
    name: "water",
    setup: Some(setup_water),
    on_viewport_resize: Some(resize_water_render_targets),
    destroy: Some(destroy_water_render_targets),
    passes: super::passes::WATER_PASS_NODES,
};

unsafe fn create_water_render_targets(
    instance: &Instance,
    rrdevice: &RRDevice,
    data: &mut AppData,
    depth_view: vk::ImageView,
) -> Result<bool> {
    let Some(hdr_view) = data
        .viewport
        .hdr_buffer
        .as_ref()
        .map(|hdr| hdr.color_image_view)
    else {
        return Ok(false);
    };

    let buffer = WaterBuffer::new(
        instance,
        rrdevice,
        &mut data.viewport.storage,
        data.raytracing.command_pool,
        data.viewport.width,
        data.viewport.height,
        hdr_view,
        depth_view,
    )?;

    for image in buffer.history_images {
        data.pass_image_states.mark_shader_read_only(image);
    }
    data.pass_image_states.forget(buffer.caustic_accum_image);
    data.ecs_world
        .insert_resource(WaterRenderTargets::new(buffer));
    Ok(true)
}

unsafe fn setup_water(
    instance: &Instance,
    rrdevice: &RRDevice,
    data: &mut AppData,
    rrrender: &RRRender,
) -> Result<()> {
    if !create_water_render_targets(instance, rrdevice, data, rrrender.gbuffer_depth_image_view)? {
        log!("HDR buffer not available, skipping water pipeline");
        return Ok(());
    }

    let (Some(water_targets), Some(hdr_buffer)) = (
        data.ecs_world
            .get_resource::<crate::ecs::resource::WaterRenderTargets>(),
        data.viewport.hdr_buffer.as_ref(),
    ) else {
        log!("Water buffer not available, skipping water pipeline");
        return Ok(());
    };

    data.raytracing.create_water_pipeline(
        instance,
        rrdevice,
        rrrender,
        &data.graphics_resources,
        &water_targets.buffer,
        hdr_buffer,
        crate::app::init::MAX_FRAMES_IN_FLIGHT,
    )?;

    log!("Water pipeline created successfully");
    Ok(())
}

unsafe fn resize_water_render_targets(app: &mut App) -> Result<()> {
    let depth_view = app
        .resource::<RenderTargets>()
        .render
        .gbuffer_depth_image_view;
    if depth_view == vk::ImageView::null() {
        return Ok(());
    }
    if app
        .data
        .ecs_world
        .get_resource::<WaterRenderTargets>()
        .is_none()
    {
        return Ok(());
    }

    destroy_water_render_targets(app)?;
    if !create_water_render_targets(&app.instance, &app.rrdevice, &mut app.data, depth_view)? {
        return Ok(());
    }

    update_water_caustic_descriptor(app)
}

unsafe fn destroy_water_render_targets(app: &mut App) -> Result<()> {
    if let Some(mut targets) = app.data.ecs_world.get_resource_mut::<WaterRenderTargets>() {
        for image in targets.buffer.history_images {
            app.data.pass_image_states.forget(image);
        }
        app.data
            .pass_image_states
            .forget(targets.buffer.caustic_accum_image);
        targets.buffer.destroy(&app.rrdevice.device);
        targets.forget_bindings();
    }
    Ok(())
}

unsafe fn update_water_caustic_descriptor(app: &mut App) -> Result<()> {
    let Some(caustic_accum_view) = app
        .data
        .ecs_world
        .get_resource::<WaterRenderTargets>()
        .map(|targets| targets.buffer.caustic_accum_view)
    else {
        return Ok(());
    };
    let Some(hdr_color_view) = app
        .data
        .viewport
        .hdr_buffer
        .as_ref()
        .map(|hdr| hdr.color_image_view)
    else {
        return Ok(());
    };

    let rrdevice = &app.rrdevice;
    let raytracing = &mut app.data.raytracing;
    let tlas = raytracing
        .acceleration_structure
        .as_ref()
        .and_then(|accel| accel.tlas.acceleration_structure);
    let (Some(position_image_view), Some(scene_buffer), Some(water_ubo)) = (
        raytracing
            .gbuffer
            .as_ref()
            .map(|gbuffer| gbuffer.position_image_view),
        raytracing.scene_uniform_buffer,
        raytracing.water_ubo.as_ref().map(|ubo| ubo.handle()),
    ) else {
        return Ok(());
    };
    let Some(descriptor) = raytracing.water_caustic_descriptor.as_mut() else {
        return Ok(());
    };

    descriptor.allocate_and_update(
        rrdevice,
        caustic_accum_view,
        position_image_view,
        tlas,
        scene_buffer,
        water_ubo,
        hdr_color_view,
    )
}
