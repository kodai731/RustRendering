use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::{App, AppData};
use crate::ecs::resource::WindRenderTargets;
use crate::hooks::effect::EffectHook;
use crate::vulkanr::context::RenderTargets;
use crate::vulkanr::core::RRDevice;
use crate::vulkanr::render::RRRender;
use crate::vulkanr::resource::WindBuffer;

pub const WIND_EFFECT_HOOK: EffectHook = EffectHook {
    name: "wind",
    setup: Some(setup_wind),
    on_viewport_resize: Some(resize_wind_render_targets),
    destroy: Some(destroy_wind_render_targets),
    passes: &[&super::passes::WindPassNode],
};

unsafe fn create_wind_render_targets(rrdevice: &RRDevice, data: &mut AppData) -> Result<bool> {
    let Some(hdr_view) = data
        .viewport
        .hdr_buffer
        .as_ref()
        .map(|hdr| hdr.color_image_view)
    else {
        return Ok(false);
    };

    let buffer = WindBuffer::new(
        rrdevice,
        data.viewport.width,
        data.viewport.height,
        hdr_view,
    )?;
    data.ecs_world.insert_resource(WindRenderTargets { buffer });
    Ok(true)
}

unsafe fn setup_wind(
    instance: &Instance,
    rrdevice: &RRDevice,
    data: &mut AppData,
    rrrender: &RRRender,
) -> Result<()> {
    if !create_wind_render_targets(rrdevice, data)? {
        log!("HDR buffer not available, skipping wind pipeline");
        return Ok(());
    }
    let Some(wind_targets) = data.ecs_world.get_resource::<WindRenderTargets>() else {
        return Ok(());
    };

    data.raytracing.create_wind_pipeline(
        instance,
        rrdevice,
        rrrender,
        &data.graphics_resources,
        &wind_targets.buffer,
        rrrender.gbuffer_depth_image_view,
    )?;

    log!("Wind pipeline created successfully");
    Ok(())
}

unsafe fn resize_wind_render_targets(app: &mut App) -> Result<()> {
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
    let scene_depth_view = app
        .resource::<RenderTargets>()
        .render
        .gbuffer_depth_image_view;

    if let Some(mut targets) = app.data.ecs_world.get_resource_mut::<WindRenderTargets>() {
        targets
            .buffer
            .resize(&app.rrdevice, width, height, hdr_view)?;
    }

    if let Some(descriptor) = app.data.raytracing.wind_descriptor.as_ref() {
        descriptor.update_scene_depth(&app.rrdevice, scene_depth_view)?;
    }
    Ok(())
}

unsafe fn destroy_wind_render_targets(app: &mut App) -> Result<()> {
    if let Some(mut targets) = app.data.ecs_world.get_resource_mut::<WindRenderTargets>() {
        targets.buffer.destroy(&app.rrdevice.device);
    }
    Ok(())
}
