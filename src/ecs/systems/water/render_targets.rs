use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::{App, AppData};
use crate::ecs::resource::{WaterBindingKey, WaterRenderTargets};
use crate::hooks::effect::EffectHook;
use crate::vulkanr::context::RenderTargets;
use crate::vulkanr::core::RRDevice;
use crate::vulkanr::render::RRRender;
use crate::vulkanr::resource::WaterBuffer;

pub const WATER_EFFECT_HOOK: EffectHook = EffectHook {
    name: "water",
    setup: Some(setup_water),
    prepare_frame: Some(prepare_water_frame_targets),
    on_viewport_resize: Some(resize_water_render_targets),
    destroy: Some(destroy_water_render_targets),
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
        targets.buffer.destroy(&app.rrdevice.device);
        targets.clear_handles();
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

unsafe fn prepare_water_frame_targets(app: &mut App, frame_slot: usize) -> Result<()> {
    let has_water = !app.data.ecs_world.query_waters().is_empty();
    let scene = water_scene_bindings(app);
    let Some(mut targets) = app.data.ecs_world.get_resource_mut::<WaterRenderTargets>() else {
        return Ok(());
    };
    let (Some((tlas, hit_table)), true) = (scene, has_water) else {
        targets.clear_handles();
        return Ok(());
    };

    let scene_color_desc = targets.buffer.scene_color_desc();
    let trace_desc = targets.buffer.trace_desc();
    let history_views = targets.buffer.history_image_views;
    let history_sampler = targets.buffer.history_sampler;

    let transient = &mut app.data.viewport.transient;
    let scene_color = transient.acquire(&app.instance, &app.rrdevice, scene_color_desc)?;
    let trace = transient.acquire(&app.instance, &app.rrdevice, trace_desc)?;
    let scene_color_image = transient.get(scene_color)?;
    let trace_image = transient.get(trace)?;
    targets.scene_color = Some(scene_color);
    targets.trace = Some(trace);

    let key = WaterBindingKey {
        tlas,
        hit_table,
        history_views,
        scene_color_generation: scene_color_image.generation,
        trace_generation: trace_image.generation,
    };
    if targets.is_bound(frame_slot, key) {
        return Ok(());
    }

    let Some(water_ubo) = app.data.raytracing.water_ubo.as_ref() else {
        return Ok(());
    };
    if let Some(descriptor) = app.data.raytracing.water_descriptor.as_ref() {
        descriptor.write_all_at(
            &app.rrdevice,
            frame_slot,
            water_ubo,
            scene_color_image.view,
            history_sampler,
            history_views,
            history_sampler,
            trace_image.view,
            history_sampler,
            tlas,
            hit_table,
        )?;
    }
    if let Some(trace_descriptor) = app.data.raytracing.water_trace_descriptor.as_ref() {
        trace_descriptor.write_all_at(
            &app.rrdevice,
            frame_slot,
            tlas,
            trace_image.view,
            water_ubo,
            hit_table,
        )?;
    }

    targets.mark_bound(frame_slot, key);
    Ok(())
}

fn water_scene_bindings(app: &App) -> Option<(vk::AccelerationStructureKHR, vk::Buffer)> {
    let accel = app.data.raytracing.acceleration_structure.as_ref()?;
    let tlas = accel.tlas.acceleration_structure?;
    let hit_table = accel.hit_shading_table.as_ref()?.buffer;
    Some((tlas, hit_table))
}
