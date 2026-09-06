use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;

pub unsafe fn record_tonemap_to_offscreen(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let offscreen = app
        .data
        .viewport
        .offscreen
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Offscreen framebuffer not initialized"))?;

    let render_pass = offscreen.render_pass;
    let framebuffer = offscreen.framebuffer;
    let extent = offscreen.extent();

    let pipeline = app
        .data
        .raytracing
        .tonemap_pipeline
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("ToneMap pipeline not initialized"))?;
    let descriptor = app
        .data
        .raytracing
        .tonemap_descriptor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("ToneMap descriptor not initialized"))?;

    let tonemap_default = crate::ecs::resource::ToneMapping::default();
    let exposure_default = crate::ecs::resource::Exposure::default();
    let lens_default = crate::ecs::resource::LensEffects::default();
    let bloom_default = crate::ecs::resource::BloomSettings::default();

    let tonemap = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::ToneMapping>();
    let exposure = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::Exposure>();
    let lens = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::LensEffects>();
    let bloom = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::BloomSettings>();

    let tonemap_ref = tonemap.as_deref().unwrap_or(&tonemap_default);
    let exposure_ref = exposure.as_deref().unwrap_or(&exposure_default);
    let lens_ref = lens.as_deref().unwrap_or(&lens_default);
    let bloom_ref = bloom.as_deref().unwrap_or(&bloom_default);

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

    // Query for first entity with both FlameEffect and HeatPlume to build plume push constants
    let plume_data: Option<([f32; 4], [f32; 4], [f32; 4], [f32; 4])> = {
        let flame_entities: Vec<_> = app.data.ecs_world.query_flames();
        flame_entities.into_iter().find_map(|e| {
            let effect = app
                .data
                .ecs_world
                .get_component::<crate::ecs::component::FlameEffect>(e)?;
            let plume = app
                .data
                .ecs_world
                .get_component::<crate::ecs::component::HeatPlume>(e)?;
            Some((
                [effect.position.x, effect.position.y, effect.position.z, 1.0],
                [
                    plume.plume_temperature,
                    plume.width_base,
                    plume.width_slope,
                    plume.distortion_gain,
                ],
                [plume.plume_height, effect.time, plume.turbulence_amp, 0.0],
                [
                    effect.wind.direction.x,
                    effect.warp.rise_speed,
                    effect.wind.direction.y,
                    0.0,
                ],
            ))
        })
    };
    thyllore_vulkan_core::renderer::begin_tonemap_render_pass(
        &ctx,
        render_pass,
        framebuffer,
        extent,
        command_buffer,
    );
    let frame_slot = app
        .resource::<crate::vulkanr::context::FrameSync>()
        .current_frame;
    thyllore_vulkan_core::renderer::record_tonemap_draw(
        &ctx,
        pipeline,
        descriptor,
        frame_slot,
        tonemap_ref,
        exposure_ref,
        lens_ref,
        bloom_ref,
        extent,
        command_buffer,
        plume_data,
    )?;
    // Grid is already drawn inside record_composite_to_hdr, before the flame composite.
    // Drawing it again here would put it on top of the flame.
    super::OverlayRenderer::new(app).draw_all_overlays(command_buffer, image_index, false)?;
    thyllore_vulkan_core::renderer::end_tonemap_render_pass(&ctx, command_buffer);

    Ok(())
}
