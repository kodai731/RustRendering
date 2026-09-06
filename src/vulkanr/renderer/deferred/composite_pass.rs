use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::ray_query_pass::collect_selected_mesh_ids;
use crate::app::App;

fn debug_view_mode_value(mode: crate::ecs::resource::DebugViewMode) -> i32 {
    use crate::ecs::resource::DebugViewMode;
    match mode {
        DebugViewMode::Final => 0,
        DebugViewMode::Position => 1,
        DebugViewMode::Normal => 2,
        DebugViewMode::ShadowMask => 3,
        DebugViewMode::NdotL => 4,
        DebugViewMode::LightDirection => 5,
        DebugViewMode::ViewDepth => 6,
        DebugViewMode::ObjectID => 7,
        DebugViewMode::SelectionView => 8,
        DebugViewMode::SelectionUBO => 9,
    }
}

unsafe fn prepare_composite_resources<'a>(
    app: &'a App,
) -> Result<(
    &'a crate::vulkanr::pipeline::RRPipeline,
    &'a crate::vulkanr::descriptor::RRCompositeDescriptorSet,
    i32,
)> {
    let pipeline = app
        .data
        .raytracing
        .composite_pipeline
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Composite pipeline not initialized"))?;
    let descriptor = app
        .data
        .raytracing
        .composite_descriptor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Composite descriptor set not initialized"))?;
    let mode = app
        .resource::<crate::ecs::resource::DebugViewState>()
        .debug_view_mode;
    Ok((pipeline, descriptor, debug_view_mode_value(mode)))
}

pub unsafe fn record_composite_pass(
    app: &mut App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
    draw_data: &imgui::DrawData,
) -> Result<()> {
    let selected_mesh_ids = collect_selected_mesh_ids(app);

    if let Some(ref composite_descriptor) = app.data.raytracing.composite_descriptor {
        composite_descriptor.update_selection(&app.rrdevice, &selected_mesh_ids)?;
    }

    let render_targets = app.resource::<crate::vulkanr::context::RenderTargets>();
    let render_pass = render_targets.render.render_pass;
    let framebuffer = render_targets.render.framebuffers[image_index];
    let extent = app
        .resource::<crate::vulkanr::context::SwapchainState>()
        .swapchain
        .swapchain_extent;

    let (pipeline, descriptor, view_mode_value) = prepare_composite_resources(app)?;

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

    thyllore_vulkan_core::renderer::begin_composite_render_pass(
        &ctx,
        render_pass,
        framebuffer,
        extent,
        2,
        command_buffer,
    );
    thyllore_vulkan_core::renderer::record_composite_draw(
        &ctx,
        pipeline,
        descriptor,
        extent,
        view_mode_value,
        command_buffer,
    )?;
    super::OverlayRenderer::new(app).draw_all_overlays(command_buffer, image_index, true)?;

    app.record_imgui_rendering(command_buffer, draw_data)?;
    app.rrdevice.device.cmd_end_render_pass(command_buffer);

    Ok(())
}

pub unsafe fn record_composite_to_offscreen(
    app: &mut App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let selected_mesh_ids = collect_selected_mesh_ids(app);

    if let Some(ref composite_descriptor) = app.data.raytracing.composite_descriptor {
        composite_descriptor.update_selection(&app.rrdevice, &selected_mesh_ids)?;
    }

    let offscreen = app
        .data
        .viewport
        .offscreen
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Offscreen framebuffer not initialized"))?;

    let render_pass = offscreen.render_pass;
    let framebuffer = offscreen.framebuffer;
    let extent = offscreen.extent();

    let (pipeline, descriptor, view_mode_value) = prepare_composite_resources(app)?;

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

    thyllore_vulkan_core::renderer::begin_composite_render_pass(
        &ctx,
        render_pass,
        framebuffer,
        extent,
        3,
        command_buffer,
    );
    thyllore_vulkan_core::renderer::record_composite_draw(
        &ctx,
        pipeline,
        descriptor,
        extent,
        view_mode_value,
        command_buffer,
    )?;
    super::OverlayRenderer::new(app).draw_all_overlays(command_buffer, image_index, false)?;
    thyllore_vulkan_core::renderer::end_composite_render_pass(&ctx, command_buffer);

    Ok(())
}

pub unsafe fn record_composite_to_hdr(app: &App, command_buffer: vk::CommandBuffer) -> Result<()> {
    let selected_mesh_ids = collect_selected_mesh_ids(app);

    if let Some(ref composite_descriptor) = app.data.raytracing.composite_descriptor {
        composite_descriptor.update_selection(&app.rrdevice, &selected_mesh_ids)?;
    }

    let hdr_buffer = app
        .data
        .viewport
        .hdr_buffer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("HDR buffer not initialized"))?;

    let render_pass = hdr_buffer.render_pass;
    let framebuffer = hdr_buffer.framebuffer;
    let extent = hdr_buffer.extent();
    let (pipeline, descriptor, view_mode_value) = prepare_composite_resources(app)?;
    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, 0);
    let black_background = app
        .resource::<crate::ecs::resource::DebugViewState>()
        .black_background;
    let background_radiance = if black_background {
        0.0
    } else {
        thyllore_vulkan_core::renderer::BACKGROUND_RADIANCE
    };

    thyllore_vulkan_core::renderer::begin_hdr_render_pass(
        &ctx,
        render_pass,
        framebuffer,
        extent,
        background_radiance,
        command_buffer,
    );

    thyllore_vulkan_core::renderer::record_composite_draw(
        &ctx,
        pipeline,
        descriptor,
        extent,
        view_mode_value,
        command_buffer,
    )?;

    if !black_background {
        let pipeline_override = app.data.viewport.hdr_grid_pipeline_id;
        super::OverlayRenderer::new(app).draw_grid_overlay(command_buffer, 0, pipeline_override)?;
    }

    thyllore_vulkan_core::renderer::end_composite_render_pass(&ctx, command_buffer);

    Ok(())
}
