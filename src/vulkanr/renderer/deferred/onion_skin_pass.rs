use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;

pub unsafe fn record_onion_skin_pass(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let Some(resources) = app.data.raytracing.onion_skin_pass.as_ref() else {
        return Ok(());
    };
    let Some(onion_skin_gpu) = app.data.onion_skin_gpu.as_ref() else {
        return Ok(());
    };
    if onion_skin_gpu.source_mesh_index.is_none() {
        return Ok(());
    }
    if onion_skin_gpu.active_ghost_count() == 0 {
        return Ok(());
    }

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

    thyllore_vulkan_core::renderer::record_onion_skin_ghost_pass(
        &ctx,
        resources,
        onion_skin_gpu,
        image_index,
        command_buffer,
    )?;
    Ok(())
}

pub unsafe fn record_onion_skin_composite(
    app: &App,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    let Some(resources) = app.data.raytracing.onion_skin_pass.as_ref() else {
        return Ok(());
    };
    let Some(onion_skin_gpu) = app.data.onion_skin_gpu.as_ref() else {
        return Ok(());
    };
    if onion_skin_gpu.source_mesh_index.is_none() {
        return Ok(());
    }
    if onion_skin_gpu.active_ghost_count() == 0 {
        return Ok(());
    }

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, 0);

    thyllore_vulkan_core::renderer::record_onion_skin_composite_pass(
        &ctx,
        resources,
        command_buffer,
    );
    Ok(())
}
