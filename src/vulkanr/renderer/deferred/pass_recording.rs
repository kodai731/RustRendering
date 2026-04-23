use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::ecs::resource::HierarchyState;
use crate::ecs::world::MeshRef;

use super::{CompositePass, GBufferPass, OnionSkinRenderPass, RayQueryPass};

pub unsafe fn record_gbuffer_pass(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let pass = GBufferPass::new(app)?;
    let render_targets = app.resource::<crate::vulkanr::context::RenderTargets>();
    pass.record(
        command_buffer,
        render_targets.render.gbuffer_render_pass,
        render_targets.render.gbuffer_framebuffer,
        image_index,
    )
}

pub unsafe fn record_ray_query_pass(app: &App, command_buffer: vk::CommandBuffer) -> Result<()> {
    let pass = RayQueryPass::new(app)?;
    let normal_offset = app
        .resource::<crate::ecs::resource::LightState>()
        .shadow_normal_offset;
    pass.record(command_buffer, normal_offset)
}

fn collect_selected_mesh_ids(app: &App) -> Vec<u32> {
    let hierarchy_state = app.data.ecs_world.resource::<HierarchyState>();
    let mut selected_ids = Vec::new();

    for &entity in hierarchy_state.multi_selection.iter() {
        if let Some(mesh_ref) = app.data.ecs_world.get_component::<MeshRef>(entity) {
            if let Some(mesh_asset) = app.data.ecs_assets.get_mesh(mesh_ref.mesh_asset_id) {
                let mesh_id = (mesh_asset.graphics_mesh_index + 1) as u32;
                if !selected_ids.contains(&mesh_id) {
                    selected_ids.push(mesh_id);
                }
            }
        }
    }

    selected_ids
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

    {
        let pass = CompositePass::new(app)?;
        pass.record(command_buffer, render_pass, framebuffer, image_index)?;
    }

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

    let pass = CompositePass::new_for_offscreen(app, extent)?;
    pass.record_to_offscreen(command_buffer, render_pass, framebuffer, image_index)?;

    Ok(())
}

pub unsafe fn record_composite_to_hdr(
    app: &mut App,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
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

    let pass = CompositePass::new_for_offscreen(app, extent)?;
    pass.record_to_hdr(command_buffer, render_pass, framebuffer)?;

    Ok(())
}

pub unsafe fn record_bloom(app: &App, command_buffer: vk::CommandBuffer) -> Result<()> {
    let bloom_settings = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::BloomSettings>();
    let Some(bloom_settings) = bloom_settings else {
        return Ok(());
    };
    if !bloom_settings.enabled {
        return Ok(());
    }

    let (Some(bloom_chain), Some(downsample_pipeline), Some(upsample_pipeline)) = (
        app.data.viewport.bloom_chain.as_ref(),
        app.data.raytracing.bloom_downsample_pipeline.as_ref(),
        app.data.raytracing.bloom_upsample_pipeline.as_ref(),
    ) else {
        return Ok(());
    };

    let bloom_descriptors = app
        .data
        .raytracing
        .bloom_descriptors
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Bloom descriptors not initialized"))?;

    let ctx = thyllore_vulkan_core::FrameRenderContext {
        device: &app.rrdevice,
        graphics: &app.data.graphics_resources,
        buffers: &app.data.buffer_registry,
        pipelines: &app.data.pipeline_storage,
        image_index: 0,
    };

    thyllore_vulkan_core::renderer::record_bloom_pass(
        &ctx,
        downsample_pipeline,
        upsample_pipeline,
        bloom_descriptors,
        bloom_chain,
        &bloom_settings,
        command_buffer,
    )?;

    Ok(())
}

pub unsafe fn record_dof(app: &App, command_buffer: vk::CommandBuffer) -> Result<()> {
    let (Some(pipeline), Some(dof_descriptor), Some(dof_buffer)) = (
        app.data.raytracing.dof_pipeline.as_ref(),
        app.data.raytracing.dof_descriptor.as_ref(),
        app.data.viewport.dof_buffer.as_ref(),
    ) else {
        return Ok(());
    };

    let dof_settings = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::DepthOfField>();
    let camera_params = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::PhysicalCameraParameters>();
    let camera = app.resource::<crate::ecs::resource::Camera>();

    let dof_default = crate::ecs::resource::DepthOfField::default();
    let camera_default = crate::ecs::resource::PhysicalCameraParameters::default();

    let dof_ref: &crate::ecs::resource::DepthOfField =
        dof_settings.as_deref().unwrap_or(&dof_default);
    let camera_ref: &crate::ecs::resource::PhysicalCameraParameters =
        camera_params.as_deref().unwrap_or(&camera_default);

    let ctx = thyllore_vulkan_core::FrameRenderContext {
        device: &app.rrdevice,
        graphics: &app.data.graphics_resources,
        buffers: &app.data.buffer_registry,
        pipelines: &app.data.pipeline_storage,
        image_index: 0,
    };

    thyllore_vulkan_core::renderer::record_dof_pass(
        &ctx,
        pipeline,
        dof_descriptor,
        dof_buffer,
        dof_ref,
        camera_ref,
        camera.near_plane,
        command_buffer,
    )?;

    Ok(())
}

pub unsafe fn record_auto_exposure(app: &App, command_buffer: vk::CommandBuffer) -> Result<()> {
    let ae_settings = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::AutoExposure>();
    let Some(ae_settings) = ae_settings else {
        return Ok(());
    };
    if !ae_settings.enabled {
        return Ok(());
    }

    let (
        Some(histogram_pipeline),
        Some(average_pipeline),
        Some(histogram_descriptor),
        Some(average_descriptor),
        Some(buffers),
    ) = (
        app.data
            .raytracing
            .auto_exposure_histogram_pipeline
            .as_ref(),
        app.data.raytracing.auto_exposure_average_pipeline.as_ref(),
        app.data
            .raytracing
            .auto_exposure_histogram_descriptor
            .as_ref(),
        app.data
            .raytracing
            .auto_exposure_average_descriptor
            .as_ref(),
        app.data.viewport.auto_exposure_buffers.as_ref(),
    )
    else {
        return Ok(());
    };

    let delta_time = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::TimelineState>()
        .map(|t| 1.0 / 60.0 * t.speed.max(0.01))
        .unwrap_or(1.0 / 60.0);

    let ctx = thyllore_vulkan_core::FrameRenderContext {
        device: &app.rrdevice,
        graphics: &app.data.graphics_resources,
        buffers: &app.data.buffer_registry,
        pipelines: &app.data.pipeline_storage,
        image_index: 0,
    };

    thyllore_vulkan_core::renderer::record_auto_exposure_pass(
        &ctx,
        histogram_pipeline,
        average_pipeline,
        histogram_descriptor,
        average_descriptor,
        buffers,
        &ae_settings,
        delta_time,
        command_buffer,
    )?;

    Ok(())
}

pub unsafe fn record_onion_skin_pass(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    if let Some(pass) = OnionSkinRenderPass::new(app)? {
        pass.record_ghost_pass(command_buffer, image_index)?;
    }
    Ok(())
}

pub unsafe fn record_onion_skin_composite(
    app: &App,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    if let Some(pass) = OnionSkinRenderPass::new(app)? {
        pass.record_composite_pass(command_buffer);
    }
    Ok(())
}

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

    let ctx = thyllore_vulkan_core::FrameRenderContext {
        device: &app.rrdevice,
        graphics: &app.data.graphics_resources,
        buffers: &app.data.buffer_registry,
        pipelines: &app.data.pipeline_storage,
        image_index,
    };

    thyllore_vulkan_core::renderer::begin_tonemap_render_pass(
        &ctx,
        render_pass,
        framebuffer,
        extent,
        command_buffer,
    );
    thyllore_vulkan_core::renderer::record_tonemap_draw(
        &ctx,
        pipeline,
        descriptor,
        tonemap_ref,
        exposure_ref,
        lens_ref,
        bloom_ref,
        extent,
        command_buffer,
    )?;
    super::OverlayRenderer::new(app).draw_all_overlays(command_buffer, image_index)?;
    thyllore_vulkan_core::renderer::end_tonemap_render_pass(&ctx, command_buffer);

    Ok(())
}
