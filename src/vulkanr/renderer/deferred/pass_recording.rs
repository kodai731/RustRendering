use anyhow::Result;
use cgmath::{SquareMatrix, Vector3};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrRayTracingPipelineExtension;

use crate::app::App;
use crate::ecs::resource::HierarchyState;
use crate::ecs::world::MeshRef;

pub unsafe fn record_gbuffer_pass(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let gbuffer = app
        .data
        .raytracing
        .gbuffer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("G-Buffer not initialized"))?;
    let pipeline = app
        .data
        .raytracing
        .gbuffer_pipeline
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("G-Buffer pipeline not initialized"))?;
    let render_targets = app.resource::<crate::vulkanr::context::RenderTargets>();
    let render_pass = render_targets.render.gbuffer_render_pass;
    let framebuffer = render_targets.render.gbuffer_framebuffer;
    drop(render_targets);

    let draw_mesh_indices = collect_gbuffer_mesh_indices(app);
    let heatmap_mode = resolve_heatmap_mode(app);

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

    thyllore_vulkan_core::renderer::record_gbuffer_pass(
        &ctx,
        gbuffer,
        pipeline,
        render_pass,
        framebuffer,
        &draw_mesh_indices,
        heatmap_mode,
        command_buffer,
    )
}

fn collect_gbuffer_mesh_indices(app: &App) -> Vec<usize> {
    let ecs_world = &app.data.ecs_world;
    let ecs_assets = &app.data.ecs_assets;

    if ecs_world.has_mesh_entities() {
        ecs_world
            .query_renderable()
            .iter()
            .filter_map(|&entity| {
                let mesh_ref = ecs_world.get_component::<MeshRef>(entity)?;
                let mesh_asset = ecs_assets.get_mesh(mesh_ref.mesh_asset_id)?;
                if !mesh_asset.render_to_gbuffer {
                    return None;
                }
                Some(mesh_asset.graphics_mesh_index)
            })
            .collect()
    } else {
        (0..app.data.graphics_resources.meshes.len()).collect()
    }
}

fn resolve_heatmap_mode(app: &App) -> u32 {
    app.data
        .ecs_world
        .get_resource::<crate::ecs::resource::WeightHeatmapState>()
        .map(|state| if state.enabled { 1 } else { 0 })
        .unwrap_or(0)
}

pub unsafe fn record_ray_query_pass(app: &App, command_buffer: vk::CommandBuffer) -> Result<()> {
    let gbuffer = app
        .data
        .raytracing
        .gbuffer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("G-Buffer not initialized"))?;
    let pipeline = app
        .data
        .raytracing
        .ray_query_pipeline
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Ray Query pipeline not initialized"))?;
    let descriptor = app
        .data
        .raytracing
        .ray_query_descriptor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Ray Query descriptor set not initialized"))?;

    let normal_offset = app
        .resource::<crate::ecs::resource::LightState>()
        .shadow_normal_offset;

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, 0);

    thyllore_vulkan_core::renderer::record_ray_query_pass(
        &ctx,
        gbuffer,
        pipeline,
        descriptor,
        normal_offset,
        command_buffer,
    )
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

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, 0);

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

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, 0);

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

pub unsafe fn record_auto_exposure(
    app: &App,
    command_buffer: vk::CommandBuffer,
    frame_slot: usize,
) -> Result<()> {
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

    let mut delta_time = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::TimelineState>()
        .map(|t| 1.0 / 60.0 * t.speed.max(0.01))
        .unwrap_or(1.0 / 60.0);

    // Override with fixed timestep (1/60) during batch runs to ensure determinism
    if app
        .data
        .ecs_world
        .contains_resource::<crate::ecs::resource::BatchRun>()
    {
        delta_time = 1.0 / 60.0;
    }
    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, 0);

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

    // BufferMemoryBarrier: COMPUTE_SHADER (SHADER_WRITE) → TRANSFER (TRANSFER_READ)
    let barrier = vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffers.luminance_buffer)
        .offset(0)
        .size(8u64)
        .build();

    app.rrdevice.device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[barrier],
        &[] as &[vk::ImageMemoryBarrier],
    );

    // Copy luminance_buffer → readback_buffers[frame_slot] (8 bytes)
    app.rrdevice.device.cmd_copy_buffer(
        command_buffer,
        buffers.luminance_buffer,
        buffers.readback_buffers[frame_slot],
        &[vk::BufferCopy::builder()
            .src_offset(0)
            .dst_offset(0)
            .size(thyllore_vulkan_core::resource::LUMINANCE_BUFFER_SIZE)
            .build()],
    );

    Ok(())
}

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

pub unsafe fn record_flame_passes(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let (Some(flame_buffer), Some(shading_pipeline), Some(descriptor)) = (
        app.data.viewport.flame_buffer.as_ref(),
        app.data.raytracing.flame_shading_pipeline.as_ref(),
        app.data.raytracing.flame_descriptor.as_ref(),
    ) else {
        return Ok(());
    };

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

    let mut flames = app.data.ecs_world.query_flames();

    // Sort flames by descending camera distance (back-to-front) for correct overdraw
    if let Some(projection) = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::ProjectionData>()
    {
        let view_inverse = projection
            .view
            .invert()
            .unwrap_or_else(|| cgmath::Matrix4::identity());
        let camera_pos =
            cgmath::Vector3::new(view_inverse[3][0], view_inverse[3][1], view_inverse[3][2]);
        flames.sort_by(|a, b| {
            let effect_a = app
                .data
                .ecs_world
                .get_component::<crate::ecs::component::FlameEffect>(*a);
            let effect_b = app
                .data
                .ecs_world
                .get_component::<crate::ecs::component::FlameEffect>(*b);
            match (effect_a, effect_b) {
                (Some(ea), Some(eb)) => {
                    let pos_a = Vector3::new(ea.position[0], ea.position[1], ea.position[2]);
                    let dist_a = ((pos_a.x - camera_pos.x).powi(2)
                        + (pos_a.y - camera_pos.y).powi(2)
                        + (pos_a.z - camera_pos.z).powi(2))
                    .sqrt();
                    let pos_b = Vector3::new(eb.position[0], eb.position[1], eb.position[2]);
                    let dist_b = ((pos_b.x - camera_pos.x).powi(2)
                        + (pos_b.y - camera_pos.y).powi(2)
                        + (pos_b.z - camera_pos.z).powi(2))
                    .sqrt();
                    dist_b
                        .partial_cmp(&dist_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
                _ => std::cmp::Ordering::Equal,
            }
        });
    }

    let instance_count = flames
        .len()
        .min(thyllore_vulkan_core::resource::MAX_FLAME_INSTANCES);

    if instance_count == 0 {
        return Ok(());
    }

    // Calculate history_index from the first flame's frame_index (shared by all instances)
    let history_index = if let Some(first) = flames.first() {
        if let Some(temporal) = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::FlameTemporalAccum>(*first)
        {
            (temporal.frame_index as usize) & 1
        } else {
            0
        }
    } else {
        0
    };

    // Process each instance sequentially: F1_i -> F2_i before moving to i+1
    for i in 0..instance_count {
        let flame = flames[i];
        let effect = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::FlameEffect>(flame)
            .ok_or_else(|| anyhow::anyhow!("Missing FlameEffect for instance {}", i))?;

        // Build UBO for this instance (trail-aware)
        let trail = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::FlameTrail>(flame);
        let is_noise_mode = app
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::FlameRenderSettings>()
            .map(|s| s.shading_mode == thyllore_effect_core::FlameShadingMode::NoiseRaymarch)
            .unwrap_or(false);
        let baked = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::FlameBaked>(flame)
            .cloned()
            .unwrap_or_default();
        let temporal_accum = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::FlameTemporalAccum>(flame)
            .cloned()
            .unwrap_or_default();
        let ubo = thyllore_effect_core::build_flame_ubo_with_trail(
            effect,
            &baked,
            &temporal_accum,
            trail.map(|t| &t.state),
            is_noise_mode,
        );

        let Some(flame_ubo) = app.data.raytracing.flame_ubo.as_ref() else {
            return Ok(());
        };
        let ubo_dynamic_offset = flame_ubo.slot_offset(i)? as u32;
        flame_ubo.record_update(
            &ctx.device.device,
            command_buffer,
            i,
            &ubo,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        )?;

        // Build per-instance model matrix
        let model_matrix = ubo.model;

        // Compute per-instance scissor using the model matrix
        let bend_offset = [
            ubo.wind_bend.wind_direction[0] * ubo.wind_bend.bend_amount,
            ubo.wind_bend.wind_direction[1] * ubo.wind_bend.bend_amount,
        ];
        let support_scale = thyllore_effect_core::flame_shell_support_scale(
            ubo.emitter_params.kind as u32,
            ubo.emitter_params.ring_major_ratio,
            ubo.support_motion.support_margin,
        );
        let Some(scissor) = compute_flame_scissor(
            app,
            flame_buffer.extent(),
            &model_matrix,
            bend_offset,
            support_scale,
            ubo.support_motion.support_margin,
            thyllore_effect_core::FlameProxyPad {
                radial: thyllore_effect_core::flame_proxy_radial_pad(
                    ubo.branch_field.bounding_pad,
                    ubo.support_motion.meander_amp,
                ),
                top: ubo.branch_field.bounding_pad_y,
            },
        ) else {
            continue;
        };

        // Get render settings for push constants
        let settings = app
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::FlameRenderSettings>()
            .map(|settings| *settings)
            .unwrap_or_default();
        let push_constants = thyllore_vulkan_core::renderer::FlamePushConstants::new(
            settings.shading_mode.as_shader_value(),
            settings.resolved_step_count() as i32,
            settings.debug_view.as_shader_value(),
        );

        // Record shading pass for this instance (F2_i completed before next instance)
        thyllore_vulkan_core::renderer::record_flame_shading_pass(
            &ctx,
            flame_buffer,
            shading_pipeline,
            descriptor,
            history_index,
            ubo_dynamic_offset,
            scissor,
            push_constants,
            image_index,
            command_buffer,
        )?;
    }

    Ok(())
}

pub unsafe fn record_water_passes(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let (Some(water_buffer), Some(shading_pipeline), Some(descriptor)) = (
        app.data.viewport.water_buffer.as_ref(),
        app.data.raytracing.water_shading_pipeline.as_ref(),
        app.data.raytracing.water_descriptor.as_ref(),
    ) else {
        return Ok(());
    };

    if !app.data.raytracing.has_valid_tlas()
        || app
            .data
            .raytracing
            .acceleration_structure
            .as_ref()
            .map(|a| a.hit_shading_table.is_some())
            .unwrap_or(false)
            == false
    {
        return Ok(());
    }

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

    let waters: Vec<_> = app.data.ecs_world.query_waters();

    let instance_count = waters
        .len()
        .min(thyllore_vulkan_core::resource::MAX_WATER_INSTANCES);

    if instance_count == 0 {
        return Ok(());
    }
    // Re-write descriptor with current TLAS and hit table (only if TLAS handle changed)
    {
        let accel = app.data.raytracing.acceleration_structure.as_ref().unwrap();
        let tlas = accel.tlas.acceleration_structure.unwrap();
        if tlas != app.data.raytracing.water_descriptor_tlas.get() {
            let hit_table = accel.hit_shading_table.as_ref().unwrap().buffer;
            let (scene_color_view, scene_color_sampler) = water_buffer.scene_color_binding();
            let water_ubo = app.data.raytracing.water_ubo.as_ref().unwrap();
            descriptor.write_all(
                ctx.device,
                water_ubo,
                scene_color_view,
                scene_color_sampler,
                water_buffer.history_image_views,
                water_buffer.history_sampler,
                water_buffer.trace_image_view,
                water_buffer.history_sampler,
                tlas,
                hit_table,
            )?;

            if let Some(trace_descriptor) = app.data.raytracing.water_trace_descriptor.as_ref() {
                if let Some(water_ubo) = app.data.raytracing.water_ubo.as_ref() {
                    trace_descriptor.write_all(
                        ctx.device,
                        tlas,
                        water_buffer.trace_image_view,
                        water_ubo,
                        hit_table,
                    )?;
                }
            }

            app.data.raytracing.water_descriptor_tlas.set(tlas);
        }
    }

    let settings = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::WaterRenderSettings>()
        .map(|s| *s)
        .unwrap_or_default();
    if settings.secondary_rays == thyllore_effect_core::WaterSecondaryRays::RayTracingPipeline {
        if let (Some(trace_pipeline), Some(trace_descriptor)) = (
            app.data.raytracing.water_trace_pipeline.as_ref(),
            app.data.raytracing.water_trace_descriptor.as_ref(),
        ) {
            if let Some(effect) = app
                .data
                .ecs_world
                .get_component::<crate::ecs::component::WaterTorusEffect>(waters[0])
            {
                let device = &ctx.device.device;
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    trace_pipeline.pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    trace_pipeline.pipeline_layout,
                    0,
                    &[trace_descriptor.descriptor_set],
                    &[],
                );
                let radii = [effect.major_radius, effect.minor_radius];
                let radii_bytes = std::slice::from_raw_parts(radii.as_ptr() as *const u8, 8);
                device.cmd_push_constants(
                    command_buffer,
                    trace_pipeline.pipeline_layout,
                    vk::ShaderStageFlags::INTERSECTION_KHR,
                    0,
                    radii_bytes,
                );
                let projection = app
                    .data
                    .ecs_world
                    .resource::<crate::ecs::resource::ProjectionData>();
                let inv_view_proj = crate::ecs::systems::water::probe::inverse_view_proj_f64(
                    projection.proj,
                    projection.view,
                );
                let view_inverse = projection
                    .view
                    .invert()
                    .unwrap_or_else(cgmath::Matrix4::identity);
                let m: &[f32; 16] = inv_view_proj.as_ref();
                let mut frame_data = [0.0f32; 20];
                frame_data[..16].copy_from_slice(m);
                frame_data[16] = view_inverse[3][0];
                frame_data[17] = view_inverse[3][1];
                frame_data[18] = view_inverse[3][2];
                frame_data[19] = 1.0;
                let frame_bytes = std::slice::from_raw_parts(frame_data.as_ptr() as *const u8, 80);
                device.cmd_push_constants(
                    command_buffer,
                    trace_pipeline.pipeline_layout,
                    vk::ShaderStageFlags::RAYGEN_KHR,
                    16,
                    frame_bytes,
                );
                let light_position = app
                    .data
                    .ecs_world
                    .resource::<crate::ecs::resource::LightState>()
                    .light_position;
                let mut light_data = [0.0f32; 8];
                light_data[0] = light_position.x;
                light_data[1] = light_position.y;
                light_data[2] = light_position.z;
                light_data[3] = 1.0;
                light_data[4] = 1.0;
                light_data[5] = 1.0;
                light_data[6] = 1.0;
                light_data[7] = 1.0;
                let light_bytes = std::slice::from_raw_parts(light_data.as_ptr() as *const u8, 32);
                device.cmd_push_constants(
                    command_buffer,
                    trace_pipeline.pipeline_layout,
                    vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                    96,
                    light_bytes,
                );
                let extent = water_buffer.extent();
                device.cmd_trace_rays_khr(
                    command_buffer,
                    &trace_pipeline.raygen_region,
                    &trace_pipeline.miss_region,
                    &trace_pipeline.hit_region,
                    &trace_pipeline.callable_region,
                    extent.width,
                    extent.height,
                    1,
                );
                let barrier = vk::ImageMemoryBarrier::builder()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(water_buffer.trace_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build();
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[] as &[vk::MemoryBarrier],
                    &[] as &[vk::BufferMemoryBarrier],
                    &[barrier],
                );
            }
        }
    }

    let hdr_buffer = app
        .data
        .viewport
        .hdr_buffer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("HDR buffer not initialized"))?;

    record_water_caustic_pass(app, water_buffer, hdr_buffer, waters[0], command_buffer);

    thyllore_vulkan_core::renderer::record_water_scene_color_copy(
        &ctx,
        hdr_buffer.color_image,
        water_buffer,
        command_buffer,
    );

    let first_water_accum = waters.first().and_then(|water| {
        app.data
            .ecs_world
            .get_component::<crate::ecs::component::WaterTemporalAccum>(*water)
            .cloned()
    });

    let history_index = first_water_accum
        .as_ref()
        .map(|accum| (accum.frame_index & 1) as usize)
        .unwrap_or(0);

    if first_water_accum
        .as_ref()
        .is_some_and(|accum| accum.history_invalidated)
    {
        clear_water_history_images(&ctx.device.device, water_buffer, command_buffer);
    }

    for i in 0..instance_count {
        let water = waters[i];
        let effect = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::WaterTorusEffect>(water)
            .ok_or_else(|| anyhow::anyhow!("Missing WaterTorusEffect for instance {}", i))?;

        let accum = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::WaterTemporalAccum>(water)
            .cloned()
            .unwrap_or_default();

        // Build UBO for this instance
        let mut ubo = thyllore_effect_core::build_water_ubo(effect, accum.frame_index as u32);

        // Overwrite inv_view_proj with f64-precision calculation for probe consistency
        let projection = app
            .data
            .ecs_world
            .resource::<crate::ecs::resource::ProjectionData>();
        ubo.inv_view_proj = crate::ecs::systems::water::probe::inverse_view_proj_f64(
            projection.proj,
            projection.view,
        );

        ubo.temporal = [accum.weight, accum.frame_index as f32, 0.0, 0.0];

        let settings = app
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::WaterRenderSettings>()
            .map(|settings| *settings)
            .unwrap_or_default();
        ubo.composite[3] = settings.caustic_debug as f32;

        let Some(water_ubo) = app.data.raytracing.water_ubo.as_ref() else {
            return Ok(());
        };
        let ubo_dynamic_offset = water_ubo.slot_offset(i)? as u32;
        water_ubo.record_update(
            &ctx.device.device,
            command_buffer,
            i,
            &ubo,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        )?;

        // Compute per-instance scissor using the model matrix
        let Some(scissor) = compute_water_scissor(
            app,
            water_buffer.extent(),
            &ubo.model,
            effect.major_radius,
            effect.minor_radius,
        ) else {
            continue;
        };

        let push_constants = thyllore_vulkan_core::renderer::WaterPushConstants::new(
            settings.secondary_rays.as_shader_value(),
            settings.debug_view,
        );

        // Record shading pass for this instance
        thyllore_vulkan_core::renderer::record_water_shading_pass(
            &ctx,
            water_buffer,
            shading_pipeline,
            descriptor,
            ubo_dynamic_offset,
            scissor,
            push_constants,
            image_index,
            history_index,
            command_buffer,
        )?;
    }

    Ok(())
}

/// Must match CAUSTIC_GRID_SIZE and local_size in waterCausticSplat.comp
const CAUSTIC_GRID_SIZE: u32 = 256;
const CAUSTIC_WORKGROUP_SIZE: u32 = 16;

const COLOR_SUBRESOURCE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

fn build_color_image_barrier(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) -> vk::ImageMemoryBarrier {
    vk::ImageMemoryBarrier::builder()
        .image(image)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(COLOR_SUBRESOURCE_RANGE)
        .build()
}

/// Both history images stay in SHADER_READ_ONLY_OPTIMAL between frames (shading render pass final layout).
unsafe fn clear_water_history_images(
    device: &Device,
    water_buffer: &thyllore_vulkan_core::resource::WaterBuffer,
    command_buffer: vk::CommandBuffer,
) {
    let black = vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 1.0],
    };

    for &image in &water_buffer.history_images {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[build_color_image_barrier(
                image,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::SHADER_READ,
                vk::AccessFlags::TRANSFER_WRITE,
            )],
        );

        device.cmd_clear_color_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &black,
            &[COLOR_SUBRESOURCE_RANGE],
        );

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[build_color_image_barrier(
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            )],
        );
    }
}

/// Splats refracted light into the accumulation image and adds it to the HDR color.
/// The G-buffer position image stays in GENERAL from the ray query pass.
unsafe fn record_water_caustic_pass(
    app: &App,
    water_buffer: &thyllore_vulkan_core::resource::WaterBuffer,
    hdr_buffer: &thyllore_vulkan_core::resource::HdrBuffer,
    water: crate::ecs::world::Entity,
    command_buffer: vk::CommandBuffer,
) {
    let caustic_strength = app
        .data
        .ecs_world
        .get_component::<crate::ecs::component::WaterTorusEffect>(water)
        .map(|effect| effect.caustic_strength)
        .unwrap_or(0.0);
    if caustic_strength <= 0.0 {
        return;
    }

    let (Some(splat_pipeline), Some(apply_pipeline), Some(descriptor)) = (
        app.data.raytracing.water_caustic_splat_pipeline.as_ref(),
        app.data.raytracing.water_caustic_apply_pipeline.as_ref(),
        app.data.raytracing.water_caustic_descriptor.as_ref(),
    ) else {
        return;
    };

    if !app.data.raytracing.has_valid_tlas()
        || descriptor.splat_descriptor_set == vk::DescriptorSet::null()
    {
        return;
    }

    let device = &app.rrdevice.device;
    let accum_image = water_buffer.caustic_accum_image;
    let hdr_image = hdr_buffer.color_image;

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[build_color_image_barrier(
            accum_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
        )],
    );

    device.cmd_clear_color_image(
        command_buffer,
        accum_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &vk::ClearColorValue { uint32: [0; 4] },
        &[COLOR_SUBRESOURCE_RANGE],
    );

    let pre_splat_barriers = [
        build_color_image_barrier(
            accum_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        ),
        build_color_image_barrier(
            hdr_image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        ),
    ];
    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &pre_splat_barriers,
    );

    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        splat_pipeline.pipeline,
    );
    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        splat_pipeline.pipeline_layout,
        0,
        &[descriptor.splat_descriptor_set],
        &[],
    );
    let splat_group_count = CAUSTIC_GRID_SIZE / CAUSTIC_WORKGROUP_SIZE;
    device.cmd_dispatch(command_buffer, splat_group_count, splat_group_count, 1);

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[build_color_image_barrier(
            accum_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ,
        )],
    );

    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        apply_pipeline.pipeline,
    );
    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        apply_pipeline.pipeline_layout,
        0,
        &[descriptor.apply_descriptor_set],
        &[],
    );
    device.cmd_dispatch(
        command_buffer,
        hdr_buffer.width.div_ceil(CAUSTIC_WORKGROUP_SIZE),
        hdr_buffer.height.div_ceil(CAUSTIC_WORKGROUP_SIZE),
        1,
    );

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[build_color_image_barrier(
            hdr_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ,
        )],
    );
}

fn compute_water_scissor(
    app: &App,
    extent: vk::Extent2D,
    model: &cgmath::Matrix4<f32>,
    major_radius: f32,
    minor_radius: f32,
) -> Option<vk::Rect2D> {
    use crate::ecs::resource::ProjectionData;
    const SCISSOR_MARGIN_PX: f32 = 2.0;

    let Some(projection) = app.data.ecs_world.get_resource::<ProjectionData>() else {
        return Some(full_extent_scissor(extent));
    };
    let view_proj = projection.proj * projection.view;

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for corner in thyllore_effect_core::water_local_bounds_corners(major_radius, minor_radius) {
        let clip = view_proj * model * cgmath::vec4(corner.x, corner.y, corner.z, 1.0);
        if clip.w <= 0.0 {
            return Some(full_extent_scissor(extent));
        }
        let screen_x = (clip.x / clip.w + 1.0) * 0.5 * extent.width as f32;
        let screen_y = (clip.y / clip.w + 1.0) * 0.5 * extent.height as f32;
        min_x = min_x.min(screen_x);
        min_y = min_y.min(screen_y);
        max_x = max_x.max(screen_x);
        max_y = max_y.max(screen_y);
    }

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

fn full_extent_scissor(extent: vk::Extent2D) -> vk::Rect2D {
    vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(extent)
        .build()
}

fn compute_flame_scissor(
    app: &App,
    extent: vk::Extent2D,
    model: &cgmath::Matrix4<f32>,
    bend_offset: [f32; 2],
    support_scale: f32,
    support_margin: f32,
    proxy_pad: thyllore_effect_core::FlameProxyPad,
) -> Option<vk::Rect2D> {
    use crate::ecs::resource::ProjectionData;
    const SCISSOR_MARGIN_PX: f32 = 2.0;

    let Some(projection) = app.data.ecs_world.get_resource::<ProjectionData>() else {
        return Some(full_extent_scissor(extent));
    };
    let view_proj = projection.proj * projection.view;

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let bounds = thyllore_effect_core::flame_local_bounds(
        bend_offset,
        support_scale,
        support_margin,
        proxy_pad,
    );
    for corner in thyllore_effect_core::flame_local_bounds_corners(&bounds) {
        let clip = view_proj * model * cgmath::vec4(corner.x, corner.y, corner.z, 1.0);
        if clip.w <= 0.0 {
            return Some(full_extent_scissor(extent));
        }
        let screen_x = (clip.x / clip.w + 1.0) * 0.5 * extent.width as f32;
        let screen_y = (clip.y / clip.w + 1.0) * 0.5 * extent.height as f32;
        min_x = min_x.min(screen_x);
        min_y = min_y.min(screen_y);
        max_x = max_x.max(screen_x);
        max_y = max_y.max(screen_y);
    }

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
        plume_data,
    )?;
    // Grid is already drawn inside record_composite_to_hdr, before the flame composite.
    // Drawing it again here would put it on top of the flame.
    super::OverlayRenderer::new(app).draw_all_overlays(command_buffer, image_index, false)?;
    thyllore_vulkan_core::renderer::end_tonemap_render_pass(&ctx, command_buffer);

    Ok(())
}
