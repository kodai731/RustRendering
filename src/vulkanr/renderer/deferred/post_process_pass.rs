use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;

struct BloomFrame<'a> {
    bloom_chain: &'a thyllore_vulkan_core::resource::BloomChain,
    downsample_pipeline: &'a thyllore_vulkan_core::pipeline::RRPipeline,
    upsample_pipeline: &'a thyllore_vulkan_core::pipeline::RRPipeline,
    descriptors: &'a thyllore_vulkan_core::descriptor::RRBloomDescriptorSets,
    mips: &'a [thyllore_vulkan_core::resource::BloomMipTarget],
    settings: crate::ecs::resource::BloomSettings,
}

fn bloom_frame(app: &App) -> Result<Option<BloomFrame<'_>>> {
    let Some(settings) = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::BloomSettings>()
        .filter(|settings| settings.enabled)
        .map(|settings| settings.clone())
    else {
        return Ok(None);
    };
    let mips = &app.data.post_process.bloom_mips;
    if mips.is_empty() {
        return Ok(None);
    }

    let (Some(bloom_chain), Some(downsample_pipeline), Some(upsample_pipeline)) = (
        app.data.viewport.bloom_chain.as_ref(),
        app.data.raytracing.bloom_downsample_pipeline.as_ref(),
        app.data.raytracing.bloom_upsample_pipeline.as_ref(),
    ) else {
        return Ok(None);
    };
    let descriptors = app
        .data
        .raytracing
        .bloom_descriptors
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Bloom descriptors not initialized"))?;

    Ok(Some(BloomFrame {
        bloom_chain,
        downsample_pipeline,
        upsample_pipeline,
        descriptors,
        mips,
        settings,
    }))
}

pub unsafe fn record_bloom_downsample(
    app: &App,
    command_buffer: vk::CommandBuffer,
    mip_index: usize,
    frame_slot: usize,
) -> Result<()> {
    let Some(bloom) = bloom_frame(app)? else {
        return Ok(());
    };
    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, 0);
    thyllore_vulkan_core::renderer::record_bloom_downsample_mip(
        &ctx,
        bloom.downsample_pipeline,
        bloom.descriptors,
        bloom.bloom_chain,
        bloom.mips,
        mip_index,
        frame_slot,
        &bloom.settings,
        command_buffer,
    )
}

pub unsafe fn record_bloom_upsample(
    app: &App,
    command_buffer: vk::CommandBuffer,
    pass_index: usize,
    frame_slot: usize,
) -> Result<()> {
    let Some(bloom) = bloom_frame(app)? else {
        return Ok(());
    };
    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, 0);
    thyllore_vulkan_core::renderer::record_bloom_upsample_pass(
        &ctx,
        bloom.upsample_pipeline,
        bloom.descriptors,
        bloom.bloom_chain,
        bloom.mips,
        pass_index,
        frame_slot,
        command_buffer,
    )
}

pub unsafe fn record_dof(app: &App, command_buffer: vk::CommandBuffer) -> Result<()> {
    let (Some(pipeline), Some(dof_descriptor), Some(dof_buffer)) = (
        app.data.raytracing.dof_pipeline.as_ref(),
        app.data.raytracing.dof_descriptor.as_ref(),
        app.data.viewport.dof_buffer.as_ref(),
    ) else {
        return Ok(());
    };
    let dof_framebuffer = app.data.post_process.dof_framebuffer;
    if dof_framebuffer == vk::Framebuffer::null() {
        return Ok(());
    }

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
        dof_framebuffer,
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
        frame_slot,
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
