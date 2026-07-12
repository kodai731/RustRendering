use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::descriptor::{RRAutoExposureAverageDescriptorSet, RRAutoExposureHistogramDescriptorSet};
use crate::frame_context::FrameRenderContext;
use crate::pipeline::RRPipeline;
use crate::resource::AutoExposureBuffers;
use thyllore_render_core::AutoExposure;

#[repr(C)]
#[derive(Clone, Copy)]
struct HistogramPushConstants {
    min_log_luminance: f32,
    log_luminance_range: f32,
    pixel_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AveragePushConstants {
    low_percent: f32,
    high_percent: f32,
    time_delta: f32,
    adaptation_speed_up: f32,
    adaptation_speed_down: f32,
    min_ev: f32,
    max_ev: f32,
    min_log_luminance: f32,
    log_luminance_range: f32,
    pixel_count: u32,
}

pub unsafe fn record_auto_exposure_pass(
    ctx: &FrameRenderContext,
    histogram_pipeline: &RRPipeline,
    average_pipeline: &RRPipeline,
    histogram_descriptor: &RRAutoExposureHistogramDescriptorSet,
    average_descriptor: &RRAutoExposureAverageDescriptorSet,
    buffers: &AutoExposureBuffers,
    settings: &AutoExposure,
    delta_time: f32,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;
    let pixel_count = buffers.width * buffers.height;

    let histogram_push = HistogramPushConstants {
        min_log_luminance: settings.min_log_luminance,
        log_luminance_range: settings.log_luminance_range,
        pixel_count,
    };

    let average_push = AveragePushConstants {
        low_percent: settings.low_percent,
        high_percent: settings.high_percent,
        time_delta: delta_time,
        adaptation_speed_up: settings.adaptation_speed_up,
        adaptation_speed_down: settings.adaptation_speed_down,
        min_ev: settings.min_ev,
        max_ev: settings.max_ev,
        min_log_luminance: settings.min_log_luminance,
        log_luminance_range: settings.log_luminance_range,
        pixel_count,
    };

    insert_pre_histogram_barrier(device, cmd);
    dispatch_histogram(
        device,
        histogram_pipeline,
        histogram_descriptor,
        &histogram_push,
        buffers,
        cmd,
    );
    insert_histogram_to_average_barrier(device, cmd);
    dispatch_average(
        device,
        average_pipeline,
        average_descriptor,
        &average_push,
        cmd,
    );
    insert_post_average_barrier(device, cmd);

    Ok(())
}

unsafe fn insert_pre_histogram_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
        .build();

    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[] as &[vk::ImageMemoryBarrier],
    );
}

unsafe fn dispatch_histogram(
    device: &Device,
    pipeline: &RRPipeline,
    descriptor: &RRAutoExposureHistogramDescriptorSet,
    push: &HistogramPushConstants,
    buffers: &AutoExposureBuffers,
    cmd: vk::CommandBuffer,
) {
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);

    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        pipeline.pipeline_layout,
        0,
        &[descriptor.descriptor_set],
        &[],
    );

    let push_bytes = std::slice::from_raw_parts(
        push as *const HistogramPushConstants as *const u8,
        std::mem::size_of::<HistogramPushConstants>(),
    );

    device.cmd_push_constants(
        cmd,
        pipeline.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_bytes,
    );

    let group_count_x = (buffers.width + 15) / 16;
    let group_count_y = (buffers.height + 15) / 16;
    device.cmd_dispatch(cmd, group_count_x, group_count_y, 1);
}

unsafe fn insert_histogram_to_average_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .build();

    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[] as &[vk::ImageMemoryBarrier],
    );
}

unsafe fn dispatch_average(
    device: &Device,
    pipeline: &RRPipeline,
    descriptor: &RRAutoExposureAverageDescriptorSet,
    push: &AveragePushConstants,
    cmd: vk::CommandBuffer,
) {
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);

    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        pipeline.pipeline_layout,
        0,
        &[descriptor.descriptor_set],
        &[],
    );

    let push_bytes = std::slice::from_raw_parts(
        push as *const AveragePushConstants as *const u8,
        std::mem::size_of::<AveragePushConstants>(),
    );

    device.cmd_push_constants(
        cmd,
        pipeline.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_bytes,
    );

    device.cmd_dispatch(cmd, 1, 1, 1);
}

unsafe fn insert_post_average_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)
        .build();

    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::HOST,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[] as &[vk::ImageMemoryBarrier],
    );
}
