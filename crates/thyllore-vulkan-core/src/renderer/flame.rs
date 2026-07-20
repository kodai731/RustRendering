use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::descriptor::RRFlameDescriptorSet;
use crate::frame_context::FrameRenderContext;
use crate::pipeline::RRPipeline;
use crate::renderer::push_constants::FlamePushConstants;
use crate::resource::flame_buffer::{FlameBuffer, FLAME_INTERVAL_CLEAR};
use thyllore_render_core::FlameUBO;

// In-command-buffer update keeps the single UBO race-free across frames in
// flight: the transfer executes in submission order before this frame's
// flame passes read it.
pub unsafe fn record_flame_ubo_update(
    ctx: &FrameRenderContext,
    ubo: &FlameUBO,
    buffer: vk::Buffer,
    cmd: vk::CommandBuffer,
) {
    let device = &ctx.device.device;

    let before_write = vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::UNIFORM_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(std::mem::size_of::<FlameUBO>() as vk::DeviceSize)
        .build();
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::GEOMETRY_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[before_write],
        &[] as &[vk::ImageMemoryBarrier],
    );

    let bytes = std::slice::from_raw_parts(
        ubo as *const FlameUBO as *const u8,
        std::mem::size_of::<FlameUBO>(),
    );
    device.cmd_update_buffer(cmd, buffer, 0, bytes);

    let after_write = vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::UNIFORM_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(std::mem::size_of::<FlameUBO>() as vk::DeviceSize)
        .build();
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::GEOMETRY_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[after_write],
        &[] as &[vk::ImageMemoryBarrier],
    );
}

pub unsafe fn record_flame_thickness_pass(
    ctx: &FrameRenderContext,
    flame_buffer: &FlameBuffer,
    pipeline: &RRPipeline,
    descriptor: &RRFlameDescriptorSet,
    image_index: usize,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;

    let render_area = vk::Rect2D::builder()
        .offset(vk::Offset2D::default())
        .extent(flame_buffer.extent());

    let clear_values = [
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        },
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [FLAME_INTERVAL_CLEAR, FLAME_INTERVAL_CLEAR, 0.0, 0.0],
            },
        },
    ];

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(flame_buffer.thickness_render_pass)
        .framebuffer(flame_buffer.thickness_framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);

    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(flame_buffer.width as f32)
        .height(flame_buffer.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);
    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(flame_buffer.extent());
    device.cmd_set_viewport(cmd, 0, &[viewport]);
    device.cmd_set_scissor(cmd, 0, &[scissor]);

    let frame_set = ctx.graphics.frame_set.sets[image_index];
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.pipeline_layout,
        0,
        &[frame_set, descriptor.descriptor_set],
        &[],
    );

    device.cmd_draw(cmd, 4, 1, 0, 0);

    device.cmd_end_render_pass(cmd);
    Ok(())
}

pub unsafe fn record_flame_shading_pass(
    ctx: &FrameRenderContext,
    flame_buffer: &FlameBuffer,
    pipeline: &RRPipeline,
    descriptor: &RRFlameDescriptorSet,
    scissor: vk::Rect2D,
    push_constants: FlamePushConstants,
    image_index: usize,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;

    let render_area = vk::Rect2D::builder()
        .offset(vk::Offset2D::default())
        .extent(flame_buffer.extent());

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(flame_buffer.shading_render_pass)
        .framebuffer(flame_buffer.shading_framebuffer)
        .render_area(render_area)
        .clear_values(&[]);

    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(flame_buffer.width as f32)
        .height(flame_buffer.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);
    device.cmd_set_viewport(cmd, 0, &[viewport]);
    device.cmd_set_scissor(cmd, 0, &[scissor]);

    let frame_set = ctx.graphics.frame_set.sets[image_index];
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.pipeline_layout,
        0,
        &[frame_set, descriptor.descriptor_set],
        &[],
    );

    device.cmd_push_constants(
        cmd,
        pipeline.pipeline_layout,
        vk::ShaderStageFlags::FRAGMENT,
        0,
        push_constants.as_bytes(),
    );

    device.cmd_draw(cmd, 3, 1, 0, 0);

    device.cmd_end_render_pass(cmd);
    Ok(())
}
