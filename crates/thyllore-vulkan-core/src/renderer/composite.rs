use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::descriptor::RRCompositeDescriptorSet;
use crate::frame_context::FrameRenderContext;
use crate::pipeline::RRPipeline;

pub unsafe fn begin_composite_render_pass(
    ctx: &FrameRenderContext,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    extent: vk::Extent2D,
    attachment_count: usize,
    cmd: vk::CommandBuffer,
) {
    let device = &ctx.device.device;
    let render_area = vk::Rect2D::builder()
        .offset(vk::Offset2D::default())
        .extent(extent);

    let color_clear_value = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };
    let depth_clear_value = vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 0.0,
            stencil: 0,
        },
    };

    let clear_values: Vec<vk::ClearValue> = if attachment_count == 3 {
        vec![color_clear_value, depth_clear_value, color_clear_value]
    } else {
        vec![color_clear_value, depth_clear_value]
    };

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);

    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
}

pub unsafe fn record_composite_draw(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    descriptor: &RRCompositeDescriptorSet,
    extent: vk::Extent2D,
    debug_view_mode_value: i32,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(extent.width as f32)
        .height(extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(extent);

    device.cmd_set_viewport(cmd, 0, &[viewport]);
    device.cmd_set_scissor(cmd, 0, &[scissor]);

    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.pipeline_layout,
        0,
        &[descriptor.descriptor_set],
        &[],
    );

    let push_constants = [debug_view_mode_value];
    let push_constant_bytes = std::slice::from_raw_parts(
        push_constants.as_ptr() as *const u8,
        std::mem::size_of_val(&push_constants),
    );

    device.cmd_push_constants(
        cmd,
        pipeline.pipeline_layout,
        vk::ShaderStageFlags::FRAGMENT,
        0,
        push_constant_bytes,
    );

    device.cmd_draw(cmd, 3, 1, 0, 0);

    Ok(())
}

pub unsafe fn end_composite_render_pass(ctx: &FrameRenderContext, cmd: vk::CommandBuffer) {
    ctx.device.device.cmd_end_render_pass(cmd);
}

pub unsafe fn begin_hdr_render_pass(
    ctx: &FrameRenderContext,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    extent: vk::Extent2D,
    cmd: vk::CommandBuffer,
) {
    let device = &ctx.device.device;
    let render_area = vk::Rect2D::builder()
        .offset(vk::Offset2D::default())
        .extent(extent);

    let color_clear_value = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };

    let depth_clear_value = vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        },
    };

    let clear_values = [color_clear_value, depth_clear_value];

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);

    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
}

pub unsafe fn record_composite_to_hdr_pass(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    descriptor: &RRCompositeDescriptorSet,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    extent: vk::Extent2D,
    debug_view_mode_value: i32,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    begin_hdr_render_pass(ctx, render_pass, framebuffer, extent, cmd);

    record_composite_draw(
        ctx,
        pipeline,
        descriptor,
        extent,
        debug_view_mode_value,
        cmd,
    )?;

    end_composite_render_pass(ctx, cmd);

    Ok(())
}
