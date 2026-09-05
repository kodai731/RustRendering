use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::descriptor::RRWindDescriptorSet;
use crate::frame_context::FrameRenderContext;
use crate::pipeline::RRPipeline;
use crate::renderer::push_constants::WindPushConstants;
use crate::resource::wind_buffer::WindBuffer;

pub unsafe fn record_wind_shading_pass(
    ctx: &FrameRenderContext,
    wind_buffer: &WindBuffer,
    pipeline: &RRPipeline,
    descriptor: &RRWindDescriptorSet,
    ubo_dynamic_offset: u32,
    scissor: vk::Rect2D,
    push_constants: WindPushConstants,
    image_index: usize,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(wind_buffer.render_pass)
        .framebuffer(wind_buffer.framebuffer)
        .render_area(scissor);
    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(wind_buffer.width as f32)
        .height(wind_buffer.height as f32)
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
        &[ubo_dynamic_offset],
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
