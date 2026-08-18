use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::descriptor::RRToneMapDescriptorSet;
use crate::frame_context::FrameRenderContext;
use crate::pipeline::RRPipeline;
use thyllore_render_core::{BloomSettings, Exposure, LensEffects, ToneMapping};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ToneMapPushConstants {
    tone_map_operator: i32,
    gamma: f32,
    exposure_value: f32,
    vignette_intensity: f32,
    chromatic_aberration_intensity: f32,
    bloom_intensity: f32,
    _pad: [f32; 2],
    plume_position: [f32; 4],
    plume_params0: [f32; 4],
    plume_params1: [f32; 4],
    plume_params2: [f32; 4],
}

pub unsafe fn begin_tonemap_render_pass(
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
            depth: 0.0,
            stencil: 0,
        },
    };
    let resolve_clear_value = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };

    let clear_values = vec![color_clear_value, depth_clear_value, resolve_clear_value];

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);

    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
}

pub unsafe fn record_tonemap_draw(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    descriptor: &RRToneMapDescriptorSet,
    tonemap: &ToneMapping,
    exposure: &Exposure,
    lens: &LensEffects,
    bloom: &BloomSettings,
    extent: vk::Extent2D,
    cmd: vk::CommandBuffer,
    plume: Option<([f32; 4], [f32; 4], [f32; 4], [f32; 4])>,
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

    let operator = if tonemap.enabled {
        tonemap.operator as i32
    } else {
        0
    };
    let vignette_intensity = if lens.vignette_enabled {
        lens.vignette_intensity
    } else {
        0.0
    };
    let chromatic_aberration_intensity = if lens.chromatic_aberration_enabled {
        lens.chromatic_aberration_intensity
    } else {
        0.0
    };
    let bloom_intensity = if bloom.enabled { bloom.intensity } else { 0.0 };

    let (plume_position, plume_params0, plume_params1, plume_params2) = match plume {
        Some((pos, p0, p1, p2)) => (pos, p0, p1, p2),
        None => ([0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]),
    };

    let push_constants = ToneMapPushConstants {
        tone_map_operator: operator,
        gamma: tonemap.gamma,
        exposure_value: exposure.exposure_value,
        vignette_intensity,
        chromatic_aberration_intensity,
        bloom_intensity,
        _pad: [0.0; 2],
        plume_position,
        plume_params0,
        plume_params1,
        plume_params2,
    };

    let push_bytes = std::slice::from_raw_parts(
        &push_constants as *const ToneMapPushConstants as *const u8,
        std::mem::size_of::<ToneMapPushConstants>(),
    );

    device.cmd_push_constants(
        cmd,
        pipeline.pipeline_layout,
        vk::ShaderStageFlags::FRAGMENT,
        0,
        push_bytes,
    );

    device.cmd_draw(cmd, 3, 1, 0, 0);

    Ok(())
}

pub unsafe fn end_tonemap_render_pass(ctx: &FrameRenderContext, cmd: vk::CommandBuffer) {
    ctx.device.device.cmd_end_render_pass(cmd);
}
