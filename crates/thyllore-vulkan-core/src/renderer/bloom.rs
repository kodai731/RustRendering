use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::descriptor::RRBloomDescriptorSets;
use crate::frame_context::FrameRenderContext;
use crate::pipeline::RRPipeline;
use crate::resource::bloom_chain::{BloomChain, BloomMipTarget};
use thyllore_render_core::BloomSettings;

#[repr(C)]
#[derive(Clone, Copy)]
struct BloomDownsamplePushConstants {
    threshold: f32,
    knee: f32,
    is_first_pass: i32,
}

/// Downsamples into `mips[mip_index]` from the HDR color (mip 0) or the previous mip.
/// The caller brings the source mip to SHADER_READ_ONLY_OPTIMAL; the render pass discards the target.
pub unsafe fn record_bloom_downsample_mip(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    bloom_descriptors: &RRBloomDescriptorSets,
    bloom_chain: &BloomChain,
    mips: &[BloomMipTarget],
    mip_index: usize,
    frame_slot: usize,
    settings: &BloomSettings,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;
    let Some(mip) = mips.get(mip_index) else {
        return Ok(());
    };
    let extent = mip.extent;

    begin_downsample_render_pass(device, bloom_chain, cmd, mip.framebuffer, extent);
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
    set_viewport_and_scissor(device, cmd, extent);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.pipeline_layout,
        0,
        &[bloom_descriptors.downsample_set(frame_slot, mip_index)?],
        &[],
    );

    let push_constants = BloomDownsamplePushConstants {
        threshold: settings.threshold,
        knee: settings.knee,
        is_first_pass: if mip_index == 0 { 1 } else { 0 },
    };
    let push_bytes = std::slice::from_raw_parts(
        &push_constants as *const BloomDownsamplePushConstants as *const u8,
        std::mem::size_of::<BloomDownsamplePushConstants>(),
    );
    device.cmd_push_constants(
        cmd,
        pipeline.pipeline_layout,
        vk::ShaderStageFlags::FRAGMENT,
        0,
        push_bytes,
    );
    device.cmd_draw(cmd, 3, 1, 0, 0);
    device.cmd_end_render_pass(cmd);

    Ok(())
}

/// Number of upsample passes for a chain of `mip_count` mips: each pass blends mip n+1 into mip n.
pub fn bloom_upsample_pass_count(mip_count: usize) -> usize {
    mip_count.saturating_sub(1)
}

/// Target mip written by upsample pass `pass_index` (the passes walk from the smallest mip up to mip 0).
pub fn bloom_upsample_target_mip(mip_count: usize, pass_index: usize) -> Option<usize> {
    let pass_count = bloom_upsample_pass_count(mip_count);
    (pass_index < pass_count).then(|| pass_count - 1 - pass_index)
}

/// Blends the next smaller mip into the target mip of `pass_index`.
/// The caller brings the target mip to COLOR_ATTACHMENT_OPTIMAL (the render pass loads it)
/// and the source mip to SHADER_READ_ONLY_OPTIMAL.
pub unsafe fn record_bloom_upsample_pass(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    bloom_descriptors: &RRBloomDescriptorSets,
    bloom_chain: &BloomChain,
    mips: &[BloomMipTarget],
    pass_index: usize,
    frame_slot: usize,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;
    let Some(target_mip_index) = bloom_upsample_target_mip(mips.len(), pass_index) else {
        return Ok(());
    };
    let mip = &mips[target_mip_index];
    let extent = mip.extent;

    begin_upsample_render_pass(device, bloom_chain, cmd, mip.framebuffer, extent);
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
    set_viewport_and_scissor(device, cmd, extent);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.pipeline_layout,
        0,
        &[bloom_descriptors.upsample_set(frame_slot, pass_index)?],
        &[],
    );
    device.cmd_draw(cmd, 3, 1, 0, 0);
    device.cmd_end_render_pass(cmd);

    Ok(())
}

unsafe fn begin_downsample_render_pass(
    device: &Device,
    bloom_chain: &BloomChain,
    cmd: vk::CommandBuffer,
    framebuffer: vk::Framebuffer,
    extent: vk::Extent2D,
) {
    let clear_value = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(bloom_chain.downsample_render_pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        })
        .clear_values(std::slice::from_ref(&clear_value));

    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
}

unsafe fn begin_upsample_render_pass(
    device: &Device,
    bloom_chain: &BloomChain,
    cmd: vk::CommandBuffer,
    framebuffer: vk::Framebuffer,
    extent: vk::Extent2D,
) {
    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(bloom_chain.upsample_render_pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        });

    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
}

unsafe fn set_viewport_and_scissor(device: &Device, cmd: vk::CommandBuffer, extent: vk::Extent2D) {
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
}
