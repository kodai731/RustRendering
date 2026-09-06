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

pub unsafe fn record_bloom_pass(
    ctx: &FrameRenderContext,
    downsample_pipeline: &RRPipeline,
    upsample_pipeline: &RRPipeline,
    bloom_descriptors: &RRBloomDescriptorSets,
    bloom_chain: &BloomChain,
    mips: &[BloomMipTarget],
    frame_slot: usize,
    settings: &BloomSettings,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    record_downsample_passes(
        ctx,
        downsample_pipeline,
        bloom_descriptors,
        bloom_chain,
        mips,
        frame_slot,
        settings.threshold,
        settings.knee,
        cmd,
    )?;
    record_upsample_passes(
        ctx,
        upsample_pipeline,
        bloom_descriptors,
        bloom_chain,
        mips,
        frame_slot,
        cmd,
    )?;
    Ok(())
}

unsafe fn record_downsample_passes(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    bloom_descriptors: &RRBloomDescriptorSets,
    bloom_chain: &BloomChain,
    mips: &[BloomMipTarget],
    frame_slot: usize,
    threshold: f32,
    knee: f32,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;

    for (i, mip) in mips.iter().enumerate() {
        let extent = mip.extent;

        begin_downsample_render_pass(device, bloom_chain, cmd, mip.framebuffer, extent);

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

        set_viewport_and_scissor(device, cmd, extent);

        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            0,
            &[bloom_descriptors.downsample_set(frame_slot, i)?],
            &[],
        );

        let push_constants = BloomDownsamplePushConstants {
            threshold,
            knee,
            is_first_pass: if i == 0 { 1 } else { 0 },
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
    }

    Ok(())
}

unsafe fn record_upsample_passes(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    bloom_descriptors: &RRBloomDescriptorSets,
    bloom_chain: &BloomChain,
    mips: &[BloomMipTarget],
    frame_slot: usize,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;
    let mip_count = mips.len();
    if mip_count < 2 {
        return Ok(());
    }

    for (pass_idx, target_mip_idx) in (0..mip_count - 1).rev().enumerate() {
        let mip = &mips[target_mip_idx];
        let extent = mip.extent;

        transition_to_color_attachment(device, cmd, mip.image);

        begin_upsample_render_pass(device, bloom_chain, cmd, mip.framebuffer, extent);

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

        set_viewport_and_scissor(device, cmd, extent);

        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            0,
            &[bloom_descriptors.upsample_set(frame_slot, pass_idx)?],
            &[],
        );

        device.cmd_draw(cmd, 3, 1, 0, 0);

        device.cmd_end_render_pass(cmd);
    }

    Ok(())
}

unsafe fn transition_to_color_attachment(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
) {
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        );

    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );
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
