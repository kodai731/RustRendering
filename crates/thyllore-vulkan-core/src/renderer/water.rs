use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::descriptor::RRWaterDescriptorSet;
use crate::frame_context::FrameRenderContext;
use crate::pipeline::RRPipeline;
use crate::renderer::push_constants::WaterPushConstants;
use crate::resource::water_buffer::WaterBuffer;

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build()
}

fn image_barrier(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
) -> vk::ImageMemoryBarrier {
    vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_subresource_range())
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .build()
}

pub unsafe fn record_water_scene_color_copy(
    ctx: &FrameRenderContext,
    hdr_image: vk::Image,
    scene_color_image: vk::Image,
    extent: vk::Extent2D,
    cmd: vk::CommandBuffer,
) {
    let device = &ctx.device.device;

    let to_transfer = [
        image_barrier(
            hdr_image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_READ,
        ),
        image_barrier(
            scene_color_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        ),
    ];
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &to_transfer,
    );

    let subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
        .build();
    let region = vk::ImageCopy::builder()
        .src_subresource(subresource)
        .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .dst_subresource(subresource)
        .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .build();
    device.cmd_copy_image(
        cmd,
        hdr_image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        scene_color_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    let to_shader_read = [
        image_barrier(
            hdr_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::SHADER_READ,
        ),
        image_barrier(
            scene_color_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        ),
    ];
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &to_shader_read,
    );
}

pub unsafe fn record_water_shading_pass(
    ctx: &FrameRenderContext,
    water_buffer: &WaterBuffer,
    pipeline: &RRPipeline,
    descriptor: &RRWaterDescriptorSet,
    ubo_dynamic_offset: u32,
    scissor: vk::Rect2D,
    push_constants: WaterPushConstants,
    image_index: usize,
    frame_slot: usize,
    history_index: usize,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;

    let render_area = scissor;

    let clear_values: [vk::ClearValue; 0] = [];

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(water_buffer.render_pass)
        .framebuffer(water_buffer.framebuffers[history_index])
        .render_area(render_area)
        .clear_values(&clear_values);

    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(water_buffer.width as f32)
        .height(water_buffer.height as f32)
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
        &[
            frame_set,
            descriptor.descriptor_set(frame_slot, history_index)?,
        ],
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
