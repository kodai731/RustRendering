use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::descriptor::RRRayQueryDescriptorSet;
use crate::frame_context::FrameRenderContext;
use crate::pipeline::RRPipeline;
use crate::resource::RRGBuffer;

pub unsafe fn record_ray_query_pass(
    ctx: &FrameRenderContext,
    gbuffer: &RRGBuffer,
    pipeline: &RRPipeline,
    descriptor: &RRRayQueryDescriptorSet,
    normal_offset: f32,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;

    insert_pre_compute_barriers(device, gbuffer, cmd);
    dispatch_compute(device, gbuffer, pipeline, descriptor, normal_offset, cmd);
    insert_post_compute_barriers(device, gbuffer, cmd);

    Ok(())
}

unsafe fn insert_pre_compute_barriers(
    device: &Device,
    gbuffer: &RRGBuffer,
    cmd: vk::CommandBuffer,
) {
    let image_barriers = [
        vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(gbuffer.position_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build(),
        vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(gbuffer.normal_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build(),
        vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(gbuffer.shadow_mask_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build(),
    ];

    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &image_barriers,
    );
}

unsafe fn dispatch_compute(
    device: &Device,
    gbuffer: &RRGBuffer,
    pipeline: &RRPipeline,
    descriptor: &RRRayQueryDescriptorSet,
    normal_offset: f32,
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

    let push_constant_data = [normal_offset];
    device.cmd_push_constants(
        cmd,
        pipeline.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        std::slice::from_raw_parts(
            push_constant_data.as_ptr() as *const u8,
            std::mem::size_of::<f32>(),
        ),
    );

    let group_count_x = (gbuffer.width + 15) / 16;
    let group_count_y = (gbuffer.height + 15) / 16;
    device.cmd_dispatch(cmd, group_count_x, group_count_y, 1);
}

unsafe fn insert_post_compute_barriers(
    device: &Device,
    gbuffer: &RRGBuffer,
    cmd: vk::CommandBuffer,
) {
    let shadow_barrier = vk::ImageMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image(gbuffer.shadow_mask_image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .build();

    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[shadow_barrier],
    );
}
