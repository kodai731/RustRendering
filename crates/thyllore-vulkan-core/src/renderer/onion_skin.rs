use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;

use crate::frame_context::FrameRenderContext;
use crate::renderer::onion_skin_buffers::OnionSkinGpuState;
use crate::renderer::push_constants::OnionSkinPushConstants;
use crate::resource::OnionSkinPassResources;

pub unsafe fn record_onion_skin_ghost_pass(
    ctx: &FrameRenderContext,
    resources: &OnionSkinPassResources,
    onion_skin_gpu: &OnionSkinGpuState,
    image_index: usize,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;
    begin_ghost_render_pass(device, resources, cmd);
    bind_ghost_pipeline_and_state(device, resources, cmd);
    draw_ghosts(ctx, resources, onion_skin_gpu, image_index, cmd)?;
    device.cmd_end_render_pass(cmd);
    Ok(())
}

pub unsafe fn record_onion_skin_composite_pass(
    ctx: &FrameRenderContext,
    resources: &OnionSkinPassResources,
    cmd: vk::CommandBuffer,
) {
    let device = &ctx.device.device;
    begin_composite_render_pass(device, resources, cmd);

    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        resources.composite_pipeline.pipeline,
    );

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(resources.width as f32)
        .height(resources.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(vk::Extent2D {
            width: resources.width,
            height: resources.height,
        });

    device.cmd_set_viewport(cmd, 0, &[viewport]);
    device.cmd_set_scissor(cmd, 0, &[scissor]);

    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        resources.composite_pipeline.pipeline_layout,
        0,
        &[resources.composite_descriptor_set],
        &[],
    );

    device.cmd_draw(cmd, 3, 1, 0, 0);

    device.cmd_end_render_pass(cmd);
}

unsafe fn begin_ghost_render_pass(
    device: &Device,
    resources: &OnionSkinPassResources,
    cmd: vk::CommandBuffer,
) {
    let render_area = vk::Rect2D::builder()
        .offset(vk::Offset2D::default())
        .extent(vk::Extent2D {
            width: resources.width,
            height: resources.height,
        });

    let clear_value = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 0.0],
        },
    };
    let clear_values = [clear_value];

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(resources.ghost_render_pass)
        .framebuffer(resources.ghost_framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);

    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
}

unsafe fn begin_composite_render_pass(
    device: &Device,
    resources: &OnionSkinPassResources,
    cmd: vk::CommandBuffer,
) {
    let render_area = vk::Rect2D::builder()
        .offset(vk::Offset2D::default())
        .extent(vk::Extent2D {
            width: resources.width,
            height: resources.height,
        });

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(resources.composite_render_pass)
        .framebuffer(resources.composite_framebuffer)
        .render_area(render_area)
        .clear_values(&[]);

    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
}

unsafe fn bind_ghost_pipeline_and_state(
    device: &Device,
    resources: &OnionSkinPassResources,
    cmd: vk::CommandBuffer,
) {
    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        resources.ghost_pipeline.pipeline,
    );

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(resources.width as f32)
        .height(resources.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(vk::Extent2D {
            width: resources.width,
            height: resources.height,
        });

    device.cmd_set_viewport(cmd, 0, &[viewport]);
    device.cmd_set_scissor(cmd, 0, &[scissor]);
}

unsafe fn draw_ghosts(
    ctx: &FrameRenderContext,
    resources: &OnionSkinPassResources,
    onion_skin_gpu: &OnionSkinGpuState,
    image_index: usize,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;
    let graphics_resources = ctx.graphics;

    let source_mesh_index = onion_skin_gpu
        .source_mesh_index
        .ok_or_else(|| anyhow!("No source mesh index for onion skin"))?;

    if source_mesh_index >= graphics_resources.meshes.len() {
        return Ok(());
    }

    let source_mesh = &graphics_resources.meshes[source_mesh_index];
    let pipeline_layout = resources.ghost_pipeline.pipeline_layout;

    for ghost_buffer in &onion_skin_gpu.ghost_buffers {
        if ghost_buffer.vertex_count == 0 {
            continue;
        }

        device.cmd_bind_vertex_buffers(cmd, 0, &[ghost_buffer.vertex_buffer], &[0]);

        device.cmd_bind_index_buffer(
            cmd,
            onion_skin_gpu.source_index_buffer,
            0,
            vk::IndexType::UINT32,
        );

        let frame_set = graphics_resources.frame_set.sets[image_index];
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0,
            &[frame_set],
            &[],
        );

        if let Some(material_id) = graphics_resources.get_material_id(source_mesh_index) {
            if let Some(material) = graphics_resources.materials.get(material_id) {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    1,
                    &[material.descriptor_set],
                    &[],
                );
            }
        }

        let object_set_idx = graphics_resources
            .objects
            .get_set_index(image_index, source_mesh.object_index);
        let object_set = graphics_resources.objects.sets[object_set_idx];
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            2,
            &[object_set],
            &[],
        );

        let push_constants =
            OnionSkinPushConstants::new(ghost_buffer.tint_color, ghost_buffer.opacity);
        device.cmd_push_constants(
            cmd,
            pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            0,
            push_constants.as_bytes(),
        );

        device.cmd_draw_indexed(cmd, onion_skin_gpu.source_index_count, 1, 0, 0, 0);
    }

    Ok(())
}
