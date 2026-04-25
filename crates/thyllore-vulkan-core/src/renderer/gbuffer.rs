use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::frame_context::FrameRenderContext;
use crate::pipeline::RRPipeline;
use crate::renderer::push_constants::GBufferPushConstants;
use crate::resource::graphics_resource::MeshBuffer;
use crate::resource::RRGBuffer;

pub unsafe fn record_gbuffer_pass(
    ctx: &FrameRenderContext,
    gbuffer: &RRGBuffer,
    pipeline: &RRPipeline,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    draw_mesh_indices: &[usize],
    heatmap_mode: u32,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;

    let render_area = vk::Rect2D::builder()
        .offset(vk::Offset2D::default())
        .extent(vk::Extent2D {
            width: gbuffer.width,
            height: gbuffer.height,
        });

    let clear_values = create_clear_values();

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);

    device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);

    bind_pipeline_and_state(device, pipeline, gbuffer, cmd);

    if !draw_mesh_indices.is_empty() {
        draw_meshes(ctx, pipeline, draw_mesh_indices, heatmap_mode, cmd)?;
    }

    device.cmd_end_render_pass(cmd);

    Ok(())
}

fn create_clear_values() -> [vk::ClearValue; 5] {
    let position_clear = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 0.0],
        },
    };
    let normal_clear = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 0.0],
        },
    };
    let albedo_clear = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 0.0],
        },
    };
    let object_id_clear = vk::ClearValue {
        color: vk::ClearColorValue {
            uint32: [0, 0, 0, 0],
        },
    };
    let depth_clear = vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 0.0,
            stencil: 0,
        },
    };

    [
        position_clear,
        normal_clear,
        albedo_clear,
        object_id_clear,
        depth_clear,
    ]
}

unsafe fn bind_pipeline_and_state(
    device: &Device,
    pipeline: &RRPipeline,
    gbuffer: &RRGBuffer,
    cmd: vk::CommandBuffer,
) {
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(gbuffer.width as f32)
        .height(gbuffer.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(vk::Extent2D {
            width: gbuffer.width,
            height: gbuffer.height,
        });

    device.cmd_set_viewport(cmd, 0, &[viewport]);
    device.cmd_set_scissor(cmd, 0, &[scissor]);
}

unsafe fn draw_meshes(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    mesh_indices: &[usize],
    heatmap_mode: u32,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let graphics = ctx.graphics;
    for &mesh_index in mesh_indices {
        if mesh_index >= graphics.meshes.len() {
            continue;
        }
        let mesh = &graphics.meshes[mesh_index];
        if !mesh.render_to_gbuffer {
            continue;
        }
        draw_single_mesh(ctx, pipeline, mesh, mesh_index, heatmap_mode, cmd)?;
    }
    Ok(())
}

unsafe fn draw_single_mesh(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    mesh: &MeshBuffer,
    mesh_index: usize,
    heatmap_mode: u32,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;
    let graphics = ctx.graphics;
    let image_index = ctx.image_index;

    device.cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertex_buffer.buffer], &[0]);

    device.cmd_bind_index_buffer(cmd, mesh.index_buffer.buffer, 0, vk::IndexType::UINT32);

    let frame_set = graphics.frame_set.sets[image_index];
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.pipeline_layout,
        0,
        &[frame_set],
        &[],
    );

    if let Some(material_id) = graphics.get_material_id(mesh_index) {
        if let Some(material) = graphics.materials.get(material_id) {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline_layout,
                1,
                &[material.descriptor_set],
                &[],
            );
        }
    }

    let object_set_idx = graphics
        .objects
        .get_set_index(image_index, mesh.object_index);
    let object_set = graphics.objects.sets[object_set_idx];
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.pipeline_layout,
        2,
        &[object_set],
        &[],
    );

    let object_id: u32 = (mesh_index + 1) as u32;
    let push_constants = GBufferPushConstants::new(object_id, heatmap_mode);
    device.cmd_push_constants(
        cmd,
        pipeline.pipeline_layout,
        vk::ShaderStageFlags::FRAGMENT,
        0,
        push_constants.as_bytes(),
    );

    device.cmd_draw_indexed(cmd, mesh.index_buffer.indices, 1, 0, 0, 0);

    Ok(())
}
