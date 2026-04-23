use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::frame_context::FrameRenderContext;
use crate::pipeline::RRPipeline;
use crate::resource::GpuBufferRegistry;
use thyllore_render_core::{LineMesh, RenderInfo};

pub struct LineMeshDrawOptions {
    pub line_width: f32,
    pub index_count_override: Option<u32>,
    pub bind_object_set: bool,
}

impl Default for LineMeshDrawOptions {
    fn default() -> Self {
        Self {
            line_width: 1.0,
            index_count_override: None,
            bind_object_set: true,
        }
    }
}

pub unsafe fn record_line_mesh_draw(
    ctx: &FrameRenderContext,
    mesh: &LineMesh,
    render_info: &RenderInfo,
    options: &LineMeshDrawOptions,
    cmd: vk::CommandBuffer,
) -> Result<bool> {
    let buffers: &GpuBufferRegistry = ctx.buffers;
    let Some(vertex_buffer) = buffers.get_vertex_buffer(mesh.vertex_buffer_handle) else {
        return Ok(false);
    };
    let Some(index_buffer) = buffers.get_index_buffer(mesh.index_buffer_handle) else {
        return Ok(false);
    };
    let Some(pipeline_id) = render_info.pipeline_id else {
        return Ok(false);
    };
    let Some(pipeline) = ctx.pipelines.get(pipeline_id) else {
        return Ok(false);
    };

    bind_line_pipeline(ctx, pipeline, options.line_width, cmd);
    bind_line_mesh_buffers(
        ctx,
        pipeline,
        render_info,
        vertex_buffer,
        index_buffer,
        options,
        cmd,
    );

    let index_count = options
        .index_count_override
        .unwrap_or_else(|| mesh.indices.len() as u32);

    let device = &ctx.device.device;
    device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);

    Ok(true)
}

unsafe fn bind_line_pipeline(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    line_width: f32,
    cmd: vk::CommandBuffer,
) {
    let device = &ctx.device.device;
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
    device.cmd_set_line_width(cmd, line_width);
}

unsafe fn bind_line_mesh_buffers(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    render_info: &RenderInfo,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    options: &LineMeshDrawOptions,
    cmd: vk::CommandBuffer,
) {
    let device = &ctx.device.device;
    let image_index = ctx.image_index;
    let graphics = ctx.graphics;

    device.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0]);
    device.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);

    let frame_set = graphics.frame_set.sets[image_index];
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.pipeline_layout,
        0,
        &[frame_set],
        &[],
    );

    if options.bind_object_set {
        let object_set_idx = graphics
            .objects
            .get_set_index(image_index, render_info.object_index);
        let object_set = graphics.objects.sets[object_set_idx];
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            2,
            &[object_set],
            &[],
        );
    }
}
