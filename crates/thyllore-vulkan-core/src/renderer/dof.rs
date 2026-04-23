use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::descriptor::RRDofDescriptorSet;
use crate::frame_context::FrameRenderContext;
use crate::pipeline::RRPipeline;
use crate::resource::DofBuffer;
use thyllore_render_core::{DepthOfField, PhysicalCameraParameters};

#[repr(C)]
#[derive(Clone, Copy)]
struct DofPushConstants {
    focal_length_mm: f32,
    aperture_f_stops: f32,
    sensor_height_mm: f32,
    focus_distance: f32,
    near_plane: f32,
    max_blur_radius: f32,
    viewport_height: f32,
    enabled: i32,
}

pub unsafe fn record_dof_pass(
    ctx: &FrameRenderContext,
    pipeline: &RRPipeline,
    dof_descriptor: &RRDofDescriptorSet,
    dof_buffer: &DofBuffer,
    settings: &DepthOfField,
    camera: &PhysicalCameraParameters,
    camera_near_plane: f32,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    let device = &ctx.device.device;
    let extent = dof_buffer.extent();

    let push_constants = DofPushConstants {
        focal_length_mm: camera.focal_length_mm,
        aperture_f_stops: camera.aperture_f_stops,
        sensor_height_mm: camera.sensor_height_mm,
        focus_distance: settings.focus_distance,
        near_plane: camera_near_plane,
        max_blur_radius: settings.max_blur_radius,
        viewport_height: dof_buffer.height as f32,
        enabled: if settings.enabled { 1 } else { 0 },
    };

    begin_render_pass(device, dof_buffer, cmd, extent);

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

    set_viewport_and_scissor(device, cmd, extent);

    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.pipeline_layout,
        0,
        &[dof_descriptor.descriptor_set],
        &[],
    );

    let push_bytes = std::slice::from_raw_parts(
        &push_constants as *const DofPushConstants as *const u8,
        std::mem::size_of::<DofPushConstants>(),
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

unsafe fn begin_render_pass(
    device: &Device,
    dof_buffer: &DofBuffer,
    cmd: vk::CommandBuffer,
    extent: vk::Extent2D,
) {
    let clear_value = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };

    let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(dof_buffer.render_pass)
        .framebuffer(dof_buffer.framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        })
        .clear_values(std::slice::from_ref(&clear_value));

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
