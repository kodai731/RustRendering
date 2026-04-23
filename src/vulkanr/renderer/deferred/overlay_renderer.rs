use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::ecs::component::{LineMesh, RenderInfo};
use crate::ecs::resource::billboard::BillboardData;
use crate::ecs::resource::gizmo::{
    BoneDisplayStyle, BoneGizmoData, ConstraintGizmoData, GridGizmoData, LightGizmoData,
    TransformGizmoData,
};
use crate::ecs::resource::GridMeshData;
use crate::vulkanr::core::Device;
use crate::vulkanr::pipeline::RRPipeline;
use crate::vulkanr::resource::graphics_resource::GraphicsResources;
use crate::vulkanr::resource::{GpuBufferRegistry, PipelineStorage};

pub struct OverlayRenderer<'a> {
    app: &'a App,
    graphics_resources: &'a GraphicsResources,
    buffer_registry: &'a GpuBufferRegistry,
    device: &'a Device,
}

impl<'a> OverlayRenderer<'a> {
    pub fn new(app: &'a App) -> Self {
        Self {
            app,
            graphics_resources: &app.data.graphics_resources,
            buffer_registry: &app.data.buffer_registry,
            device: &app.rrdevice.device,
        }
    }

    fn pipeline_storage(&self) -> &PipelineStorage {
        self.app.pipeline_storage()
    }

    pub unsafe fn draw_all_overlays(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<()> {
        self.draw_grid(command_buffer, image_index)?;
        self.draw_gizmo(command_buffer, image_index)?;
        self.draw_transform_gizmo(command_buffer, image_index);
        self.draw_light_lines(command_buffer, image_index);
        self.draw_bone_gizmo(command_buffer, image_index);
        self.draw_constraint_gizmo(command_buffer, image_index);
        self.draw_billboard(command_buffer, image_index)?;

        Ok(())
    }

    unsafe fn draw_grid(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<()> {
        let grid = self.app.resource::<GridMeshData>();
        let index_count = if grid.show_y_axis_grid {
            grid.mesh.indices.len() as u32
        } else {
            grid.xz_only_index_count
        };

        let ctx = crate::ecs::systems::phases::build_frame_render_context(self.app, image_index);
        let options = thyllore_vulkan_core::renderer::LineMeshDrawOptions {
            line_width: 1.0,
            index_count_override: Some(index_count),
            bind_object_set: true,
        };
        thyllore_vulkan_core::renderer::record_line_mesh_draw(
            &ctx,
            &grid.mesh,
            &grid.render_info,
            &options,
            command_buffer,
        )?;

        Ok(())
    }

    unsafe fn draw_gizmo(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<()> {
        let gizmo = self.app.resource::<GridGizmoData>();
        let pipeline_storage = self.pipeline_storage();

        let vertex_buffer = match self
            .buffer_registry
            .get_vertex_buffer(gizmo.mesh.vertex_buffer_handle)
        {
            Some(b) => b,
            None => return Ok(()),
        };
        let index_buffer = match self
            .buffer_registry
            .get_index_buffer(gizmo.mesh.index_buffer_handle)
        {
            Some(b) => b,
            None => return Ok(()),
        };

        let pipeline_id = match gizmo.render_info.pipeline_id {
            Some(id) => id,
            None => return Ok(()),
        };
        let pipeline = match pipeline_storage.get(pipeline_id) {
            Some(p) => p,
            None => return Ok(()),
        };

        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline,
        );

        self.device.cmd_set_line_width(command_buffer, 1.0);

        self.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);

        self.device
            .cmd_bind_index_buffer(command_buffer, index_buffer, 0, vk::IndexType::UINT32);

        let frame_set = self.graphics_resources.frame_set.sets[image_index];
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            0,
            &[frame_set],
            &[],
        );

        let object_set_idx = self
            .graphics_resources
            .objects
            .get_set_index(image_index, gizmo.render_info.object_index);
        let object_set = self.graphics_resources.objects.sets[object_set_idx];
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            2,
            &[object_set],
            &[],
        );

        self.device
            .cmd_draw_indexed(command_buffer, gizmo.mesh.indices.len() as u32, 1, 0, 0, 0);

        Ok(())
    }

    unsafe fn draw_transform_gizmo(&self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let tg = match self.app.get_resource::<TransformGizmoData>() {
            Some(tg) => tg,
            None => return,
        };

        if !tg.visible {
            return;
        }

        let pipeline_storage = self.pipeline_storage();

        if !tg.solid_mesh.indices.is_empty() {
            if let Some(pid) = tg.solid_render_info.pipeline_id {
                if let Some(pipeline) = pipeline_storage.get(pid) {
                    self.push_bone_alpha(command_buffer, pipeline, 1.0);
                    self.draw_triangle_mesh(
                        &tg.solid_mesh,
                        pipeline,
                        tg.solid_render_info.object_index,
                        command_buffer,
                        image_index,
                    );
                }
            }
        }

        if !tg.line_mesh.indices.is_empty() {
            if let Some(pid) = tg.line_render_info.pipeline_id {
                if let Some(pipeline) = pipeline_storage.get(pid) {
                    self.push_bone_alpha(command_buffer, pipeline, 1.0);
                    self.draw_line_mesh(
                        &tg.line_mesh,
                        pipeline,
                        tg.line_render_info.object_index,
                        command_buffer,
                        image_index,
                    );
                }
            }
        }
    }

    unsafe fn draw_light_lines(&self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let grid_mesh = self.app.resource::<GridMeshData>();
        let light_gizmo = self.app.resource::<LightGizmoData>();
        let pipeline_storage = self.pipeline_storage();

        let Some(pipeline_id) = grid_mesh.render_info.pipeline_id else {
            return;
        };
        let Some(pipeline) = pipeline_storage.get(pipeline_id) else {
            return;
        };

        self.draw_line_mesh(
            &light_gizmo.ray_to_model,
            pipeline,
            grid_mesh.render_info.object_index,
            command_buffer,
            image_index,
        );
        self.draw_line_mesh(
            &light_gizmo.vertical_lines,
            pipeline,
            grid_mesh.render_info.object_index,
            command_buffer,
            image_index,
        );
    }

    unsafe fn draw_bone_gizmo(&self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let bone_gizmo = match self.app.get_resource::<BoneGizmoData>() {
            Some(bg) => bg,
            None => return,
        };

        if !bone_gizmo.visible {
            return;
        }

        match bone_gizmo.display_style {
            BoneDisplayStyle::Stick => {
                if bone_gizmo.stick_mesh.indices.is_empty() {
                    return;
                }
                let pipeline_id = match bone_gizmo.stick_render_info.pipeline_id {
                    Some(id) => id,
                    None => return,
                };
                let pipeline = match self.pipeline_storage().get(pipeline_id) {
                    Some(p) => p,
                    None => return,
                };
                self.draw_line_mesh(
                    &bone_gizmo.stick_mesh,
                    pipeline,
                    bone_gizmo.stick_render_info.object_index,
                    command_buffer,
                    image_index,
                );
            }

            BoneDisplayStyle::Octahedral | BoneDisplayStyle::Box | BoneDisplayStyle::Sphere => {
                if bone_gizmo.in_front {
                    self.draw_bone_solid_pass(
                        &bone_gizmo.solid_mesh,
                        &bone_gizmo.solid_render_info,
                        1.0,
                        command_buffer,
                        image_index,
                    );
                    self.draw_bone_wire_pass(
                        &bone_gizmo.wire_mesh,
                        &bone_gizmo.wire_render_info,
                        1.0,
                        command_buffer,
                        image_index,
                    );
                } else {
                    self.draw_bone_solid_pass(
                        &bone_gizmo.solid_mesh,
                        &bone_gizmo.solid_depth_render_info,
                        1.0,
                        command_buffer,
                        image_index,
                    );
                    self.draw_bone_wire_pass(
                        &bone_gizmo.wire_mesh,
                        &bone_gizmo.wire_depth_render_info,
                        1.0,
                        command_buffer,
                        image_index,
                    );

                    self.draw_bone_solid_pass(
                        &bone_gizmo.solid_mesh,
                        &bone_gizmo.solid_occluded_render_info,
                        0.25,
                        command_buffer,
                        image_index,
                    );
                    self.draw_bone_wire_pass(
                        &bone_gizmo.wire_mesh,
                        &bone_gizmo.wire_occluded_render_info,
                        0.25,
                        command_buffer,
                        image_index,
                    );
                }
            }
        }
    }

    unsafe fn draw_constraint_gizmo(&self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let constraint_gizmo = match self.app.get_resource::<ConstraintGizmoData>() {
            Some(cg) => cg,
            None => return,
        };

        if !constraint_gizmo.visible {
            return;
        }
        if constraint_gizmo.wire_mesh.indices.is_empty() {
            return;
        }

        let pipeline_id = match constraint_gizmo.wire_render_info.pipeline_id {
            Some(id) => id,
            None => return,
        };
        let pipeline = match self.pipeline_storage().get(pipeline_id) {
            Some(p) => p,
            None => return,
        };

        self.draw_line_mesh(
            &constraint_gizmo.wire_mesh,
            pipeline,
            constraint_gizmo.wire_render_info.object_index,
            command_buffer,
            image_index,
        );
    }

    unsafe fn draw_billboard(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<()> {
        let billboard = self.app.resource::<BillboardData>();
        let pipeline_storage = self.pipeline_storage();

        let vertex_buffer = match self
            .buffer_registry
            .get_vertex_buffer(billboard.mesh.vertex_buffer_handle)
        {
            Some(b) => b,
            None => return Ok(()),
        };
        let index_buffer = match self
            .buffer_registry
            .get_index_buffer(billboard.mesh.index_buffer_handle)
        {
            Some(b) => b,
            None => return Ok(()),
        };

        let pipeline_id = match billboard.render_info.pipeline_id {
            Some(id) => id,
            None => return Ok(()),
        };
        let pipeline = match pipeline_storage.get(pipeline_id) {
            Some(p) => p,
            None => return Ok(()),
        };

        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline,
        );

        self.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);

        self.device
            .cmd_bind_index_buffer(command_buffer, index_buffer, 0, vk::IndexType::UINT32);

        let descriptor_set_index = image_index;

        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            0,
            &[billboard.render_state.descriptor_set.descriptor_sets[descriptor_set_index]],
            &[],
        );

        self.device.cmd_draw_indexed(
            command_buffer,
            billboard.mesh.indices.len() as u32,
            1,
            0,
            0,
            0,
        );

        Ok(())
    }

    unsafe fn draw_line_mesh(
        &self,
        mesh: &LineMesh,
        pipeline: &RRPipeline,
        object_index: usize,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) {
        if mesh.indices.is_empty() {
            return;
        }

        let vertex_buffer = match self
            .buffer_registry
            .get_vertex_buffer(mesh.vertex_buffer_handle)
        {
            Some(vb) => vb,
            None => return,
        };
        let index_buffer = match self
            .buffer_registry
            .get_index_buffer(mesh.index_buffer_handle)
        {
            Some(ib) => ib,
            None => return,
        };

        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline,
        );

        self.device.cmd_set_line_width(command_buffer, 1.0);
        self.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);
        self.device
            .cmd_bind_index_buffer(command_buffer, index_buffer, 0, vk::IndexType::UINT32);

        let frame_set = self.graphics_resources.frame_set.sets[image_index];
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            0,
            &[frame_set],
            &[],
        );

        let object_set_idx = self
            .graphics_resources
            .objects
            .get_set_index(image_index, object_index);
        let object_set = self.graphics_resources.objects.sets[object_set_idx];
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            2,
            &[object_set],
            &[],
        );

        self.device
            .cmd_draw_indexed(command_buffer, mesh.indices.len() as u32, 1, 0, 0, 0);
    }

    unsafe fn draw_triangle_mesh(
        &self,
        mesh: &LineMesh,
        pipeline: &RRPipeline,
        object_index: usize,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) {
        if mesh.indices.is_empty() {
            return;
        }

        let vertex_buffer = match self
            .buffer_registry
            .get_vertex_buffer(mesh.vertex_buffer_handle)
        {
            Some(vb) => vb,
            None => return,
        };
        let index_buffer = match self
            .buffer_registry
            .get_index_buffer(mesh.index_buffer_handle)
        {
            Some(ib) => ib,
            None => return,
        };

        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline,
        );

        self.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);

        self.device
            .cmd_bind_index_buffer(command_buffer, index_buffer, 0, vk::IndexType::UINT32);

        let frame_set = self.graphics_resources.frame_set.sets[image_index];
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            0,
            &[frame_set],
            &[],
        );

        let object_set_idx = self
            .graphics_resources
            .objects
            .get_set_index(image_index, object_index);
        let object_set = self.graphics_resources.objects.sets[object_set_idx];
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            2,
            &[object_set],
            &[],
        );

        self.device
            .cmd_draw_indexed(command_buffer, mesh.indices.len() as u32, 1, 0, 0, 0);
    }

    unsafe fn push_bone_alpha(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline: &RRPipeline,
        alpha: f32,
    ) {
        let alpha_bytes = std::slice::from_raw_parts(
            &alpha as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
        );
        self.device.cmd_push_constants(
            command_buffer,
            pipeline.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            0,
            alpha_bytes,
        );
    }

    unsafe fn draw_bone_solid_pass(
        &self,
        mesh: &LineMesh,
        render_info: &RenderInfo,
        alpha: f32,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) {
        if mesh.indices.is_empty() {
            return;
        }
        let Some(pid) = render_info.pipeline_id else {
            return;
        };
        let Some(pipeline) = self.pipeline_storage().get(pid) else {
            return;
        };
        self.push_bone_alpha(command_buffer, pipeline, alpha);
        self.draw_triangle_mesh(
            mesh,
            pipeline,
            render_info.object_index,
            command_buffer,
            image_index,
        );
    }

    unsafe fn draw_bone_wire_pass(
        &self,
        mesh: &LineMesh,
        render_info: &RenderInfo,
        alpha: f32,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) {
        if mesh.indices.is_empty() {
            return;
        }
        let Some(pid) = render_info.pipeline_id else {
            return;
        };
        let Some(pipeline) = self.pipeline_storage().get(pid) else {
            return;
        };
        self.push_bone_alpha(command_buffer, pipeline, alpha);
        self.draw_line_mesh(
            mesh,
            pipeline,
            render_info.object_index,
            command_buffer,
            image_index,
        );
    }
}
