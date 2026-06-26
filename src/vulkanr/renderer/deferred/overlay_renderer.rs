use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::ecs::component::RenderInfo;
use crate::ecs::resource::billboard::BillboardData;
use crate::ecs::resource::gizmo::{
    BoneDisplayStyle, BoneGizmoData, ConstraintGizmoData, GridGizmoData, LightGizmoData,
    TransformGizmoData,
};
use crate::ecs::resource::{CurveMeshData, GridMeshData, PointCloudData};
use thyllore_render_core::DynamicMesh;
use thyllore_vulkan_core::renderer::{
    push_fragment_alpha_constant, record_line_mesh_draw, LineMeshDrawOptions,
};
use thyllore_vulkan_core::FrameRenderContext;

pub struct OverlayRenderer<'a> {
    app: &'a App,
}

impl<'a> OverlayRenderer<'a> {
    pub fn new(app: &'a App) -> Self {
        Self { app }
    }

    pub unsafe fn draw_all_overlays(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<()> {
        let ctx = crate::ecs::systems::phases::build_frame_render_context(self.app, image_index);

        self.draw_grid(&ctx, command_buffer)?;
        self.draw_curves(&ctx, command_buffer)?;
        self.draw_point_cloud(&ctx, command_buffer)?;
        self.draw_gizmo(&ctx, command_buffer)?;
        self.draw_transform_gizmo(&ctx, command_buffer)?;
        self.draw_light_lines(&ctx, command_buffer)?;
        self.draw_bone_gizmo(&ctx, command_buffer)?;
        self.draw_constraint_gizmo(&ctx, command_buffer)?;
        self.draw_billboard(&ctx, command_buffer)?;

        Ok(())
    }

    unsafe fn draw_grid(
        &self,
        ctx: &FrameRenderContext<'_>,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        let grid = self.app.resource::<GridMeshData>();
        let index_count = if grid.show_y_axis_grid {
            grid.mesh.indices.len() as u32
        } else {
            grid.xz_only_index_count
        };

        let options = LineMeshDrawOptions {
            line_width: Some(1.0),
            index_count_override: Some(index_count),
            bind_object_set: true,
            frame_set_override: None,
        };
        record_line_mesh_draw(ctx, &grid.mesh, &grid.render_info, &options, command_buffer)?;

        Ok(())
    }

    unsafe fn draw_curves(
        &self,
        ctx: &FrameRenderContext<'_>,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        let Some(curves) = self.app.get_resource::<CurveMeshData>() else {
            return Ok(());
        };
        if !curves.visible || curves.mesh.indices.is_empty() {
            return Ok(());
        }

        let options = LineMeshDrawOptions {
            line_width: Some(1.0),
            index_count_override: None,
            bind_object_set: true,
            frame_set_override: None,
        };
        record_line_mesh_draw(
            ctx,
            &curves.mesh,
            &curves.render_info,
            &options,
            command_buffer,
        )?;

        Ok(())
    }

    unsafe fn draw_point_cloud(
        &self,
        ctx: &FrameRenderContext<'_>,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        let Some(points) = self.app.get_resource::<PointCloudData>() else {
            return Ok(());
        };
        if !points.visible || points.mesh.indices.is_empty() {
            return Ok(());
        }

        let options = LineMeshDrawOptions {
            line_width: None,
            index_count_override: None,
            bind_object_set: true,
            frame_set_override: None,
        };
        record_line_mesh_draw(
            ctx,
            &points.mesh,
            &points.render_info,
            &options,
            command_buffer,
        )?;

        Ok(())
    }

    unsafe fn draw_gizmo(
        &self,
        ctx: &FrameRenderContext<'_>,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        let gizmo = self.app.resource::<GridGizmoData>();
        let options = LineMeshDrawOptions::default();
        record_line_mesh_draw(
            ctx,
            &gizmo.mesh,
            &gizmo.render_info,
            &options,
            command_buffer,
        )?;
        Ok(())
    }

    unsafe fn draw_transform_gizmo(
        &self,
        ctx: &FrameRenderContext<'_>,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        let Some(tg) = self.app.get_resource::<TransformGizmoData>() else {
            return Ok(());
        };

        if !tg.visible {
            return Ok(());
        }

        if !tg.solid_mesh.indices.is_empty() {
            self.record_alpha_mesh(
                ctx,
                &tg.solid_mesh,
                &tg.solid_render_info,
                1.0,
                LineMeshDrawOptions::default(),
                command_buffer,
            )?;
        }

        if !tg.line_mesh.indices.is_empty() {
            self.record_alpha_mesh(
                ctx,
                &tg.line_mesh,
                &tg.line_render_info,
                1.0,
                LineMeshDrawOptions::default(),
                command_buffer,
            )?;
        }

        Ok(())
    }

    unsafe fn draw_light_lines(
        &self,
        ctx: &FrameRenderContext<'_>,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        let grid_mesh = self.app.resource::<GridMeshData>();
        let light_gizmo = self.app.resource::<LightGizmoData>();

        let render_info = RenderInfo::new(
            grid_mesh.render_info.pipeline_id,
            grid_mesh.render_info.object_index,
        );
        let options = LineMeshDrawOptions::default();

        record_line_mesh_draw(
            ctx,
            &light_gizmo.ray_to_model,
            &render_info,
            &options,
            command_buffer,
        )?;
        record_line_mesh_draw(
            ctx,
            &light_gizmo.vertical_lines,
            &render_info,
            &options,
            command_buffer,
        )?;
        Ok(())
    }

    unsafe fn draw_bone_gizmo(
        &self,
        ctx: &FrameRenderContext<'_>,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        let Some(bone_gizmo) = self.app.get_resource::<BoneGizmoData>() else {
            return Ok(());
        };

        if !bone_gizmo.visible {
            return Ok(());
        }

        match bone_gizmo.display_style {
            BoneDisplayStyle::Stick => {
                let options = LineMeshDrawOptions::default();
                record_line_mesh_draw(
                    ctx,
                    &bone_gizmo.stick_mesh,
                    &bone_gizmo.stick_render_info,
                    &options,
                    command_buffer,
                )?;
            }

            BoneDisplayStyle::Octahedral | BoneDisplayStyle::Box | BoneDisplayStyle::Sphere => {
                if bone_gizmo.in_front {
                    self.record_alpha_mesh(
                        ctx,
                        &bone_gizmo.solid_mesh,
                        &bone_gizmo.solid_render_info,
                        1.0,
                        LineMeshDrawOptions::default(),
                        command_buffer,
                    )?;
                    self.record_alpha_mesh(
                        ctx,
                        &bone_gizmo.wire_mesh,
                        &bone_gizmo.wire_render_info,
                        1.0,
                        LineMeshDrawOptions::default(),
                        command_buffer,
                    )?;
                } else {
                    self.record_alpha_mesh(
                        ctx,
                        &bone_gizmo.solid_mesh,
                        &bone_gizmo.solid_depth_render_info,
                        1.0,
                        LineMeshDrawOptions::default(),
                        command_buffer,
                    )?;
                    self.record_alpha_mesh(
                        ctx,
                        &bone_gizmo.wire_mesh,
                        &bone_gizmo.wire_depth_render_info,
                        1.0,
                        LineMeshDrawOptions::default(),
                        command_buffer,
                    )?;

                    self.record_alpha_mesh(
                        ctx,
                        &bone_gizmo.solid_mesh,
                        &bone_gizmo.solid_occluded_render_info,
                        0.25,
                        LineMeshDrawOptions::default(),
                        command_buffer,
                    )?;
                    self.record_alpha_mesh(
                        ctx,
                        &bone_gizmo.wire_mesh,
                        &bone_gizmo.wire_occluded_render_info,
                        0.25,
                        LineMeshDrawOptions::default(),
                        command_buffer,
                    )?;
                }
            }
        }

        Ok(())
    }

    unsafe fn draw_constraint_gizmo(
        &self,
        ctx: &FrameRenderContext<'_>,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        let Some(constraint_gizmo) = self.app.get_resource::<ConstraintGizmoData>() else {
            return Ok(());
        };

        if !constraint_gizmo.visible {
            return Ok(());
        }

        let options = LineMeshDrawOptions::default();
        record_line_mesh_draw(
            ctx,
            &constraint_gizmo.wire_mesh,
            &constraint_gizmo.wire_render_info,
            &options,
            command_buffer,
        )?;
        Ok(())
    }

    unsafe fn draw_billboard(
        &self,
        ctx: &FrameRenderContext<'_>,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        let billboard = self.app.resource::<BillboardData>();
        let descriptor_set = billboard
            .render_state
            .descriptor_set
            .descriptor_sets
            .get(ctx.image_index)
            .copied();
        let Some(descriptor_set) = descriptor_set else {
            return Ok(());
        };

        let options = LineMeshDrawOptions {
            line_width: None,
            index_count_override: None,
            bind_object_set: false,
            frame_set_override: Some(descriptor_set),
        };
        record_line_mesh_draw(
            ctx,
            &billboard.mesh,
            &billboard.render_info,
            &options,
            command_buffer,
        )?;
        Ok(())
    }

    unsafe fn record_alpha_mesh<V>(
        &self,
        ctx: &FrameRenderContext<'_>,
        mesh: &DynamicMesh<V>,
        render_info: &RenderInfo,
        alpha: f32,
        options: LineMeshDrawOptions,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        if mesh.indices.is_empty() {
            return Ok(());
        }
        let Some(pipeline_id) = render_info.pipeline_id else {
            return Ok(());
        };
        let Some(pipeline) = ctx.pipelines.get(pipeline_id) else {
            return Ok(());
        };

        push_fragment_alpha_constant(ctx, pipeline.pipeline_layout, alpha, command_buffer);
        record_line_mesh_draw(ctx, mesh, render_info, &options, command_buffer)?;
        Ok(())
    }
}
