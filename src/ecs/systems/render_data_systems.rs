use cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector3};

use crate::ecs::component::GpuMeshRef;
use crate::ecs::resource::billboard::BillboardData;
use crate::ecs::resource::gizmo::{
    BoneDisplayStyle, BoneGizmoData, ConstraintGizmoData, GridGizmoData, LightGizmoData,
    SpringBoneGizmoData, TransformGizmoData,
};
use crate::ecs::resource::{CurveMeshData, GridMeshData, PointCloudData};
use crate::ecs::RenderData;

pub fn curve_mesh_render_data(curves: &CurveMeshData) -> RenderData {
    let mesh_ref = GpuMeshRef::new(
        curves.mesh.vertex_buffer_handle,
        curves.mesh.index_buffer_handle,
        curves.mesh.indices.len() as u32,
    );
    RenderData::new(mesh_ref, curves.render_info)
}

pub fn point_cloud_render_data(points: &PointCloudData) -> RenderData {
    let mesh_ref = GpuMeshRef::new(
        points.mesh.vertex_buffer_handle,
        points.mesh.index_buffer_handle,
        points.mesh.indices.len() as u32,
    );
    RenderData::new(mesh_ref, points.render_info)
}

pub fn grid_mesh_render_data(grid: &GridMeshData) -> RenderData {
    let index_count = if grid.show_y_axis_grid {
        grid.mesh.indices.len() as u32
    } else {
        grid.xz_only_index_count
    };

    let mesh_ref = GpuMeshRef::new(
        grid.mesh.vertex_buffer_handle,
        grid.mesh.index_buffer_handle,
        index_count,
    );
    RenderData::new(mesh_ref, grid.render_info)
        .with_model_matrix(Matrix4::from_scale(grid.scale.value()))
}

pub fn gizmo_mesh_render_data(gizmo: &GridGizmoData) -> RenderData {
    let mesh_ref = GpuMeshRef::new(
        gizmo.mesh.vertex_buffer_handle,
        gizmo.mesh.index_buffer_handle,
        gizmo.mesh.indices.len() as u32,
    );
    RenderData::new(mesh_ref, gizmo.render_info)
}

pub fn gizmo_selectable_render_data(
    gizmo: &LightGizmoData,
    camera_position: Vector3<f32>,
) -> RenderData {
    let gizmo_pos = gizmo.position.position;
    let distance = (gizmo_pos - camera_position).magnitude();
    let scale_factor = distance * 0.03;
    let model_matrix = Matrix4::from_translation(gizmo_pos) * Matrix4::from_scale(scale_factor);

    let mesh_ref = GpuMeshRef::new(
        gizmo.mesh.vertex_buffer_handle,
        gizmo.mesh.index_buffer_handle,
        gizmo.mesh.indices.len() as u32,
    );
    RenderData::new(mesh_ref, gizmo.render_info).with_model_matrix(model_matrix)
}

pub fn billboard_render_data(billboard: &BillboardData) -> RenderData {
    let model_matrix = billboard
        .transform
        .as_ref()
        .map(|t| t.model_matrix)
        .unwrap_or_else(Matrix4::identity);

    let mesh_ref = GpuMeshRef::new(
        billboard.mesh.vertex_buffer_handle,
        billboard.mesh.index_buffer_handle,
        billboard.mesh.indices.len() as u32,
    );
    RenderData::new(mesh_ref, billboard.render_info).with_model_matrix(model_matrix)
}

pub fn bone_gizmo_render_data(bone_gizmo: &BoneGizmoData) -> Vec<RenderData> {
    match bone_gizmo.display_style {
        BoneDisplayStyle::Stick => {
            let mesh_ref = GpuMeshRef::new(
                bone_gizmo.stick_mesh.vertex_buffer_handle,
                bone_gizmo.stick_mesh.index_buffer_handle,
                bone_gizmo.stick_mesh.indices.len() as u32,
            );
            vec![RenderData::new(mesh_ref, bone_gizmo.stick_render_info)]
        }
        BoneDisplayStyle::Octahedral | BoneDisplayStyle::Box | BoneDisplayStyle::Sphere => {
            let solid_ref = GpuMeshRef::new(
                bone_gizmo.solid_mesh.vertex_buffer_handle,
                bone_gizmo.solid_mesh.index_buffer_handle,
                bone_gizmo.solid_mesh.indices.len() as u32,
            );
            let wire_ref = GpuMeshRef::new(
                bone_gizmo.wire_mesh.vertex_buffer_handle,
                bone_gizmo.wire_mesh.index_buffer_handle,
                bone_gizmo.wire_mesh.indices.len() as u32,
            );
            vec![
                RenderData::new(solid_ref, bone_gizmo.solid_render_info),
                RenderData::new(wire_ref, bone_gizmo.wire_render_info),
            ]
        }
    }
}

pub fn constraint_gizmo_render_data(constraint_gizmo: &ConstraintGizmoData) -> Vec<RenderData> {
    if constraint_gizmo.wire_mesh.indices.is_empty() {
        return Vec::new();
    }

    let mesh_ref = GpuMeshRef::new(
        constraint_gizmo.wire_mesh.vertex_buffer_handle,
        constraint_gizmo.wire_mesh.index_buffer_handle,
        constraint_gizmo.wire_mesh.indices.len() as u32,
    );
    vec![RenderData::new(mesh_ref, constraint_gizmo.wire_render_info)]
}

pub fn spring_bone_gizmo_render_data(gizmo: &SpringBoneGizmoData) -> Vec<RenderData> {
    if gizmo.wire_mesh.indices.is_empty() {
        return Vec::new();
    }

    let mesh_ref = GpuMeshRef::new(
        gizmo.wire_mesh.vertex_buffer_handle,
        gizmo.wire_mesh.index_buffer_handle,
        gizmo.wire_mesh.indices.len() as u32,
    );
    vec![RenderData::new(mesh_ref, gizmo.wire_render_info)]
}

pub fn transform_gizmo_render_data(
    gizmo: &TransformGizmoData,
    camera_position: Vector3<f32>,
    gizmo_scale: f32,
) -> Vec<RenderData> {
    if !gizmo.visible {
        return Vec::new();
    }

    let gizmo_pos = gizmo.position.position;
    let distance = (gizmo_pos - camera_position).magnitude();
    let scale_factor = distance * gizmo_scale;
    let model_matrix = Matrix4::from_translation(gizmo_pos) * Matrix4::from_scale(scale_factor);

    let mut result = Vec::new();

    if !gizmo.line_mesh.indices.is_empty() {
        let line_ref = GpuMeshRef::new(
            gizmo.line_mesh.vertex_buffer_handle,
            gizmo.line_mesh.index_buffer_handle,
            gizmo.line_mesh.indices.len() as u32,
        );
        result.push(
            RenderData::new(line_ref, gizmo.line_render_info).with_model_matrix(model_matrix),
        );
    }

    if !gizmo.solid_mesh.indices.is_empty() {
        let solid_ref = GpuMeshRef::new(
            gizmo.solid_mesh.vertex_buffer_handle,
            gizmo.solid_mesh.index_buffer_handle,
            gizmo.solid_mesh.indices.len() as u32,
        );
        result.push(
            RenderData::new(solid_ref, gizmo.solid_render_info).with_model_matrix(model_matrix),
        );
    }

    result
}

pub fn collect_scene_render_data(
    grid: &GridMeshData,
    gizmo: &GridGizmoData,
    light_gizmo: &LightGizmoData,
    billboard: &BillboardData,
    camera_position: Vector3<f32>,
) -> Vec<RenderData> {
    vec![
        grid_mesh_render_data(grid),
        gizmo_mesh_render_data(gizmo),
        gizmo_selectable_render_data(light_gizmo, camera_position),
        billboard_render_data(billboard),
    ]
}
