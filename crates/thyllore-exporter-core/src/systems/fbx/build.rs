use std::path::Path;

use thyllore_anim_core::editable::EditableAnimationClip;
use thyllore_anim_core::Skeleton;
use thyllore_importer_core::fbx::fbx::FbxModel;

use super::connections::{
    generate_animation_connections, generate_bone_connections, generate_mesh_connections,
};
use super::curves::build_animation_curves;
use super::geometry::build_geometry_exports;
use super::mesh_material::{
    build_material_exports, build_mesh_model_exports, build_texture_exports,
};
use super::skin::build_skin_exports;
use crate::components::fbx::*;
use crate::fbx_animation::{
    build_bone_export_list, seconds_to_ktime, FbxBoneExport, FbxConnection, FbxCurveNodeExport,
    FbxExportData, UidAllocator,
};
pub(crate) fn build_full_export_data(
    fbx_model: &FbxModel,
    clip: Option<&EditableAnimationClip>,
    skeleton: &Skeleton,
    export_path: &Path,
) -> anyhow::Result<FullFbxExportData> {
    let inv_unit_scale = 1.0_f32 / fbx_model.unit_scale;
    let needs_coord_conversion = fbx_model.fbx_data.iter().any(|d| !d.clusters.is_empty());

    let mesh_node_names: std::collections::HashSet<String> = fbx_model
        .fbx_data
        .iter()
        .filter_map(|d| d.mesh_node_name.clone())
        .collect();

    let mut uid_alloc = UidAllocator::new();
    let bones = build_bone_export_list(
        skeleton,
        &mut uid_alloc,
        &mesh_node_names,
        inv_unit_scale,
        needs_coord_conversion,
    );

    let stack_uid = uid_alloc.allocate();
    let layer_uid = uid_alloc.allocate();
    let document_uid = uid_alloc.allocate();

    let (clip_name, duration_ktime) = resolve_clip_metadata(clip, fbx_model);

    let mut name_to_model_uid: std::collections::HashMap<String, i64> = bones
        .iter()
        .map(|b| (b.name.clone(), b.model_uid))
        .collect();

    let (geometries, mesh_models, materials, textures, skins) = build_mesh_assets(
        fbx_model,
        &mut name_to_model_uid,
        &mut uid_alloc,
        inv_unit_scale,
        export_path,
    );

    let (curve_nodes, curves) =
        build_animation_curves(clip, &name_to_model_uid, &mut uid_alloc, inv_unit_scale);

    let connections = build_all_connections(
        &bones,
        &mesh_models,
        &geometries,
        &materials,
        &textures,
        &skins,
        stack_uid,
        layer_uid,
        &curve_nodes,
    );

    let anim_data = FbxExportData {
        clip_name,
        duration_ktime,
        needs_coord_conversion,
        axes: fbx_model.axes.clone(),
        fps: fbx_model.fps,
        bones,
        stack_uid,
        layer_uid,
        document_uid,
        curve_nodes,
        curves,
        connections,
    };

    Ok(FullFbxExportData {
        anim_data,
        geometries,
        mesh_models,
        materials,
        textures,
        skins,
        unit_scale: fbx_model.unit_scale,
    })
}

pub(crate) fn resolve_clip_metadata(
    clip: Option<&EditableAnimationClip>,
    fbx_model: &FbxModel,
) -> (String, i64) {
    let duration_ktime = clip
        .map(|c| seconds_to_ktime(c.duration))
        .unwrap_or_else(|| {
            fbx_model
                .animations
                .first()
                .map(|a| seconds_to_ktime(a.duration))
                .unwrap_or(0)
        });

    let clip_name = clip
        .map(|c| c.name.clone())
        .unwrap_or_else(|| "DefaultAnimation".to_string());

    (clip_name, duration_ktime)
}

pub(crate) fn build_mesh_assets(
    fbx_model: &FbxModel,
    name_to_model_uid: &mut std::collections::HashMap<String, i64>,
    uid_alloc: &mut UidAllocator,
    inv_unit_scale: f32,
    export_path: &Path,
) -> (
    Vec<FbxGeometryExport>,
    Vec<FbxMeshModelExport>,
    Vec<FbxMaterialExport>,
    Vec<FbxTextureExport>,
    Vec<FbxSkinExport>,
) {
    let geometries = build_geometry_exports(&fbx_model.fbx_data, uid_alloc, inv_unit_scale);

    let mesh_models = build_mesh_model_exports(
        &fbx_model.fbx_data,
        &geometries,
        name_to_model_uid,
        &fbx_model.nodes,
        uid_alloc,
        inv_unit_scale,
    );

    for mesh_model in &mesh_models {
        name_to_model_uid.insert(mesh_model.name.clone(), mesh_model.uid);
    }

    let materials = build_material_exports(&fbx_model.fbx_data, &mesh_models, uid_alloc);
    let export_dir = export_path.parent().unwrap_or_else(|| Path::new("."));
    let textures = build_texture_exports(
        &fbx_model.fbx_data,
        &materials,
        uid_alloc,
        export_dir,
        fbx_model.source_path.as_deref(),
    );

    let skins = build_skin_exports(
        &fbx_model.fbx_data,
        &geometries,
        name_to_model_uid,
        uid_alloc,
        inv_unit_scale,
    );

    (geometries, mesh_models, materials, textures, skins)
}

pub(crate) fn build_all_connections(
    bones: &[FbxBoneExport],
    mesh_models: &[FbxMeshModelExport],
    geometries: &[FbxGeometryExport],
    materials: &[FbxMaterialExport],
    textures: &[FbxTextureExport],
    skins: &[FbxSkinExport],
    stack_uid: i64,
    layer_uid: i64,
    curve_nodes: &[FbxCurveNodeExport],
) -> Vec<FbxConnection> {
    let mut connections = Vec::new();
    generate_bone_connections(bones, &mut connections);
    generate_mesh_connections(
        mesh_models,
        geometries,
        materials,
        textures,
        skins,
        &mut connections,
    );
    generate_animation_connections(stack_uid, layer_uid, curve_nodes, &mut connections);
    connections
}
