use std::io::{Seek, Write};
use std::path::{Path, PathBuf};

use cgmath::Matrix4;
use fbxcel::low::FbxVersion;
use fbxcel::writer::v7400::binary::{FbxFooter, Writer};

use thyllore_anim_core::editable::EditableAnimationClip;
use thyllore_anim_core::Skeleton;
use thyllore_importer_core::fbx::fbx::{FbxData, FbxModel};

use super::fbx_animation::{
    build_bone_export_list, build_channel_exports, decompose_matrix_to_trs, seconds_to_ktime,
    write_anim_curve, write_anim_curve_node, write_anim_layer, write_anim_stack, write_bone_model,
    write_connections, write_documents, write_global_settings, write_header_extension,
    write_node_attribute, write_object_type, write_property_f64, write_property_f64x3,
    write_property_i32, write_references, FbxBoneExport, FbxChannel, FbxConnection, FbxCurveExport,
    FbxCurveNodeExport, FbxExportData, FbxWriteResult, UidAllocator,
};

struct FbxGeometryExport {
    uid: i64,
    mesh_model_uid: i64,
    positions: Vec<f64>,
    polygon_vertex_index: Vec<i32>,
    normals: Vec<f64>,
    uv_values: Vec<f64>,
}

struct FbxMeshModelExport {
    uid: i64,
    name: String,
    parent_bone_uid: Option<i64>,
    translation: [f64; 3],
    rotation: [f64; 3],
    scaling: [f64; 3],
}

struct FbxMaterialExport {
    uid: i64,
    name: String,
    mesh_model_uid: i64,
    diffuse_color: [f64; 3],
}

struct FbxTextureExport {
    texture_uid: i64,
    video_uid: i64,
    material_uid: i64,
    filename: String,
    relative_filename: String,
}

struct FbxSkinExport {
    skin_uid: i64,
    geometry_uid: i64,
    clusters: Vec<FbxClusterExport>,
}

struct FbxClusterExport {
    uid: i64,
    bone_model_uid: i64,
    indices: Vec<i32>,
    weights: Vec<f64>,
    transform: [f64; 16],
    transform_link: [f64; 16],
}

struct FullFbxExportData {
    anim_data: FbxExportData,
    geometries: Vec<FbxGeometryExport>,
    mesh_models: Vec<FbxMeshModelExport>,
    materials: Vec<FbxMaterialExport>,
    textures: Vec<FbxTextureExport>,
    skins: Vec<FbxSkinExport>,
    unit_scale: f32,
}

pub fn export_full_fbx(
    fbx_model: &FbxModel,
    clip: Option<&EditableAnimationClip>,
    skeleton: &Skeleton,
    path: &Path,
) -> anyhow::Result<()> {
    let export_data = build_full_export_data(fbx_model, clip, skeleton, path)?;

    let file = std::fs::File::create(path)?;
    let writer = Writer::new(file, FbxVersion::V7_4)
        .map_err(|e| anyhow::anyhow!("FBX writer init failed: {}", e))?;

    write_full_fbx_binary(writer, &export_data)
        .map_err(|e| anyhow::anyhow!("FBX write failed: {}", e))?;

    Ok(())
}

fn build_full_export_data(
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

fn resolve_clip_metadata(
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

fn build_mesh_assets(
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

fn build_all_connections(
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

fn build_animation_curves(
    clip: Option<&EditableAnimationClip>,
    bone_name_to_model_uid: &std::collections::HashMap<String, i64>,
    uid_alloc: &mut UidAllocator,
    inv_unit_scale: f32,
) -> (Vec<FbxCurveNodeExport>, Vec<FbxCurveExport>) {
    let mut curve_nodes = Vec::new();
    let mut curves = Vec::new();

    let Some(clip) = clip else {
        return (curve_nodes, curves);
    };

    for track in clip.tracks.values() {
        let bone_model_uid = match bone_name_to_model_uid.get(track.bone_name.as_str()) {
            Some(&uid) => uid,
            None => continue,
        };

        if let Some((node, node_curves)) = build_channel_exports(
            [
                &track.translation_x,
                &track.translation_y,
                &track.translation_z,
            ],
            bone_model_uid,
            FbxChannel::Translation,
            uid_alloc,
            inv_unit_scale,
        ) {
            curve_nodes.push(node);
            curves.extend(node_curves);
        }

        if let Some((node, node_curves)) = build_channel_exports(
            [&track.rotation_x, &track.rotation_y, &track.rotation_z],
            bone_model_uid,
            FbxChannel::Rotation,
            uid_alloc,
            inv_unit_scale,
        ) {
            curve_nodes.push(node);
            curves.extend(node_curves);
        }

        if let Some((node, node_curves)) = build_channel_exports(
            [&track.scale_x, &track.scale_y, &track.scale_z],
            bone_model_uid,
            FbxChannel::Scale,
            uid_alloc,
            inv_unit_scale,
        ) {
            curve_nodes.push(node);
            curves.extend(node_curves);
        }
    }

    (curve_nodes, curves)
}

fn build_geometry_exports(
    fbx_data_list: &[FbxData],
    uid_alloc: &mut UidAllocator,
    inv_unit_scale: f32,
) -> Vec<FbxGeometryExport> {
    let mut geometries = Vec::new();

    for fbx_data in fbx_data_list {
        let geometry_uid = uid_alloc.allocate();
        let mesh_model_uid = uid_alloc.allocate();

        let positions = convert_positions_to_fbx(fbx_data, inv_unit_scale);
        let polygon_vertex_index = encode_triangle_polygon_indices(&fbx_data.indices);
        let normals = convert_normals_to_fbx(fbx_data);
        let uv_values = convert_uvs_to_fbx(fbx_data);

        geometries.push(FbxGeometryExport {
            uid: geometry_uid,
            mesh_model_uid,
            positions,
            polygon_vertex_index,
            normals,
            uv_values,
        });
    }

    geometries
}

fn convert_positions_to_fbx(fbx_data: &FbxData, inv_unit_scale: f32) -> Vec<f64> {
    let source = if !fbx_data.local_positions.is_empty() {
        &fbx_data.local_positions
    } else {
        &fbx_data.positions
    };

    source
        .iter()
        .flat_map(|p| {
            [
                (p.x * inv_unit_scale) as f64,
                (p.y * inv_unit_scale) as f64,
                (p.z * inv_unit_scale) as f64,
            ]
        })
        .collect()
}

fn convert_normals_to_fbx(fbx_data: &FbxData) -> Vec<f64> {
    let source = if !fbx_data.local_normals.is_empty() {
        &fbx_data.local_normals
    } else {
        &fbx_data.normals
    };

    source
        .iter()
        .flat_map(|n| [n.x as f64, n.y as f64, n.z as f64])
        .collect()
}

fn convert_uvs_to_fbx(fbx_data: &FbxData) -> Vec<f64> {
    fbx_data
        .tex_coords
        .iter()
        .flat_map(|uv| [uv[0] as f64, (1.0 - uv[1]) as f64])
        .collect()
}

fn encode_triangle_polygon_indices(indices: &[u32]) -> Vec<i32> {
    indices
        .chunks(3)
        .flat_map(|tri| {
            if tri.len() == 3 {
                vec![tri[0] as i32, tri[1] as i32, -(tri[2] as i32 + 1)]
            } else {
                tri.iter().map(|&i| i as i32).collect()
            }
        })
        .collect()
}

fn build_mesh_model_exports(
    fbx_data_list: &[FbxData],
    geometries: &[FbxGeometryExport],
    bone_name_to_model_uid: &std::collections::HashMap<String, i64>,
    nodes: &std::collections::HashMap<String, thyllore_importer_core::fbx::fbx::BoneNode>,
    uid_alloc: &mut UidAllocator,
    inv_unit_scale: f32,
) -> Vec<FbxMeshModelExport> {
    let mut mesh_models = Vec::new();
    let mut mesh_name_to_uid: std::collections::HashMap<String, i64> =
        std::collections::HashMap::new();
    let scale = inv_unit_scale as f64;

    for (i, fbx_data) in fbx_data_list.iter().enumerate() {
        let uid = if i < geometries.len() {
            geometries[i].mesh_model_uid
        } else {
            uid_alloc.allocate()
        };

        let mesh_name = fbx_data
            .mesh_node_name
            .clone()
            .unwrap_or_else(|| format!("MeshModel_{}", i));

        mesh_name_to_uid.insert(mesh_name.clone(), uid);

        let (mut translation, rotation, scaling) = fbx_data
            .mesh_node_name
            .as_ref()
            .and_then(|name| nodes.get(name))
            .map(|node| decompose_matrix_to_trs(&node.local_transform))
            .unwrap_or(([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));

        translation[0] *= scale;
        translation[1] *= scale;
        translation[2] *= scale;

        mesh_models.push(FbxMeshModelExport {
            uid,
            name: mesh_name,
            parent_bone_uid: None,
            translation,
            rotation,
            scaling,
        });
    }

    resolve_mesh_parent_uids(
        &mut mesh_models,
        fbx_data_list,
        nodes,
        bone_name_to_model_uid,
        &mesh_name_to_uid,
    );

    mesh_models
}

fn resolve_mesh_parent_uids(
    mesh_models: &mut [FbxMeshModelExport],
    fbx_data_list: &[FbxData],
    nodes: &std::collections::HashMap<String, thyllore_importer_core::fbx::fbx::BoneNode>,
    bone_name_to_model_uid: &std::collections::HashMap<String, i64>,
    mesh_name_to_uid: &std::collections::HashMap<String, i64>,
) {
    for (i, fbx_data) in fbx_data_list.iter().enumerate() {
        let parent_uid = fbx_data
            .mesh_node_name
            .as_ref()
            .and_then(|name| nodes.get(name))
            .and_then(|node| node.parent.as_ref())
            .and_then(|parent| {
                bone_name_to_model_uid
                    .get(parent.as_str())
                    .or_else(|| mesh_name_to_uid.get(parent.as_str()))
                    .copied()
            });

        if i < mesh_models.len() {
            mesh_models[i].parent_bone_uid = parent_uid;
        }
    }
}

fn build_material_exports(
    fbx_data_list: &[FbxData],
    mesh_models: &[FbxMeshModelExport],
    uid_alloc: &mut UidAllocator,
) -> Vec<FbxMaterialExport> {
    let mut materials = Vec::new();

    for (i, fbx_data) in fbx_data_list.iter().enumerate() {
        let mat_uid = uid_alloc.allocate();
        let mat_name = fbx_data
            .material_name
            .clone()
            .unwrap_or_else(|| format!("Material_{}", i));

        let mesh_model_uid = if i < mesh_models.len() {
            mesh_models[i].uid
        } else {
            0
        };

        let dc = fbx_data.diffuse_color;
        materials.push(FbxMaterialExport {
            uid: mat_uid,
            name: mat_name,
            mesh_model_uid,
            diffuse_color: [dc[0] as f64, dc[1] as f64, dc[2] as f64],
        });
    }

    materials
}

fn compute_relative_path(from_dir: &Path, to_path: &Path) -> String {
    let from_abs = std::env::current_dir()
        .map(|cwd| cwd.join(from_dir))
        .unwrap_or_else(|_| from_dir.to_path_buf());
    let to_abs = std::env::current_dir()
        .map(|cwd| cwd.join(to_path))
        .unwrap_or_else(|_| to_path.to_path_buf());

    let from_components: Vec<_> = from_abs.components().collect();
    let to_components: Vec<_> = to_abs.components().collect();

    let common_len = from_components
        .iter()
        .zip(to_components.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let up_count = from_components.len() - common_len;
    let mut result = PathBuf::new();
    for _ in 0..up_count {
        result.push("..");
    }
    for comp in &to_components[common_len..] {
        result.push(comp);
    }

    result.to_string_lossy().replace('\\', "/")
}

fn canonicalize_clean(path: &Path) -> PathBuf {
    let abs_path = if path.is_relative() {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    } else {
        path.to_path_buf()
    };

    match abs_path.canonicalize() {
        Ok(canonical) => {
            let s = canonical.to_string_lossy();
            if let Some(stripped) = s.strip_prefix(r"\\?\") {
                PathBuf::from(stripped)
            } else {
                canonical
            }
        }
        Err(_) => abs_path,
    }
}

fn resolve_texture_for_export(texture_path: &str, model_path: Option<&str>) -> PathBuf {
    let original = Path::new(texture_path);
    if original.exists() {
        return original.to_path_buf();
    }

    let Some(model_path) = model_path else {
        return original.to_path_buf();
    };

    let file_stem = original.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let file_name = original.file_name().and_then(|s| s.to_str()).unwrap_or("");

    let model_dir = Path::new(model_path)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let model_root = model_dir.parent().unwrap_or(model_dir);

    let texture_dir = original.parent().unwrap_or_else(|| Path::new("."));
    let texture_root = texture_dir.parent().unwrap_or(texture_dir);

    let mut search_dirs = vec![
        model_dir.to_path_buf(),
        model_dir.join("textures"),
        model_root.join("textures"),
    ];

    if texture_dir != model_dir {
        search_dirs.push(texture_dir.to_path_buf());
        search_dirs.push(texture_dir.join("textures"));
        search_dirs.push(texture_root.join("textures"));
    }

    let candidate_names: Vec<String> = vec![
        file_name.to_string(),
        format!("{}.png", file_name),
        format!("{}.png", file_stem),
        format!("{}.jpg", file_stem),
    ];

    for dir in &search_dirs {
        for name in &candidate_names {
            let candidate = dir.join(name);
            if candidate.exists() {
                return candidate;
            }
        }
    }

    original.to_path_buf()
}

fn build_texture_exports(
    fbx_data_list: &[FbxData],
    materials: &[FbxMaterialExport],
    uid_alloc: &mut UidAllocator,
    export_dir: &Path,
    model_source_path: Option<&str>,
) -> Vec<FbxTextureExport> {
    let mut textures = Vec::new();

    for (i, fbx_data) in fbx_data_list.iter().enumerate() {
        if let Some(ref tex_path) = fbx_data.diffuse_texture {
            let texture_uid = uid_alloc.allocate();
            let video_uid = uid_alloc.allocate();

            let material_uid = if i < materials.len() {
                materials[i].uid
            } else {
                0
            };

            let resolved = resolve_texture_for_export(tex_path, model_source_path);
            let resolved_abs = canonicalize_clean(&resolved);
            let resolved_str = resolved_abs.to_string_lossy().to_string();
            let relative_filename = compute_relative_path(export_dir, &resolved_abs);

            textures.push(FbxTextureExport {
                texture_uid,
                video_uid,
                material_uid,
                filename: resolved_str,
                relative_filename,
            });
        }
    }

    textures
}

fn build_skin_exports(
    fbx_data_list: &[FbxData],
    geometries: &[FbxGeometryExport],
    bone_name_to_model_uid: &std::collections::HashMap<String, i64>,
    uid_alloc: &mut UidAllocator,
    inv_unit_scale: f32,
) -> Vec<FbxSkinExport> {
    let mut skins = Vec::new();

    for (i, fbx_data) in fbx_data_list.iter().enumerate() {
        if fbx_data.clusters.is_empty() {
            continue;
        }

        let skin_uid = uid_alloc.allocate();
        let geometry_uid = if i < geometries.len() {
            geometries[i].uid
        } else {
            continue;
        };

        let clusters: Vec<FbxClusterExport> = fbx_data
            .clusters
            .iter()
            .filter_map(|cluster| {
                let bone_model_uid = bone_name_to_model_uid
                    .get(cluster.bone_name.as_str())
                    .copied()?;

                let cluster_uid = uid_alloc.allocate();

                let indices: Vec<i32> = cluster.vertex_indices.iter().map(|&i| i as i32).collect();
                let weights: Vec<f64> = cluster.vertex_weights.iter().map(|&w| w as f64).collect();

                let transform =
                    matrix4_to_flat_f64_scaled(&cluster.inverse_bind_pose, inv_unit_scale);
                let transform_link =
                    matrix4_to_flat_f64_scaled(&cluster.transform_link, inv_unit_scale);

                Some(FbxClusterExport {
                    uid: cluster_uid,
                    bone_model_uid,
                    indices,
                    weights,
                    transform,
                    transform_link,
                })
            })
            .collect();

        if !clusters.is_empty() {
            skins.push(FbxSkinExport {
                skin_uid,
                geometry_uid,
                clusters,
            });
        }
    }

    skins
}

fn matrix4_to_flat_f64_scaled(m: &Matrix4<f32>, inv_unit_scale: f32) -> [f64; 16] {
    [
        m[0][0] as f64,
        m[0][1] as f64,
        m[0][2] as f64,
        m[0][3] as f64,
        m[1][0] as f64,
        m[1][1] as f64,
        m[1][2] as f64,
        m[1][3] as f64,
        m[2][0] as f64,
        m[2][1] as f64,
        m[2][2] as f64,
        m[2][3] as f64,
        (m[3][0] * inv_unit_scale) as f64,
        (m[3][1] * inv_unit_scale) as f64,
        (m[3][2] * inv_unit_scale) as f64,
        m[3][3] as f64,
    ]
}

fn generate_bone_connections(bones: &[FbxBoneExport], connections: &mut Vec<FbxConnection>) {
    for bone in bones {
        let parent_uid = bone.parent_model_uid.unwrap_or(0);
        connections.push(FbxConnection::OO {
            child: bone.model_uid,
            parent: parent_uid,
        });

        if let Some(attr_uid) = bone.node_attribute_uid {
            connections.push(FbxConnection::OO {
                child: attr_uid,
                parent: bone.model_uid,
            });
        }
    }
}

fn generate_mesh_connections(
    mesh_models: &[FbxMeshModelExport],
    geometries: &[FbxGeometryExport],
    materials: &[FbxMaterialExport],
    textures: &[FbxTextureExport],
    skins: &[FbxSkinExport],
    connections: &mut Vec<FbxConnection>,
) {
    for (i, mesh_model) in mesh_models.iter().enumerate() {
        let parent_uid = mesh_model.parent_bone_uid.unwrap_or(0);
        connections.push(FbxConnection::OO {
            child: mesh_model.uid,
            parent: parent_uid,
        });

        if i < geometries.len() {
            connections.push(FbxConnection::OO {
                child: geometries[i].uid,
                parent: mesh_model.uid,
            });
        }
    }

    for material in materials {
        connections.push(FbxConnection::OO {
            child: material.uid,
            parent: material.mesh_model_uid,
        });
    }

    for texture in textures {
        connections.push(FbxConnection::OP {
            child: texture.texture_uid,
            parent: texture.material_uid,
            property: "DiffuseColor".to_string(),
        });

        connections.push(FbxConnection::OO {
            child: texture.video_uid,
            parent: texture.texture_uid,
        });
    }

    for skin in skins {
        connections.push(FbxConnection::OO {
            child: skin.skin_uid,
            parent: skin.geometry_uid,
        });

        for cluster in &skin.clusters {
            connections.push(FbxConnection::OO {
                child: cluster.uid,
                parent: skin.skin_uid,
            });

            connections.push(FbxConnection::OO {
                child: cluster.bone_model_uid,
                parent: cluster.uid,
            });
        }
    }
}

fn generate_animation_connections(
    stack_uid: i64,
    layer_uid: i64,
    curve_nodes: &[FbxCurveNodeExport],
    connections: &mut Vec<FbxConnection>,
) {
    connections.push(FbxConnection::OO {
        child: stack_uid,
        parent: 0,
    });

    connections.push(FbxConnection::OO {
        child: layer_uid,
        parent: stack_uid,
    });

    for cn in curve_nodes {
        connections.push(FbxConnection::OO {
            child: cn.uid,
            parent: layer_uid,
        });

        connections.push(FbxConnection::OP {
            child: cn.uid,
            parent: cn.bone_model_uid,
            property: cn.channel.property_name().to_string(),
        });

        let axis_names = ["d|X", "d|Y", "d|Z"];
        for (i, axis) in axis_names.iter().enumerate() {
            if let Some(curve_uid) = cn.curve_uids[i] {
                connections.push(FbxConnection::OP {
                    child: curve_uid,
                    parent: cn.uid,
                    property: axis.to_string(),
                });
            }
        }
    }
}

fn write_full_definitions<W: Write + Seek>(
    writer: &mut Writer<W>,
    data: &FullFbxExportData,
) -> FbxWriteResult<()> {
    let model_count = data.anim_data.bones.len() as i32 + data.mesh_models.len() as i32;
    let node_attribute_count = data
        .anim_data
        .bones
        .iter()
        .filter(|b| b.node_attribute_uid.is_some())
        .count() as i32;
    let geometry_count = data.geometries.len() as i32;
    let material_count = data.materials.len() as i32;
    let texture_count = data.textures.len() as i32;
    let video_count = data.textures.len() as i32;
    let deformer_count = data.skins.len() as i32;
    let sub_deformer_count: i32 = data.skins.iter().map(|s| s.clusters.len() as i32).sum();
    let curve_node_count = data.anim_data.curve_nodes.len() as i32;
    let curve_count = data.anim_data.curves.len() as i32;

    let total = 1
        + model_count
        + node_attribute_count
        + geometry_count
        + material_count
        + texture_count
        + video_count
        + deformer_count
        + sub_deformer_count
        + 1
        + 1
        + curve_node_count
        + curve_count;

    drop(writer.new_node("Definitions")?);

    {
        let mut attrs = writer.new_node("Version")?;
        attrs.append_i32(100)?;
        drop(attrs);
        writer.close_node()?;
    }

    {
        let mut attrs = writer.new_node("Count")?;
        attrs.append_i32(total)?;
        drop(attrs);
        writer.close_node()?;
    }

    write_object_type(writer, "GlobalSettings", 1)?;
    write_object_type(writer, "Model", model_count)?;

    if node_attribute_count > 0 {
        write_object_type(writer, "NodeAttribute", node_attribute_count)?;
    }

    if geometry_count > 0 {
        write_object_type(writer, "Geometry", geometry_count)?;
    }
    if material_count > 0 {
        write_object_type(writer, "Material", material_count)?;
    }
    if texture_count > 0 {
        write_object_type(writer, "Texture", texture_count)?;
    }
    if video_count > 0 {
        write_object_type(writer, "Video", video_count)?;
    }
    if deformer_count > 0 {
        write_object_type(writer, "Deformer", deformer_count + sub_deformer_count)?;
    }

    write_object_type(writer, "AnimationStack", 1)?;
    write_object_type(writer, "AnimationLayer", 1)?;

    if curve_node_count > 0 {
        write_object_type(writer, "AnimationCurveNode", curve_node_count)?;
    }
    if curve_count > 0 {
        write_object_type(writer, "AnimationCurve", curve_count)?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_mesh_model<W: Write + Seek>(
    writer: &mut Writer<W>,
    mesh: &FbxMeshModelExport,
) -> FbxWriteResult<()> {
    let fbx_name = format!("{}\x00\x01Model", mesh.name);
    let mut attrs = writer.new_node("Model")?;
    attrs.append_i64(mesh.uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("Mesh")?;
    drop(attrs);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(232)?;
        drop(va);
        writer.close_node()?;
    }

    drop(writer.new_node("Properties70")?);
    write_property_f64x3(
        writer,
        "Lcl Translation",
        "Lcl Translation",
        "",
        "A",
        mesh.translation[0],
        mesh.translation[1],
        mesh.translation[2],
    )?;
    write_property_f64x3(
        writer,
        "Lcl Rotation",
        "Lcl Rotation",
        "",
        "A",
        mesh.rotation[0],
        mesh.rotation[1],
        mesh.rotation[2],
    )?;
    write_property_f64x3(
        writer,
        "Lcl Scaling",
        "Lcl Scaling",
        "",
        "A",
        mesh.scaling[0],
        mesh.scaling[1],
        mesh.scaling[2],
    )?;
    writer.close_node()?;

    writer.close_node()?;
    Ok(())
}

fn write_geometry<W: Write + Seek>(
    writer: &mut Writer<W>,
    geo: &FbxGeometryExport,
    has_material: bool,
) -> FbxWriteResult<()> {
    let fbx_name = format!("\x00\x01Geometry");
    let mut attrs = writer.new_node("Geometry")?;
    attrs.append_i64(geo.uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("Mesh")?;
    drop(attrs);

    {
        let mut va = writer.new_node("Vertices")?;
        va.append_arr_f64_from_iter(None, geo.positions.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("PolygonVertexIndex")?;
        va.append_arr_i32_from_iter(None, geo.polygon_vertex_index.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    if !geo.normals.is_empty() {
        write_layer_element_normal(writer, &geo.normals)?;
    }

    if !geo.uv_values.is_empty() {
        write_layer_element_uv(writer, &geo.uv_values)?;
    }

    if has_material {
        write_layer_element_material(writer)?;
    }

    if !geo.normals.is_empty() || !geo.uv_values.is_empty() || has_material {
        write_layer(
            writer,
            !geo.normals.is_empty(),
            !geo.uv_values.is_empty(),
            has_material,
        )?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_layer_element_normal<W: Write + Seek>(
    writer: &mut Writer<W>,
    normals: &[f64],
) -> FbxWriteResult<()> {
    drop(writer.new_node("LayerElementNormal")?);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(101)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Name")?;
        va.append_string_direct("")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("MappingInformationType")?;
        va.append_string_direct("ByVertice")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("ReferenceInformationType")?;
        va.append_string_direct("Direct")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Normals")?;
        va.append_arr_f64_from_iter(None, normals.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_layer_element_uv<W: Write + Seek>(
    writer: &mut Writer<W>,
    uv_values: &[f64],
) -> FbxWriteResult<()> {
    drop(writer.new_node("LayerElementUV")?);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(101)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Name")?;
        va.append_string_direct("UVMap")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("MappingInformationType")?;
        va.append_string_direct("ByVertice")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("ReferenceInformationType")?;
        va.append_string_direct("Direct")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("UV")?;
        va.append_arr_f64_from_iter(None, uv_values.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_layer_element_material<W: Write + Seek>(writer: &mut Writer<W>) -> FbxWriteResult<()> {
    drop(writer.new_node("LayerElementMaterial")?);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(101)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Name")?;
        va.append_string_direct("")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("MappingInformationType")?;
        va.append_string_direct("AllSame")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("ReferenceInformationType")?;
        va.append_string_direct("IndexToDirect")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Materials")?;
        va.append_arr_i32_from_iter(None, [0].iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_layer<W: Write + Seek>(
    writer: &mut Writer<W>,
    has_normal: bool,
    has_uv: bool,
    has_material: bool,
) -> FbxWriteResult<()> {
    drop(writer.new_node("Layer")?);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(100)?;
        drop(va);
        writer.close_node()?;
    }

    if has_normal {
        drop(writer.new_node("LayerElement")?);

        {
            let mut va = writer.new_node("Type")?;
            va.append_string_direct("LayerElementNormal")?;
            drop(va);
            writer.close_node()?;
        }

        {
            let mut va = writer.new_node("TypedIndex")?;
            va.append_i32(0)?;
            drop(va);
            writer.close_node()?;
        }

        writer.close_node()?;
    }

    if has_uv {
        drop(writer.new_node("LayerElement")?);

        {
            let mut va = writer.new_node("Type")?;
            va.append_string_direct("LayerElementUV")?;
            drop(va);
            writer.close_node()?;
        }

        {
            let mut va = writer.new_node("TypedIndex")?;
            va.append_i32(0)?;
            drop(va);
            writer.close_node()?;
        }

        writer.close_node()?;
    }

    if has_material {
        drop(writer.new_node("LayerElement")?);

        {
            let mut va = writer.new_node("Type")?;
            va.append_string_direct("LayerElementMaterial")?;
            drop(va);
            writer.close_node()?;
        }

        {
            let mut va = writer.new_node("TypedIndex")?;
            va.append_i32(0)?;
            drop(va);
            writer.close_node()?;
        }

        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_material<W: Write + Seek>(
    writer: &mut Writer<W>,
    mat: &FbxMaterialExport,
) -> FbxWriteResult<()> {
    let fbx_name = format!("{}\x00\x01Material", mat.name);
    let mut attrs = writer.new_node("Material")?;
    attrs.append_i64(mat.uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("")?;
    drop(attrs);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(102)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("ShadingModel")?;
        va.append_string_direct("phong")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("MultiLayer")?;
        va.append_i32(0)?;
        drop(va);
        writer.close_node()?;
    }

    drop(writer.new_node("Properties70")?);

    write_property_f64x3(
        writer,
        "DiffuseColor",
        "Color",
        "",
        "A",
        mat.diffuse_color[0],
        mat.diffuse_color[1],
        mat.diffuse_color[2],
    )?;

    write_property_f64(writer, "DiffuseFactor", "Number", "", "A", 1.0)?;
    write_property_f64(writer, "Opacity", "Number", "", "A", 1.0)?;

    write_property_f64x3(
        writer,
        "AmbientColor",
        "Color",
        "",
        "A",
        mat.diffuse_color[0],
        mat.diffuse_color[1],
        mat.diffuse_color[2],
    )?;

    write_property_f64x3(writer, "SpecularColor", "Color", "", "A", 0.9, 0.9, 0.9)?;

    write_property_f64(writer, "Shininess", "Number", "", "A", 20.0)?;
    write_property_f64(writer, "ShininessExponent", "Number", "", "A", 20.0)?;

    writer.close_node()?;

    writer.close_node()?;
    Ok(())
}

fn write_texture<W: Write + Seek>(
    writer: &mut Writer<W>,
    tex: &FbxTextureExport,
) -> FbxWriteResult<()> {
    let fbx_name = format!("\x00\x01Texture");
    let mut attrs = writer.new_node("Texture")?;
    attrs.append_i64(tex.texture_uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("")?;
    drop(attrs);

    {
        let mut va = writer.new_node("Type")?;
        va.append_string_direct("TextureDiffuse")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("FileName")?;
        va.append_string_direct(&tex.filename)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("RelativeFilename")?;
        va.append_string_direct(&tex.relative_filename)?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_video<W: Write + Seek>(
    writer: &mut Writer<W>,
    tex: &FbxTextureExport,
) -> FbxWriteResult<()> {
    let fbx_name = format!("\x00\x01Video");
    let mut attrs = writer.new_node("Video")?;
    attrs.append_i64(tex.video_uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("Clip")?;
    drop(attrs);

    {
        let mut va = writer.new_node("FileName")?;
        va.append_string_direct(&tex.filename)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("RelativeFilename")?;
        va.append_string_direct(&tex.relative_filename)?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_skin_deformer<W: Write + Seek>(
    writer: &mut Writer<W>,
    skin: &FbxSkinExport,
) -> FbxWriteResult<()> {
    let mut attrs = writer.new_node("Deformer")?;
    attrs.append_i64(skin.skin_uid)?;
    attrs.append_string_direct("\x00\x01Deformer")?;
    attrs.append_string_direct("Skin")?;
    drop(attrs);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(101)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Link_DeformAcuracy")?;
        va.append_f64(50.0)?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_cluster<W: Write + Seek>(
    writer: &mut Writer<W>,
    cluster: &FbxClusterExport,
) -> FbxWriteResult<()> {
    let mut attrs = writer.new_node("Deformer")?;
    attrs.append_i64(cluster.uid)?;
    attrs.append_string_direct("\x00\x01SubDeformer")?;
    attrs.append_string_direct("Cluster")?;
    drop(attrs);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(100)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Indexes")?;
        va.append_arr_i32_from_iter(None, cluster.indices.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Weights")?;
        va.append_arr_f64_from_iter(None, cluster.weights.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Transform")?;
        va.append_arr_f64_from_iter(None, cluster.transform.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("TransformLink")?;
        va.append_arr_f64_from_iter(None, cluster.transform_link.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_full_objects<W: Write + Seek>(
    writer: &mut Writer<W>,
    data: &FullFbxExportData,
) -> FbxWriteResult<()> {
    drop(writer.new_node("Objects")?);

    for bone in &data.anim_data.bones {
        write_bone_model(writer, bone)?;
    }

    for bone in &data.anim_data.bones {
        write_node_attribute(writer, bone)?;
    }

    let has_material = !data.materials.is_empty();
    for geo in &data.geometries {
        write_geometry(writer, geo, has_material)?;
    }

    for mesh_model in &data.mesh_models {
        write_mesh_model(writer, mesh_model)?;
    }

    for material in &data.materials {
        write_material(writer, material)?;
    }

    for texture in &data.textures {
        write_texture(writer, texture)?;
        write_video(writer, texture)?;
    }

    for skin in &data.skins {
        write_skin_deformer(writer, skin)?;
        for cluster in &skin.clusters {
            write_cluster(writer, cluster)?;
        }
    }

    write_anim_stack(writer, &data.anim_data)?;
    write_anim_layer(writer, data.anim_data.layer_uid)?;

    for cn in &data.anim_data.curve_nodes {
        write_anim_curve_node(writer, cn)?;
    }

    for curve in &data.anim_data.curves {
        write_anim_curve(writer, curve)?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_full_fbx_binary<W: Write + Seek>(
    mut writer: Writer<W>,
    data: &FullFbxExportData,
) -> FbxWriteResult<()> {
    write_header_extension(&mut writer)?;
    let unit_scale_factor = (data.unit_scale * 100.0) as f64;
    write_global_settings(
        &mut writer,
        data.anim_data.duration_ktime,
        &data.anim_data.axes,
        data.anim_data.fps,
        unit_scale_factor,
    )?;
    write_documents(&mut writer, data.anim_data.document_uid)?;
    write_references(&mut writer)?;
    write_full_definitions(&mut writer, data)?;
    write_full_objects(&mut writer, data)?;
    write_connections(&mut writer, &data.anim_data)?;
    writer.finalize_and_flush(&FbxFooter::default())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_triangle_polygon_indices() {
        let indices = vec![0, 1, 2, 3, 4, 5];
        let encoded = encode_triangle_polygon_indices(&indices);
        assert_eq!(encoded, vec![0, 1, -3, 3, 4, -6]);
    }

    #[test]
    fn test_encode_triangle_polygon_indices_single() {
        let indices = vec![0, 1, 2];
        let encoded = encode_triangle_polygon_indices(&indices);
        assert_eq!(encoded, vec![0, 1, -3]);
    }

    #[test]
    fn test_convert_uvs_to_fbx_flip() {
        let mut fbx_data = FbxData::new();
        fbx_data.tex_coords = vec![[0.5, 0.3]];
        let uv_values = convert_uvs_to_fbx(&fbx_data);
        assert!((uv_values[0] - 0.5).abs() < 1e-6);
        assert!((uv_values[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_matrix4_to_flat_f64_scaled() {
        use cgmath::SquareMatrix;
        let identity = Matrix4::<f32>::identity();
        let flat = matrix4_to_flat_f64_scaled(&identity, 2.0);
        assert!((flat[0] - 1.0).abs() < 1e-8);
        assert!((flat[5] - 1.0).abs() < 1e-8);
        assert!((flat[10] - 1.0).abs() < 1e-8);
        assert!((flat[15] - 1.0).abs() < 1e-8);
        assert!((flat[12] - 0.0).abs() < 1e-8);
    }

    #[test]
    fn test_convert_positions_to_fbx_no_scale() {
        let mut fbx_data = FbxData::new();
        fbx_data.positions = vec![cgmath::Vector3::new(0.01, 0.02, 0.03)];
        fbx_data.local_positions = vec![];

        let positions = convert_positions_to_fbx(&fbx_data, 1.0);
        assert!((positions[0] - 0.01).abs() < 1e-6);
        assert!((positions[1] - 0.02).abs() < 1e-6);
        assert!((positions[2] - 0.03).abs() < 1e-6);
    }

    #[test]
    fn test_fbx_roundtrip_stickman() {
        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping roundtrip test: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");

        let result =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");
        let (load_result, _) = result;

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("roundtrip_test.fbx");

        export_full_fbx(&fbx_model, None, &skeleton, &export_path).expect("Failed to export FBX");

        let original_scene = ufbx::load_file(original_path, ufbx::LoadOpts::default())
            .expect("Failed to load original with ufbx");
        let exported_scene =
            ufbx::load_file(export_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to load exported with ufbx");

        let orig_axes = &original_scene.settings.axes;
        let exp_axes = &exported_scene.settings.axes;
        assert_eq!(
            orig_axes.up as i32, exp_axes.up as i32,
            "UpAxis mismatch: original={:?}, exported={:?}",
            orig_axes.up, exp_axes.up
        );
        assert_eq!(
            orig_axes.front as i32, exp_axes.front as i32,
            "FrontAxis mismatch: original={:?}, exported={:?}",
            orig_axes.front, exp_axes.front
        );
        assert_eq!(
            orig_axes.right as i32, exp_axes.right as i32,
            "CoordAxis mismatch: original={:?}, exported={:?}",
            orig_axes.right, exp_axes.right
        );

        let orig_non_root_nodes: Vec<_> =
            original_scene.nodes.iter().filter(|n| !n.is_root).collect();
        let exp_non_root_nodes: Vec<_> =
            exported_scene.nodes.iter().filter(|n| !n.is_root).collect();

        let orig_names: std::collections::HashSet<String> = orig_non_root_nodes
            .iter()
            .map(|n| n.element.name.to_string())
            .collect();
        let exp_names: std::collections::HashSet<String> = exp_non_root_nodes
            .iter()
            .map(|n| n.element.name.to_string())
            .collect();

        let missing_in_export: Vec<_> = orig_names.difference(&exp_names).collect();
        let extra_in_export: Vec<_> = exp_names.difference(&orig_names).collect();
        assert!(
            missing_in_export.is_empty(),
            "Nodes missing in exported FBX: {:?}",
            missing_in_export
        );
        if !extra_in_export.is_empty() {
            eprintln!("Extra nodes in export (acceptable): {:?}", extra_in_export);
        }

        for orig_node in &orig_non_root_nodes {
            let name = orig_node.element.name.to_string();
            let exp_node = exp_non_root_nodes
                .iter()
                .find(|n| n.element.name.to_string() == name);

            let Some(exp_node) = exp_node else {
                continue;
            };

            let orig_t = &orig_node.local_transform;
            let exp_t = &exp_node.local_transform;

            let position_tolerance = 0.1;
            let orig_pos = [
                orig_t.translation.x,
                orig_t.translation.y,
                orig_t.translation.z,
            ];
            let exp_pos = [
                exp_t.translation.x,
                exp_t.translation.y,
                exp_t.translation.z,
            ];

            for axis in 0..3 {
                let diff = (orig_pos[axis] - exp_pos[axis]).abs();
                assert!(
                    diff < position_tolerance,
                    "Node '{}' position[{}] mismatch: original={}, exported={}, diff={}",
                    name,
                    axis,
                    orig_pos[axis],
                    exp_pos[axis],
                    diff
                );
            }
        }

        assert!(
            !exported_scene.anim_stacks.is_empty(),
            "Exported FBX has no animation stacks"
        );

        std::fs::remove_file(&export_path).ok();
    }

    #[test]
    fn test_fbx_roundtrip_stickman_with_animation() {
        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();
        let anim_clip = load_result.clips.first().expect("No animation clip found");

        let bone_names: std::collections::HashMap<u32, String> = skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b.name.clone()))
            .collect();
        let editable = thyllore_anim_core::editable::clip_from_animation(1, anim_clip, &bone_names);
        assert!(editable.duration > 0.0);
        assert!(!editable.tracks.is_empty());

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("roundtrip_anim_test.fbx");

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &export_path)
            .expect("Failed to export FBX with animation");

        let original_scene = ufbx::load_file(original_path, ufbx::LoadOpts::default())
            .expect("Failed to load original with ufbx");
        let exported_scene =
            ufbx::load_file(export_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to load exported with ufbx");

        assert!(!exported_scene.anim_stacks.is_empty());
        assert!(
            (exported_scene.settings.frames_per_second - original_scene.settings.frames_per_second)
                .abs()
                < 1.0
        );

        let anim_stack = &exported_scene.anim_stacks[0];
        let time_span = anim_stack.time_end - anim_stack.time_begin;
        assert!(
            time_span > 0.1,
            "Animation time span too short: {:.4}s",
            time_span
        );

        let baked = ufbx::bake_anim(
            &exported_scene,
            &exported_scene.anim_stacks[0].anim,
            ufbx::BakeOpts::default(),
        )
        .expect("Failed to bake exported animation");

        let orig_baked = ufbx::bake_anim(
            &original_scene,
            &original_scene.anim_stacks[0].anim,
            ufbx::BakeOpts::default(),
        )
        .expect("Failed to bake original animation");

        let animated_count = baked
            .nodes
            .iter()
            .filter(|n| n.rotation_keys.len() > 1)
            .count();
        assert!(animated_count > 0, "Exported FBX has no animated nodes");

        let mesh_node_names: Vec<String> = fbx_model
            .fbx_data
            .iter()
            .filter_map(|d| d.mesh_node_name.clone())
            .collect();

        for mesh_name in &mesh_node_names {
            let orig_parent = original_scene
                .nodes
                .iter()
                .find(|n| n.element.name.to_string() == *mesh_name)
                .and_then(|n| n.parent.as_ref())
                .map(|p| p.element.name.to_string())
                .unwrap_or_default();
            let exp_parent = exported_scene
                .nodes
                .iter()
                .find(|n| n.element.name.to_string() == *mesh_name)
                .and_then(|n| n.parent.as_ref())
                .map(|p| p.element.name.to_string())
                .unwrap_or_default();
            assert_eq!(
                orig_parent, exp_parent,
                "Parent mismatch for mesh '{}'",
                mesh_name
            );
        }

        for orig_bn in &orig_baked.nodes {
            let orig_idx = orig_bn.typed_id as usize;
            if orig_idx >= original_scene.nodes.len() || orig_bn.rotation_keys.len() <= 2 {
                continue;
            }
            let name = original_scene.nodes[orig_idx].element.name.to_string();

            let exp_bn = baked.nodes.iter().find(|bn| {
                let idx = bn.typed_id as usize;
                idx < exported_scene.nodes.len()
                    && exported_scene.nodes[idx].element.name.to_string() == name
            });
            let exp_bn = match exp_bn {
                Some(b) => b,
                None => continue,
            };

            assert_eq!(
                orig_bn.rotation_keys.len(),
                exp_bn.rotation_keys.len(),
                "Rotation key count mismatch for bone '{}'",
                name
            );

            let sample_indices = [0, 100, 500, 1000];
            for &idx in &sample_indices {
                if idx >= orig_bn.rotation_keys.len() {
                    break;
                }
                let o = &orig_bn.rotation_keys[idx];
                let e = &exp_bn.rotation_keys[idx];
                let max_diff = (o.value.w - e.value.w)
                    .abs()
                    .max((o.value.x - e.value.x).abs())
                    .max((o.value.y - e.value.y).abs())
                    .max((o.value.z - e.value.z).abs());
                assert!(
                    max_diff < 0.01,
                    "Rotation value mismatch for bone '{}' at key {}: diff={}",
                    name,
                    idx,
                    max_diff
                );
            }
        }
    }

    #[test]
    fn test_exported_bone_node_types_match_original() {
        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();
        let anim_clip = load_result.clips.first().expect("No animation clip found");

        let bone_names: std::collections::HashMap<u32, String> = skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b.name.clone()))
            .collect();
        let editable = thyllore_anim_core::editable::clip_from_animation(1, anim_clip, &bone_names);

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("blender_compat_test.fbx");

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &export_path)
            .expect("Failed to export FBX");

        let original_scene = ufbx::load_file(original_path, ufbx::LoadOpts::default())
            .expect("Failed to load original with ufbx");
        let exported_scene =
            ufbx::load_file(export_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to load exported with ufbx");

        let mesh_node_names: std::collections::HashSet<String> = fbx_model
            .fbx_data
            .iter()
            .filter_map(|d| d.mesh_node_name.clone())
            .collect();

        let bone_only_names: std::collections::HashSet<String> = skeleton
            .bones
            .iter()
            .filter(|b| !mesh_node_names.contains(&b.name))
            .map(|b| b.name.clone())
            .collect();

        for exp_node in exported_scene.nodes.iter() {
            let name = exp_node.element.name.to_string();
            if !bone_only_names.contains(&name) {
                continue;
            }

            let orig_node = original_scene
                .nodes
                .iter()
                .find(|n| n.element.name.to_string() == name);

            if let Some(orig_node) = orig_node {
                assert_eq!(
                    exp_node.attrib_type as i32, orig_node.attrib_type as i32,
                    "Node '{}' attrib_type mismatch: exported={:?}, original={:?}",
                    name, exp_node.attrib_type, orig_node.attrib_type,
                );
            }

            assert_eq!(
                exp_node.attrib_type,
                ufbx::ElementType::Unknown,
                "Node '{}' should have no NodeAttribute (attrib_type=Unknown) for Blender object-level animation, got {:?}",
                name,
                exp_node.attrib_type,
            );
        }

        let exported_bone_count = exported_scene
            .nodes
            .iter()
            .filter(|n| n.attrib_type == ufbx::ElementType::Bone)
            .count();
        assert_eq!(
            exported_bone_count, 0,
            "Exported FBX should have 0 Bone-type nodes for object-level animation, found {}",
            exported_bone_count,
        );

        std::fs::remove_file(&export_path).ok();
    }

    #[test]
    fn test_compare_anim_structure() {
        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();
        let anim_clip = load_result.clips.first().expect("No animation clip found");

        let bone_names: std::collections::HashMap<u32, String> = skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b.name.clone()))
            .collect();
        let editable = thyllore_anim_core::editable::clip_from_animation(1, anim_clip, &bone_names);

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("anim_structure_test.fbx");

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &export_path)
            .expect("Failed to export FBX with animation");

        let original_scene = ufbx::load_file(original_path, ufbx::LoadOpts::default())
            .expect("Failed to load original with ufbx");
        let exported_scene =
            ufbx::load_file(export_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to load exported with ufbx");

        print_scene_anim_structure("ORIGINAL", &original_scene);
        eprintln!("\n{}\n", "=".repeat(80));
        print_scene_anim_structure("EXPORTED", &exported_scene);

        eprintln!("\n{}\n", "=".repeat(80));
        print_anim_prop_connections("ORIGINAL", &original_scene);
        eprintln!("\n{}\n", "=".repeat(80));
        print_anim_prop_connections("EXPORTED", &exported_scene);

        std::fs::remove_file(&export_path).ok();
    }

    fn canonicalize_no_prefix(path: &std::path::Path) -> PathBuf {
        match path.canonicalize() {
            Ok(p) => {
                let s = p.to_string_lossy();
                if let Some(stripped) = s.strip_prefix(r"\\?\") {
                    PathBuf::from(stripped)
                } else {
                    p
                }
            }
            Err(_) => path.to_path_buf(),
        }
    }

    fn read_blender_path() -> Option<String> {
        let paths_file = std::path::Path::new(".claude/local/paths.md");
        let content = std::fs::read_to_string(paths_file).ok()?;
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("- BlenderPath = ") {
                let path = rest.trim().to_string();
                if std::path::Path::new(&path).exists() {
                    return Some(path);
                }
            }
        }
        None
    }

    #[test]
    fn test_blender_animation_import() {
        let blender_path = match read_blender_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipping: BlenderPath not configured in .claude/local/paths.md");
                return;
            }
        };

        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let script_path = "scripts/blender_fbx_diagnostic.py";
        if !std::path::Path::new(script_path).exists() {
            eprintln!("Skipping: {} not found", script_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();
        let anim_clip = load_result.clips.first().expect("No animation clip found");

        let bone_names: std::collections::HashMap<u32, String> = skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b.name.clone()))
            .collect();
        let editable = thyllore_anim_core::editable::clip_from_animation(1, anim_clip, &bone_names);

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("blender_anim_test.fbx");

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &export_path)
            .expect("Failed to export FBX");

        let abs_export = canonicalize_no_prefix(&export_path);
        let abs_script = canonicalize_no_prefix(std::path::Path::new(script_path));

        let abs_output = canonicalize_no_prefix(std::path::Path::new("assets/exports"))
            .join("blender_diagnostic.json");

        let output = std::process::Command::new(&blender_path)
            .args([
                "--background",
                "--python",
                abs_script.to_str().unwrap(),
                "--",
                abs_export.to_str().unwrap(),
                abs_output.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to run Blender");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("Blender stdout:\n{}", stdout);
        eprintln!("Blender stderr:\n{}", stderr);

        assert!(
            output.status.success(),
            "Blender exited with error: {:?}",
            output.status,
        );

        assert!(
            abs_output.exists(),
            "Blender diagnostic JSON not created at {:?}",
            abs_output,
        );

        let json_content =
            std::fs::read_to_string(&abs_output).expect("Failed to read diagnostic JSON");
        let diagnostic: serde_json::Value =
            serde_json::from_str(&json_content).expect("Failed to parse diagnostic JSON");

        let summary = &diagnostic["summary"];

        let total_actions = summary["total_actions"].as_u64().unwrap_or(0);
        eprintln!("Blender imported actions: {}", total_actions);
        assert!(
            total_actions > 0,
            "Blender should import at least 1 action, got {}",
            total_actions,
        );

        let total_fcurves = summary["total_fcurves"].as_u64().unwrap_or(0);
        eprintln!("Blender imported FCurves: {}", total_fcurves);
        assert!(
            total_fcurves > 0,
            "Blender should import FCurves, got {}",
            total_fcurves,
        );

        let moved = summary["moved"].as_array().map(|a| a.len()).unwrap_or(0);
        eprintln!("Objects that moved during playback: {}", moved);
        assert!(
            moved > 0,
            "At least some objects should move during animation playback, got {}",
            moved,
        );

        std::fs::remove_file(&export_path).ok();
        std::fs::remove_file(&abs_output).ok();
    }

    fn print_scene_anim_structure(label: &str, scene: &ufbx::Scene) {
        eprintln!("--- {} ---", label);
        eprintln!("  anim_stacks: {}", scene.anim_stacks.len());
        eprintln!("  anim_layers: {}", scene.anim_layers.len());
        eprintln!("  anim_values: {}", scene.anim_values.len());
        eprintln!("  anim_curves: {}", scene.anim_curves.len());
        eprintln!("  total nodes: {}", scene.nodes.len());

        for (i, stack) in scene.anim_stacks.iter().enumerate() {
            eprintln!(
                "  AnimStack[{}]: name='{}', time_begin={:.4}, time_end={:.4}, layers={}",
                i,
                stack.element.name,
                stack.time_begin,
                stack.time_end,
                stack.layers.len(),
            );
        }

        let bake_opts = ufbx::BakeOpts::default();
        let baked = ufbx::bake_anim(scene, &scene.anim_stacks[0].anim, bake_opts)
            .expect("Failed to bake animation");

        let mut bone_only_animated = 0u32;
        let mut mesh_node_animated = 0u32;

        eprintln!("  Baked nodes total: {}", baked.nodes.len());
        for bn in &baked.nodes {
            let has_translation_anim = bn.translation_keys.len() > 1;
            let has_rotation_anim = bn.rotation_keys.len() > 1;
            let has_scale_anim = bn.scale_keys.len() > 1;
            if !has_translation_anim && !has_rotation_anim && !has_scale_anim {
                continue;
            }

            let node_idx = bn.typed_id as usize;
            if node_idx >= scene.nodes.len() {
                continue;
            }
            let node = &scene.nodes[node_idx];
            let name = node.element.name.to_string();
            let has_mesh = node.mesh.is_some();
            let attrib_type = node.attrib_type;

            if has_mesh {
                mesh_node_animated += 1;
            } else {
                bone_only_animated += 1;
            }

            eprintln!(
                "    ANIMATED: '{}' attrib={:?} has_mesh={} t_keys={} r_keys={} s_keys={} const_t={} const_r={} const_s={}",
                name,
                attrib_type,
                has_mesh,
                bn.translation_keys.len(),
                bn.rotation_keys.len(),
                bn.scale_keys.len(),
                bn.constant_translation,
                bn.constant_rotation,
                bn.constant_scale,
            );
        }

        eprintln!(
            "  SUMMARY: bone-only animated={}, mesh-node animated={}",
            bone_only_animated, mesh_node_animated
        );
    }

    #[test]
    fn test_fbx_roundtrip_skinned_fly() {
        let original_path = "tests/testmodels/fbx/skinning/source/fly.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("roundtrip_skinned_fly.fbx");

        export_full_fbx(&fbx_model, None, &skeleton, &export_path).expect("Failed to export FBX");

        let original_scene = ufbx::load_file(original_path, ufbx::LoadOpts::default())
            .expect("Failed to load original with ufbx");
        let exported_scene =
            ufbx::load_file(export_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to load exported with ufbx");

        assert!(
            !exported_scene.meshes.is_empty(),
            "Exported FBX should have at least one mesh",
        );

        assert!(
            !exported_scene.materials.is_empty(),
            "Exported FBX should have at least one material",
        );

        assert!(
            !exported_scene.skin_clusters.is_empty(),
            "Exported FBX should have skin clusters",
        );

        let orig_bone_names: std::collections::HashSet<String> = original_scene
            .skin_clusters
            .iter()
            .filter_map(|c| c.bone_node.as_ref().map(|n| n.element.name.to_string()))
            .collect();
        let exp_bone_names: std::collections::HashSet<String> = exported_scene
            .skin_clusters
            .iter()
            .filter_map(|c| c.bone_node.as_ref().map(|n| n.element.name.to_string()))
            .collect();
        let missing_bones: Vec<_> = orig_bone_names.difference(&exp_bone_names).collect();
        assert!(
            missing_bones.is_empty(),
            "Missing bone references in exported clusters: {:?}",
            missing_bones,
        );

        for exp_cluster in exported_scene.skin_clusters.iter() {
            assert!(
                exp_cluster.bone_node.is_some(),
                "Exported cluster should have a bone_node reference",
            );
            assert!(
                exp_cluster.num_weights > 0,
                "Exported cluster should have vertex weights",
            );
        }

        for exp_mesh in exported_scene.meshes.iter() {
            assert!(
                !exp_mesh.materials.is_empty(),
                "Exported mesh should have at least one material reference",
            );
            assert!(
                !exp_mesh.skin_deformers.is_empty(),
                "Exported skinned mesh should have a skin deformer",
            );
        }

        assert!(
            (original_scene.settings.unit_meters - exported_scene.settings.unit_meters).abs()
                < 1e-6,
            "UnitScaleFactor mismatch: original unit_meters={}, exported unit_meters={}",
            original_scene.settings.unit_meters,
            exported_scene.settings.unit_meters,
        );

        let orig_g2b_map: std::collections::HashMap<String, &ufbx::Matrix> = original_scene
            .skin_clusters
            .iter()
            .filter_map(|c| {
                c.bone_node
                    .as_ref()
                    .map(|n| (n.element.name.to_string(), &c.geometry_to_bone))
            })
            .collect();

        for exp_cluster in exported_scene.skin_clusters.iter() {
            let bone_name = exp_cluster
                .bone_node
                .as_ref()
                .map(|n| n.element.name.to_string())
                .unwrap_or_default();

            if let Some(&orig_g2b) = orig_g2b_map.get(&bone_name) {
                let diff = (orig_g2b.m03 - exp_cluster.geometry_to_bone.m03).abs()
                    + (orig_g2b.m13 - exp_cluster.geometry_to_bone.m13).abs()
                    + (orig_g2b.m23 - exp_cluster.geometry_to_bone.m23).abs();
                assert!(
                    diff < 0.1,
                    "geometry_to_bone translation mismatch for bone '{}': \
                     orig=[{:.4}, {:.4}, {:.4}], exp=[{:.4}, {:.4}, {:.4}]",
                    bone_name,
                    orig_g2b.m03,
                    orig_g2b.m13,
                    orig_g2b.m23,
                    exp_cluster.geometry_to_bone.m03,
                    exp_cluster.geometry_to_bone.m13,
                    exp_cluster.geometry_to_bone.m23,
                );
            }
        }

        let orig_mat_map: std::collections::HashMap<String, &ufbx::Material> = original_scene
            .materials
            .iter()
            .map(|m| (m.element.name.to_string(), m.as_ref()))
            .collect();

        for exp_mat in exported_scene.materials.iter() {
            let name = exp_mat.element.name.to_string();
            if let Some(orig_mat) = orig_mat_map.get(&name) {
                let orig_dc = &orig_mat.fbx.diffuse_color.value_vec4;
                let exp_dc = &exp_mat.fbx.diffuse_color.value_vec4;
                let color_diff = (orig_dc.x - exp_dc.x).abs()
                    + (orig_dc.y - exp_dc.y).abs()
                    + (orig_dc.z - exp_dc.z).abs();
                assert!(
                    color_diff < 0.01,
                    "DiffuseColor mismatch for '{}': orig=[{:.3},{:.3},{:.3}] exp=[{:.3},{:.3},{:.3}]",
                    name, orig_dc.x, orig_dc.y, orig_dc.z, exp_dc.x, exp_dc.y, exp_dc.z,
                );

                assert!(
                    exp_mat.fbx.diffuse_factor.has_value,
                    "DiffuseFactor must be explicitly set for '{}'",
                    name,
                );
            }
        }

        for exp_mat in exported_scene.materials.iter() {
            let name = exp_mat.element.name.to_string();
            if let Some(orig_mat) = orig_mat_map.get(&name) {
                let orig_has_tex = orig_mat.fbx.diffuse_color.texture.is_some();
                let exp_has_tex = exp_mat.fbx.diffuse_color.texture.is_some();
                assert_eq!(
                    orig_has_tex, exp_has_tex,
                    "Texture presence mismatch for '{}': original={}, exported={}",
                    name, orig_has_tex, exp_has_tex,
                );

                if let (Some(orig_tex), Some(exp_tex)) = (
                    orig_mat.fbx.diffuse_color.texture.as_ref(),
                    exp_mat.fbx.diffuse_color.texture.as_ref(),
                ) {
                    let orig_stem = Path::new(&orig_tex.filename.to_string())
                        .file_stem()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string();
                    let exp_basename = Path::new(&exp_tex.filename.to_string())
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string();
                    assert!(
                        exp_basename.starts_with(&orig_stem),
                        "Texture filename mismatch for '{}': original stem='{}', exported='{}'",
                        name,
                        orig_stem,
                        exp_basename,
                    );
                }
            }
        }

        std::fs::remove_file(&export_path).ok();
    }

    fn print_anim_prop_connections(label: &str, scene: &ufbx::Scene) {
        eprintln!("--- {} AnimProp connections ---", label);

        if scene.anim_layers.is_empty() {
            eprintln!("  No anim layers");
            return;
        }

        let layer = &scene.anim_layers[0];
        eprintln!(
            "  AnimLayer '{}' has {} anim_props",
            layer.element.name,
            layer.anim_props.len()
        );

        let mut node_prop_map: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();

        for ap in &layer.anim_props {
            let target_name = ap.element.name.to_string();
            let prop_name = ap.prop_name.to_string();
            node_prop_map
                .entry(target_name)
                .or_default()
                .push(prop_name);
        }

        for (target_name, props) in &node_prop_map {
            let node = scene
                .nodes
                .iter()
                .find(|n| n.element.name.to_string() == *target_name);

            let (has_mesh, attrib_type) = match node {
                Some(n) => (n.mesh.is_some(), format!("{:?}", n.attrib_type)),
                None => (false, "NOT_A_NODE".to_string()),
            };

            eprintln!(
                "    target='{}' attrib={} has_mesh={} props={:?}",
                target_name, attrib_type, has_mesh, props
            );
        }
    }

    #[test]
    fn test_blender_skinned_fly_import() {
        let blender_path = match read_blender_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipping: BlenderPath not configured");
                return;
            }
        };

        let original_path = "tests/testmodels/fbx/skinning/source/fly.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let script_path = "scripts/blender_fbx_diagnostic.py";
        if !std::path::Path::new(script_path).exists() {
            eprintln!("Skipping: {} not found", script_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("blender_skinned_fly.fbx");

        export_full_fbx(&fbx_model, None, &skeleton, &export_path).expect("Failed to export FBX");

        let abs_export = canonicalize_no_prefix(&export_path);
        let abs_script = canonicalize_no_prefix(std::path::Path::new(script_path));

        let abs_output = canonicalize_no_prefix(std::path::Path::new("assets/exports"))
            .join("blender_skinned_diagnostic.json");

        let output = std::process::Command::new(&blender_path)
            .args([
                "--background",
                "--python",
                abs_script.to_str().unwrap(),
                "--",
                abs_export.to_str().unwrap(),
                abs_output.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to run Blender");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("Blender stdout:\n{}", stdout);
        if !stderr.is_empty() {
            eprintln!("Blender stderr:\n{}", stderr);
        }

        assert!(
            output.status.success(),
            "Blender exited with error: {:?}",
            output.status,
        );

        assert!(
            abs_output.exists(),
            "Blender diagnostic JSON not created at {:?}",
            abs_output,
        );

        let json_content =
            std::fs::read_to_string(&abs_output).expect("Failed to read diagnostic JSON");
        let diagnostic: serde_json::Value =
            serde_json::from_str(&json_content).expect("Failed to parse diagnostic JSON");

        let summary = &diagnostic["summary"];

        let total_materials = summary["total_materials"].as_u64().unwrap_or(0);
        eprintln!("Blender imported materials: {}", total_materials);
        assert!(
            total_materials > 0,
            "Blender should import at least 1 material, got {}",
            total_materials,
        );

        let missing_textures = summary["textures_missing"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);
        eprintln!("Missing textures: {}", missing_textures);
        assert_eq!(
            missing_textures, 0,
            "All textures should be found, but {} are missing: {:?}",
            missing_textures, summary["textures_missing"],
        );

        if let Some(mesh_bounds) = diagnostic["mesh_bounds"].as_array() {
            for mb in mesh_bounds {
                let name = mb["name"].as_str().unwrap_or("");
                let bbox_min = &mb["bbox_min"];
                let bbox_max = &mb["bbox_max"];
                eprintln!("Mesh '{}': min={}, max={}", name, bbox_min, bbox_max);

                let max_coord = bbox_max
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    max_coord < 100.0,
                    "Mesh '{}' bbox is too large (max_coord={}), likely wrong scale",
                    name,
                    max_coord,
                );
            }
        }

        std::fs::remove_file(&export_path).ok();
        std::fs::remove_file(&abs_output).ok();
    }

    fn run_blender_import(blender_path: &str, fbx_path: &Path) -> (String, String, bool) {
        let script = r#"
import bpy, sys
argv = sys.argv
idx = argv.index("--") if "--" in argv else len(argv)
fbx_path = argv[idx + 1]
for obj in bpy.data.objects:
    obj.select_set(True)
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=fbx_path)
print("IMPORT_DONE")
"#;

        let temp_script = std::env::temp_dir().join("blender_import_check.py");
        std::fs::write(&temp_script, script).expect("Failed to write temp script");

        let abs_fbx = canonicalize_no_prefix(fbx_path);

        let output = std::process::Command::new(blender_path)
            .args([
                "--background",
                "--python",
                temp_script.to_str().unwrap(),
                "--",
                abs_fbx.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to run Blender");

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        std::fs::remove_file(&temp_script).ok();
        (stdout, stderr, output.status.success())
    }

    fn collect_fbx_import_warnings(stdout: &str) -> Vec<String> {
        stdout
            .lines()
            .filter(|line| {
                let lower = line.to_lowercase();
                lower.starts_with("warning") && lower.contains("layer")
            })
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn test_blender_no_import_warnings_stickman() {
        let blender_path = match read_blender_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipping: BlenderPath not configured");
                return;
            }
        };

        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");
        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("blender_warn_test_stickman.fbx");

        export_full_fbx(&fbx_model, None, &skeleton, &export_path).expect("Failed to export FBX");

        let (stdout, _stderr, success) = run_blender_import(&blender_path, &export_path);
        assert!(success, "Blender exited with error");
        assert!(
            stdout.contains("IMPORT_DONE"),
            "Blender import did not complete",
        );

        let warnings = collect_fbx_import_warnings(&stdout);
        eprintln!("FBX import warnings: {:?}", warnings);
        assert!(
            warnings.is_empty(),
            "Blender FBX import produced warnings:\n{}",
            warnings.join("\n"),
        );

        std::fs::remove_file(&export_path).ok();
    }

    #[test]
    fn test_blender_no_import_warnings_skinned() {
        let blender_path = match read_blender_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipping: BlenderPath not configured");
                return;
            }
        };

        let original_path = "tests/testmodels/fbx/skinning/source/fly.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");
        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("blender_warn_test_skinned.fbx");

        export_full_fbx(&fbx_model, None, &skeleton, &export_path).expect("Failed to export FBX");

        let (stdout, _stderr, success) = run_blender_import(&blender_path, &export_path);
        assert!(success, "Blender exited with error");
        assert!(
            stdout.contains("IMPORT_DONE"),
            "Blender import did not complete",
        );

        let warnings = collect_fbx_import_warnings(&stdout);
        eprintln!("FBX import warnings: {:?}", warnings);
        assert!(
            warnings.is_empty(),
            "Blender FBX import produced warnings:\n{}",
            warnings.join("\n"),
        );

        std::fs::remove_file(&export_path).ok();
    }

    fn load_stickman_for_roundtrip() -> Option<(
        FbxModel,
        thyllore_anim_core::Skeleton,
        thyllore_anim_core::AnimationClip,
    )> {
        let path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(path).exists() {
            return None;
        }

        let fbx_model =
            thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(path).expect("Failed to load FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(path)
                .expect("Failed to load graphics");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton")
            .clone();
        let clip = load_result.clips.first().expect("No clip").clone();

        Some((fbx_model, skeleton, clip))
    }

    fn build_bone_name_map(
        skeleton: &thyllore_anim_core::Skeleton,
    ) -> std::collections::HashMap<u32, String> {
        skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b.name.clone()))
            .collect()
    }

    fn find_rotation_x_curve_for_bone<'a>(
        scene: &'a ufbx::Scene,
        bone_name: &str,
    ) -> Option<&'a ufbx::AnimCurve> {
        if scene.anim_layers.is_empty() {
            return None;
        }

        let layer = &scene.anim_layers[0];
        for ap in &layer.anim_props {
            let target_name = ap.element.name.to_string();
            let prop_name = ap.prop_name.to_string();

            if target_name == bone_name && prop_name == "Lcl Rotation" {
                return ap.anim_value.curves[0].as_ref().map(|r| &**r);
            }
        }

        None
    }

    #[test]
    fn test_weighted_tangent_preserved_on_fbx_roundtrip() {
        let Some((fbx_model, skeleton, clip)) = load_stickman_for_roundtrip() else {
            eprintln!("Skipping: stickman model not found");
            return;
        };

        let bone_names = build_bone_name_map(&skeleton);
        let mut editable = thyllore_anim_core::editable::clip_from_animation(1, &clip, &bone_names);

        let target_bone_name = skeleton.bones[2].name.clone();

        use thyllore_anim_core::editable::{BezierHandle, InterpolationType, TangentWeightMode};

        fn set_weighted_tangents(
            editable: &mut EditableAnimationClip,
            bone_id: u32,
            in_handle: &BezierHandle,
            out_handle: &BezierHandle,
        ) {
            let track = editable
                .tracks
                .get_mut(&bone_id)
                .expect("Bone track not found");
            for kf in &mut track.rotation_x.keyframes {
                kf.interpolation = InterpolationType::Bezier;
                kf.weight_mode = TangentWeightMode::Weighted;
                kf.out_tangent = out_handle.clone();
                kf.in_tangent = in_handle.clone();
            }
        }

        set_weighted_tangents(
            &mut editable,
            2,
            &BezierHandle::new(-0.15, -5.0),
            &BezierHandle::new(0.15, 5.0),
        );

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let weighted_path = export_dir.join("weighted_tangent_roundtrip.fbx");
        let flat_path = export_dir.join("flat_tangent_roundtrip.fbx");

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &weighted_path)
            .expect("Failed to export weighted");

        set_weighted_tangents(
            &mut editable,
            2,
            &BezierHandle::new(-0.15, 0.0),
            &BezierHandle::new(0.15, 0.0),
        );

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &flat_path)
            .expect("Failed to export flat");

        let weighted_scene =
            ufbx::load_file(weighted_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to reload weighted");
        let flat_scene = ufbx::load_file(flat_path.to_str().unwrap(), ufbx::LoadOpts::default())
            .expect("Failed to reload flat");

        let weighted_curve = find_rotation_x_curve_for_bone(&weighted_scene, &target_bone_name)
            .expect("Weighted curve not found");
        let flat_curve = find_rotation_x_curve_for_bone(&flat_scene, &target_bone_name)
            .expect("Flat curve not found");

        assert!(
            weighted_curve.keyframes.len() >= 2,
            "Weighted curve should have keyframes"
        );
        assert_eq!(
            weighted_curve.keyframes.len(),
            flat_curve.keyframes.len(),
            "Both exports should have the same number of keyframes"
        );

        let has_cubic = weighted_curve
            .keyframes
            .iter()
            .any(|kf| kf.interpolation == ufbx::Interpolation::Cubic);
        assert!(
            has_cubic,
            "Re-imported curve should have cubic interpolation"
        );

        let mut max_shape_diff: f64 = 0.0;
        let duration = editable.duration as f64;
        for i in 1..10 {
            let t = i as f64 * duration / 10.0;
            let weighted_val = ufbx::evaluate_curve(weighted_curve, t, 0.0);
            let flat_val = ufbx::evaluate_curve(flat_curve, t, 0.0);
            let diff = (weighted_val - flat_val).abs();
            if diff > max_shape_diff {
                max_shape_diff = diff;
            }
        }

        assert!(
            max_shape_diff > 0.5,
            "Weighted tangent handles should produce a different curve shape than flat handles, max_diff={:.4}",
            max_shape_diff
        );

        let has_nonzero_tangent = weighted_curve.keyframes.iter().any(|kf| {
            kf.right.dx.abs() > 1e-6
                || kf.right.dy.abs() > 1e-6
                || kf.left.dx.abs() > 1e-6
                || kf.left.dy.abs() > 1e-6
        });
        assert!(
            has_nonzero_tangent,
            "Weighted tangent should have at least one non-zero tangent"
        );

        std::fs::remove_file(&weighted_path).ok();
        std::fs::remove_file(&flat_path).ok();
    }

    #[test]
    fn test_non_weighted_tangent_unchanged_on_fbx_roundtrip() {
        let Some((fbx_model, skeleton, clip)) = load_stickman_for_roundtrip() else {
            eprintln!("Skipping: stickman model not found");
            return;
        };

        let bone_names = build_bone_name_map(&skeleton);
        let mut editable_a =
            thyllore_anim_core::editable::clip_from_animation(1, &clip, &bone_names);
        let mut editable_b =
            thyllore_anim_core::editable::clip_from_animation(2, &clip, &bone_names);

        let target_bone_name = skeleton.bones[2].name.clone();

        use thyllore_anim_core::editable::{
            curve_recalculate_auto_tangents, InterpolationType, TangentWeightMode,
        };

        for editable in [&mut editable_a, &mut editable_b] {
            let track = editable.tracks.get_mut(&2).expect("Bone track not found");
            for kf in &mut track.rotation_x.keyframes {
                kf.interpolation = InterpolationType::Bezier;
                kf.weight_mode = TangentWeightMode::NonWeighted;
            }
            curve_recalculate_auto_tangents(&mut track.rotation_x);
        }

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let path_a = export_dir.join("non_weighted_roundtrip_a.fbx");
        let path_b = export_dir.join("non_weighted_roundtrip_b.fbx");

        export_full_fbx(&fbx_model, Some(&editable_a), &skeleton, &path_a)
            .expect("Failed to export A");
        export_full_fbx(&fbx_model, Some(&editable_b), &skeleton, &path_b)
            .expect("Failed to export B");

        let scene_a = ufbx::load_file(path_a.to_str().unwrap(), ufbx::LoadOpts::default())
            .expect("Failed to reload A");
        let scene_b = ufbx::load_file(path_b.to_str().unwrap(), ufbx::LoadOpts::default())
            .expect("Failed to reload B");

        let curve_a =
            find_rotation_x_curve_for_bone(&scene_a, &target_bone_name).expect("Curve A not found");
        let curve_b =
            find_rotation_x_curve_for_bone(&scene_b, &target_bone_name).expect("Curve B not found");

        assert!(
            curve_a.keyframes.len() >= 2,
            "Re-imported curve should have keyframes"
        );
        assert_eq!(
            curve_a.keyframes.len(),
            curve_b.keyframes.len(),
            "Both exports should have the same number of keyframes"
        );

        let duration = editable_a.duration as f64;
        for i in 0..=10 {
            let t = i as f64 * duration / 10.0;
            let val_a = ufbx::evaluate_curve(curve_a, t, 0.0);
            let val_b = ufbx::evaluate_curve(curve_b, t, 0.0);
            let diff = (val_a - val_b).abs();
            assert!(
                diff < 1e-4,
                "Non-weighted tangent exports should be identical at t={:.2}: a={:.4}, b={:.4}, diff={:.6}",
                t, val_a, val_b, diff
            );
        }

        for (kf_a, kf_b) in curve_a.keyframes.iter().zip(curve_b.keyframes.iter()) {
            assert!(
                (kf_a.right.dx - kf_b.right.dx).abs() < 1e-4
                    && (kf_a.right.dy - kf_b.right.dy).abs() < 1e-4,
                "Non-weighted tangent data should be identical: a=({}, {}), b=({}, {})",
                kf_a.right.dx,
                kf_a.right.dy,
                kf_b.right.dx,
                kf_b.right.dy
            );
        }

        std::fs::remove_file(&path_a).ok();
        std::fs::remove_file(&path_b).ok();
    }
}
