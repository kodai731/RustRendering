use std::path::{Path, PathBuf};

use thyllore_importer_core::fbx::fbx::FbxData;

use crate::components::fbx::*;
use crate::fbx_animation::{decompose_matrix_to_trs, UidAllocator};

pub(crate) fn build_mesh_model_exports(
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

pub(crate) fn resolve_mesh_parent_uids(
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

pub(crate) fn build_material_exports(
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

pub(crate) fn compute_relative_path(from_dir: &Path, to_path: &Path) -> String {
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

pub(crate) fn canonicalize_clean(path: &Path) -> PathBuf {
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

pub(crate) fn resolve_texture_for_export(texture_path: &str, model_path: Option<&str>) -> PathBuf {
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

pub(crate) fn build_texture_exports(
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
