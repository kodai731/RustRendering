use cgmath::Matrix4;

use thyllore_importer_core::fbx::fbx::FbxData;

use crate::components::fbx::*;
use crate::fbx_animation::UidAllocator;

pub(crate) fn build_skin_exports(
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

pub(crate) fn matrix4_to_flat_f64_scaled(m: &Matrix4<f32>, inv_unit_scale: f32) -> [f64; 16] {
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
