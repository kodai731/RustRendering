use thyllore_importer_core::fbx::fbx::FbxData;

use crate::components::fbx::FbxGeometryExport;
use crate::fbx_animation::UidAllocator;

pub(crate) fn build_geometry_exports(
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

pub(crate) fn convert_positions_to_fbx(fbx_data: &FbxData, inv_unit_scale: f32) -> Vec<f64> {
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

pub(crate) fn convert_normals_to_fbx(fbx_data: &FbxData) -> Vec<f64> {
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

pub(crate) fn convert_uvs_to_fbx(fbx_data: &FbxData) -> Vec<f64> {
    fbx_data
        .tex_coords
        .iter()
        .flat_map(|uv| [uv[0] as f64, (1.0 - uv[1]) as f64])
        .collect()
}

pub(crate) fn encode_triangle_polygon_indices(indices: &[u32]) -> Vec<i32> {
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
