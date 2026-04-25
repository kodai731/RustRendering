use cgmath::Vector3;

use crate::ecs::component::mesh::MeshData;
use crate::ecs::resource::MeshAssets;

pub use thyllore_model_core::mesh::{compute_vertex_layout, create_interleaved_buffer};

pub fn mesh_calculate_model_bounds(
    assets: &MeshAssets,
) -> Option<(Vector3<f32>, Vector3<f32>, Vector3<f32>)> {
    if assets.meshes.is_empty() {
        return None;
    }

    let mut min = Vector3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max = Vector3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut has_vertices = false;

    for mesh in &assets.meshes {
        for vertex in &mesh.vertex_data.vertices {
            has_vertices = true;
            min.x = min.x.min(vertex.pos.x);
            min.y = min.y.min(vertex.pos.y);
            min.z = min.z.min(vertex.pos.z);
            max.x = max.x.max(vertex.pos.x);
            max.y = max.y.max(vertex.pos.y);
            max.z = max.z.max(vertex.pos.z);
        }
    }

    if !has_vertices {
        return None;
    }

    let center = Vector3::new(
        (min.x + max.x) * 0.5,
        (min.y + max.y) * 0.5,
        (min.z + max.z) * 0.5,
    );

    Some((min, max, center))
}

pub fn validate_mesh_data(mesh: &MeshData) -> Result<(), String> {
    if mesh.vertex_count() == 0 && mesh.attribute_ids().next().is_none() {
        return Err("MeshData has no attributes".to_string());
    }

    let vertex_count = mesh.vertex_count();
    for id in mesh.attribute_ids() {
        if let Some(values) = mesh.attribute(*id) {
            if values.len() != vertex_count {
                return Err(format!(
                    "Attribute {:?} has {} vertices, expected {}",
                    id,
                    values.len(),
                    vertex_count
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::component::mesh::presets::{COLOR, POSITION};
    use crate::ecs::component::mesh::{MeshData, PrimitiveTopology};

    #[test]
    fn test_interleaved_buffer_creation() {
        let mesh = MeshData::new(PrimitiveTopology::TriangleList)
            .with_inserted_attribute(POSITION, vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            .with_inserted_attribute(COLOR, vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);

        let layout = compute_vertex_layout(&mesh);
        assert_eq!(layout.stride, 24);

        let buffer = create_interleaved_buffer(&mesh, &layout);
        assert_eq!(buffer.len(), 48);
    }
}
