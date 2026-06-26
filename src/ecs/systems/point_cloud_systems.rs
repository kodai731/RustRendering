use thyllore_importer_core::UsdPointData;

use crate::ecs::component::{ColorVertex, LineMesh};

const POINT_COLOR: [f32; 3] = [0.85, 0.85, 0.9];

/// Builds a `POINT_LIST` mesh from imported point clouds.
///
/// Each control point becomes one vertex; indices are the identity sequence
/// `0..N` (one index per point).
pub fn build_point_cloud_mesh(clouds: &[UsdPointData], mesh: &mut LineMesh) {
    mesh.vertices.clear();
    mesh.indices.clear();

    for cloud in clouds {
        for position in &cloud.points {
            let index = mesh.vertices.len() as u32;
            mesh.vertices.push(ColorVertex {
                pos: *position,
                color: POINT_COLOR,
            });
            mesh.indices.push(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(points: Vec<[f32; 3]>) -> UsdPointData {
        UsdPointData {
            name: "test".to_string(),
            points,
            widths: Vec::new(),
        }
    }

    #[test]
    fn builds_one_vertex_per_point() {
        let clouds = vec![
            cloud(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            cloud(vec![[0.0, 2.0, 0.0]]),
        ];
        let mut mesh = LineMesh::default();
        build_point_cloud_mesh(&clouds, &mut mesh);

        assert_eq!(mesh.vertices.len(), 3, "2 + 1 points");
        assert_eq!(mesh.indices, vec![0, 1, 2], "identity index per point");
        assert_eq!(mesh.vertices[2].pos, [0.0, 2.0, 0.0]);
    }

    #[test]
    fn empty_input_produces_empty_mesh() {
        let mut mesh = LineMesh::default();
        build_point_cloud_mesh(&[], &mut mesh);
        assert!(mesh.vertices.is_empty());
        assert!(mesh.indices.is_empty());
    }
}
