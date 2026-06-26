use thyllore_importer_core::UsdCurveData;

use crate::ecs::component::{ColorVertex, LineMesh};

const CURVE_LINE_COLOR: [f32; 3] = [0.8, 0.8, 0.85];

/// Builds a `LINE_LIST` mesh from imported curves.
///
/// Each curve's control points (partitioned by `curve_vertex_counts`) become a
/// polyline: consecutive points form one line segment (two vertices, two
/// indices). Cubic curves are approximated by their dense control points.
pub fn build_curve_line_mesh(curves: &[UsdCurveData], mesh: &mut LineMesh) {
    mesh.vertices.clear();
    mesh.indices.clear();

    for curve in curves {
        let mut point_offset = 0usize;
        for &vertex_count in &curve.curve_vertex_counts {
            let count = vertex_count as usize;
            for segment in 0..count.saturating_sub(1) {
                let start = point_offset + segment;
                let (Some(&from), Some(&to)) =
                    (curve.points.get(start), curve.points.get(start + 1))
                else {
                    continue;
                };

                let base = mesh.vertices.len() as u32;
                mesh.vertices.push(ColorVertex {
                    pos: from,
                    color: CURVE_LINE_COLOR,
                });
                mesh.vertices.push(ColorVertex {
                    pos: to,
                    color: CURVE_LINE_COLOR,
                });
                mesh.indices.push(base);
                mesh.indices.push(base + 1);
            }
            point_offset += count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(points: Vec<[f32; 3]>, counts: Vec<u32>) -> UsdCurveData {
        UsdCurveData {
            name: "test".to_string(),
            points,
            curve_vertex_counts: counts,
            widths: Vec::new(),
            is_linear: true,
        }
    }

    #[test]
    fn builds_line_segments_per_curve() {
        // Two curves: 3 points (2 segments) and 2 points (1 segment) = 3 segments.
        let s = curve(
            vec![
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            vec![3, 2],
        );

        let mut mesh = LineMesh::default();
        build_curve_line_mesh(&[s], &mut mesh);

        assert_eq!(mesh.indices.len(), 6, "3 segments x 2 indices");
        assert_eq!(mesh.vertices.len(), 6, "3 segments x 2 vertices");
        assert_eq!(mesh.indices, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(mesh.vertices[0].pos, [0.0, 0.0, 0.0]);
        assert_eq!(mesh.vertices[5].pos, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn single_point_curve_produces_no_segment() {
        let s = curve(vec![[0.0, 0.0, 0.0]], vec![1]);
        let mut mesh = LineMesh::default();
        build_curve_line_mesh(&[s], &mut mesh);
        assert!(mesh.indices.is_empty());
        assert!(mesh.vertices.is_empty());
    }
}
