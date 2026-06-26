use anyhow::Result;

use openusd::schemas::geom::{Mesh, PointBased, Points};
use openusd::sdf;
use openusd::usd::{PrimPredicate, Stage};

#[derive(Clone, Debug)]
pub struct UsdPointData {
    pub name: String,
    pub points: Vec<[f32; 3]>,
    pub widths: Vec<f32>,
}

impl UsdPointData {
    pub fn point_count(&self) -> usize {
        self.points.len()
    }
}

/// Collects point-cloud geometry from `UsdGeomPoints` prims and from
/// surface-less `UsdGeomMesh` prims (empty `faceVertexCounts`).
///
/// A faceless mesh carries only `points` — USD `Mesh` has no edge attribute, so
/// any line connectivity is already lost upstream and the only faithful
/// representation is the point set.
pub(crate) fn collect_points(stage: &Stage) -> Result<Vec<UsdPointData>> {
    let mut prim_paths = Vec::new();
    stage.traverse(PrimPredicate::DEFAULT_PROXIES, |p| {
        prim_paths.push(p.clone())
    })?;

    let mut result = Vec::new();
    for path in &prim_paths {
        if let Some(points) = Points::get(stage, path.clone())? {
            let positions = read_vec3f(points.points_attr().get::<sdf::Value>()?);
            if positions.is_empty() {
                continue;
            }
            let widths = points
                .widths_attr()
                .get::<Vec<f32>>()
                .ok()
                .flatten()
                .unwrap_or_default();
            result.push(UsdPointData {
                name: leaf_name(path),
                points: positions,
                widths,
            });
            continue;
        }

        if let Some(mesh) = Mesh::get(stage, path.clone())? {
            if mesh_has_faces(&mesh) {
                continue;
            }
            let positions = read_vec3f(mesh.points_attr().get::<sdf::Value>()?);
            if positions.is_empty() {
                continue;
            }
            result.push(UsdPointData {
                name: leaf_name(path),
                points: positions,
                widths: Vec::new(),
            });
        }
    }
    Ok(result)
}

pub(crate) fn mesh_has_faces(mesh: &Mesh) -> bool {
    mesh.face_vertex_counts_attr()
        .get::<sdf::Value>()
        .ok()
        .flatten()
        .and_then(|v| v.try_as_int_vec())
        .map(|counts| !counts.is_empty())
        .unwrap_or(false)
}

fn read_vec3f(value: Option<sdf::Value>) -> Vec<[f32; 3]> {
    value
        .and_then(|v| v.try_as_vec_3f_vec())
        .map(|points| points.into_iter().map(|p| [p.x, p.y, p.z]).collect())
        .unwrap_or_default()
}

fn leaf_name(path: &sdf::Path) -> String {
    path.as_str()
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("usd_points")
        .to_string()
}
