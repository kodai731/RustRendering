//! Verifies geometry classification routes UsdGeomPoints and surface-less
//! UsdGeomMesh prims into the point-cloud representation, while a faced mesh
//! stays a surface mesh.
//!
//! Uses inline `.usda` so the test is deterministic and machine independent.

use std::io::Write;

use thyllore_importer_core::usd::load_usd_file;

const SCENE_USDA: &str = r#"#usda 1.0
(
    defaultPrim = "root"
    upAxis = "Y"
)

def Xform "root"
{
    def Mesh "surface"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }

    def Mesh "faceless"
    {
        int[] faceVertexCounts = []
        int[] faceVertexIndices = []
        point3f[] points = [(0, 0, 0), (0, 1, 0), (0, 2, 0)]
    }

    def Points "cloud"
    {
        point3f[] points = [(5, 0, 0), (5, 1, 0)]
        float[] widths = [0.2, 0.2]
    }
}
"#;

fn write_temp_usda(name: &str, contents: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(name);
    let mut file = std::fs::File::create(&path).expect("create temp usda");
    file.write_all(contents.as_bytes()).expect("write usda");
    path
}

#[test]
fn routes_faceless_mesh_and_points_to_point_clouds() {
    let path = write_temp_usda("thyllore_points.usda", SCENE_USDA);
    let result = load_usd_file(path.to_str().unwrap()).expect("USD import should succeed");

    // Only the faced mesh is a surface.
    assert_eq!(result.meshes.len(), 1, "one faced surface mesh");
    assert!(
        !result.meshes[0].vertex_data.vertices.is_empty(),
        "surface mesh must not be empty"
    );

    // The faceless mesh and the UsdGeomPoints prim both become point clouds.
    assert_eq!(result.points.len(), 2, "faceless mesh + Points prim");
    let total: usize = result.points.iter().map(|p| p.point_count()).sum();
    assert_eq!(total, 3 + 2, "3 faceless points + 2 cloud points");

    // The UsdGeomPoints widths survive.
    let has_widths = result.points.iter().any(|p| p.widths.len() == 2);
    assert!(has_widths, "UsdGeomPoints widths should be imported");

    let _ = std::fs::remove_file(&path);
}
