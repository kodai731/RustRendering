//! Verifies UsdGeomBasisCurves import into the internal strand representation.
//!
//! Uses a small inline `.usda` so the test is deterministic and machine
//! independent (no external asset required).

use std::io::Write;

use thyllore_importer_core::usd::load_usd_file;

const HAIR_USDA: &str = r#"#usda 1.0
(
    defaultPrim = "hair"
    upAxis = "Y"
    metersPerUnit = 1
)

def BasisCurves "hair"
{
    int[] curveVertexCounts = [3, 2]
    point3f[] points = [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 0)]
    float[] widths = [0.1, 0.1, 0.1, 0.1, 0.1] (interpolation = "vertex")
    uniform token type = "linear"
    uniform token wrap = "nonperiodic"
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
fn imports_basis_curves_as_strands() {
    let path = write_temp_usda("thyllore_hair_strands.usda", HAIR_USDA);
    let result = load_usd_file(path.to_str().unwrap()).expect("USD import should succeed");

    assert_eq!(result.strands.len(), 1, "expected one BasisCurves prim");
    let strand = &result.strands[0];

    assert_eq!(strand.curve_count(), 2, "two curves partition the points");
    assert_eq!(strand.curve_vertex_counts, vec![3, 2]);
    assert_eq!(strand.point_count(), 5, "sum of curveVertexCounts");
    assert_eq!(
        strand.curve_vertex_counts.iter().sum::<u32>() as usize,
        strand.point_count(),
        "counts must partition the point buffer exactly"
    );
    assert!(strand.is_linear, "type was authored as linear");
    assert_eq!(strand.widths.len(), 5, "per-vertex widths");
    assert_eq!(strand.points[0], [0.0, 0.0, 0.0]);
    assert_eq!(strand.points[4], [1.0, 1.0, 0.0]);

    let _ = std::fs::remove_file(&path);
}
