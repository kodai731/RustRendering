use anyhow::{anyhow, Result};

use openusd::schemas::geom::{BasisCurves, Curves, PointBased};
use openusd::sdf;
use openusd::usd::{PrimPredicate, Stage};

#[derive(Clone, Debug)]
pub struct UsdCurveData {
    pub name: String,
    pub points: Vec<[f32; 3]>,
    pub curve_vertex_counts: Vec<u32>,
    pub widths: Vec<f32>,
    pub is_linear: bool,
}

impl UsdCurveData {
    pub fn curve_count(&self) -> usize {
        self.curve_vertex_counts.len()
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }
}

pub(crate) fn collect_curves(stage: &Stage) -> Result<Vec<UsdCurveData>> {
    let mut prim_paths = Vec::new();
    stage.traverse(PrimPredicate::DEFAULT_PROXIES, |p| {
        prim_paths.push(p.clone())
    })?;

    let mut result = Vec::new();
    for path in &prim_paths {
        let Some(basis_curves) = BasisCurves::get(stage, path.clone())? else {
            continue;
        };
        if let Some(curve) = build_curve(&basis_curves, path)? {
            result.push(curve);
        }
    }
    Ok(result)
}

fn build_curve(basis_curves: &BasisCurves, path: &sdf::Path) -> Result<Option<UsdCurveData>> {
    let points = read_vec3f(basis_curves.points_attr().get::<sdf::Value>()?);
    let counts = read_int(
        basis_curves
            .curve_vertex_counts_attr()
            .get::<sdf::Value>()?,
    );
    if points.is_empty() || counts.is_empty() {
        return Ok(None);
    }

    let expected_points: i64 = counts.iter().map(|&c| c.max(0) as i64).sum();
    if expected_points as usize != points.len() {
        return Err(anyhow!(
            "BasisCurves {} curveVertexCounts sum {} does not match points {}",
            path.as_str(),
            expected_points,
            points.len()
        ));
    }

    let widths = basis_curves
        .widths_attr()
        .get::<Vec<f32>>()
        .ok()
        .flatten()
        .unwrap_or_default();
    let is_linear = basis_curves
        .type_attr()
        .get::<String>()
        .ok()
        .flatten()
        .map(|token| token == "linear")
        .unwrap_or(true);

    Ok(Some(UsdCurveData {
        name: leaf_name(path),
        points,
        curve_vertex_counts: counts.into_iter().map(|c| c.max(0) as u32).collect(),
        widths,
        is_linear,
    }))
}

fn read_vec3f(value: Option<sdf::Value>) -> Vec<[f32; 3]> {
    value
        .and_then(|v| v.try_as_vec_3f_vec())
        .map(|points| points.into_iter().map(|p| [p.x, p.y, p.z]).collect())
        .unwrap_or_default()
}

fn read_int(value: Option<sdf::Value>) -> Vec<i32> {
    value.and_then(|v| v.try_as_int_vec()).unwrap_or_default()
}

fn leaf_name(path: &sdf::Path) -> String {
    path.as_str()
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("usd_curve")
        .to_string()
}
