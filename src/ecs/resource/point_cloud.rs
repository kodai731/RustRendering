use thyllore_render_core::{LineMesh, RenderInfo};

/// Point geometry for imported point clouds (USD `UsdGeomPoints` and
/// surface-less `UsdGeomMesh`).
///
/// Built once per model load from `ModelLoadResult.points`, rendered with the
/// POINT_LIST pipeline. Reuses `LineMesh` (`DynamicMesh<ColorVertex>`) for
/// storage — point and line geometry share the vertex format and differ only in
/// primitive topology. The object slot is reserved at init so it survives model
/// reloads; the model matrix is identity because points share the mesh space.
#[derive(Default)]
pub struct PointCloudData {
    pub mesh: LineMesh,
    pub render_info: RenderInfo,
    pub visible: bool,
    pub dirty: bool,
}
