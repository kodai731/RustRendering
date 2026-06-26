use thyllore_render_core::{LineMesh, RenderInfo};

/// Line geometry for imported curves (USD `BasisCurves`).
///
/// Built once per model load from `ModelLoadResult.curves`, rendered with the
/// shared LINE_LIST pipeline (same as the grid). The object slot is reserved at
/// init so it survives model reloads; the model matrix is identity because
/// curve control points share the mesh coordinate space.
#[derive(Default)]
pub struct CurveMeshData {
    pub mesh: LineMesh,
    pub render_info: RenderInfo,
    pub visible: bool,
    pub dirty: bool,
}
