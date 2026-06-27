use thyllore_render_core::{LineMesh, RenderInfo};

/// Line geometry for imported curves (USD `BasisCurves`).
///
/// Built once per model load from `ModelLoadResult.curves`, rendered with the
/// shared LINE_LIST pipeline (same as the grid). The object slot is reserved at
/// init so it survives model reloads. The render model matrix comes from a
/// `CurveMeshRef` child entity of the model root, so the curves follow the same
/// `GlobalTransform` propagation (including USD up-axis rotation) as the meshes.
#[derive(Default)]
pub struct CurveMeshData {
    pub mesh: LineMesh,
    pub render_info: RenderInfo,
    pub visible: bool,
    pub dirty: bool,
}
