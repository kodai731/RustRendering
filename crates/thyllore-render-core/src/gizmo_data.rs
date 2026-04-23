use crate::{GizmoDraggable, GizmoPosition, GizmoSelectable, LineMesh, MeshScale, RenderInfo};

#[derive(Clone, Debug, Default)]
pub struct GridMeshData {
    pub mesh: LineMesh,
    pub render_info: RenderInfo,
    pub scale: MeshScale,
    pub show_y_axis_grid: bool,
    pub xz_only_index_count: u32,
}

#[derive(Clone, Debug)]
pub struct ConstraintGizmoData {
    pub visible: bool,
    pub wire_mesh: LineMesh,
    pub wire_render_info: RenderInfo,
}

impl Default for ConstraintGizmoData {
    fn default() -> Self {
        Self {
            visible: false,
            wire_mesh: LineMesh::default(),
            wire_render_info: RenderInfo::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LightGizmoData {
    pub mesh: LineMesh,
    pub render_info: RenderInfo,
    pub position: GizmoPosition,
    pub selectable: GizmoSelectable,
    pub draggable: GizmoDraggable,
    pub drag_active: bool,
    pub ray_to_model: LineMesh,
    pub vertical_lines: LineMesh,
}
