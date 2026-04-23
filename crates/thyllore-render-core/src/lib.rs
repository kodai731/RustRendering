mod buffer_handle;
mod gizmo;
mod gizmo_data;
mod mesh;
mod render_data;
mod ubo;

pub use buffer_handle::{BufferHandle, IndexBufferHandle, VertexBufferHandle};
pub use gizmo::{
    BoneDisplayStyle, ColorVertex, GizmoAxis, GizmoDraggable, GizmoPosition, GizmoSelectable,
    TransformGizmoHandle,
};
pub use gizmo_data::{ConstraintGizmoData, GridMeshData, LightGizmoData};
pub use mesh::{DynamicMesh, GpuMeshRef, LineMesh, MeshScale, RenderInfo};
pub use render_data::{MeshHandle, ObjectIndex, RenderData, SkeletonHandle};
pub use ubo::{FrameUBO, MaterialUBO, ObjectUBO};

pub type MeshId = usize;
pub type PipelineId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferMemoryType {
    DeviceLocal,
    HostVisible,
}
