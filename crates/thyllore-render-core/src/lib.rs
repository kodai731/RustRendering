pub mod backend;
mod billboard;
mod buffer_handle;
mod gizmo;
mod gizmo_data;
mod mesh;
mod projection;
mod render_data;
mod settings;
mod ubo;

pub use backend::RenderBackend;
pub use billboard::{BillboardMesh, BillboardTransform, BillboardVertex};
pub use buffer_handle::{BufferHandle, IndexBufferHandle, VertexBufferHandle};
pub use gizmo::{
    BoneDisplayStyle, ColorVertex, GizmoAxis, GizmoDraggable, GizmoPosition, GizmoSelectable,
    TransformGizmoHandle,
};
pub use gizmo_data::{BoneGizmoData, ConstraintGizmoData, GridMeshData, LightGizmoData};
pub use mesh::{DynamicMesh, GpuMeshRef, LineMesh, MeshScale, RenderInfo, FRAMES_IN_FLIGHT};
pub use projection::{DistanceAttenuation, ProjectionData};
pub use render_data::{MeshHandle, ObjectIndex, RenderData, SkeletonHandle};
pub use settings::{
    AutoExposure, BloomSettings, DepthOfField, Exposure, LensEffects, PhysicalCameraParameters,
    ToneMapOperator, ToneMapping,
};
pub use ubo::{FrameUBO, MaterialUBO, ObjectUBO};

pub type MeshId = usize;
pub type PipelineId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferMemoryType {
    DeviceLocal,
    HostVisible,
}
