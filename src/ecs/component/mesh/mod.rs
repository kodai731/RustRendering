mod gpu_mesh;
pub mod presets;

pub use gpu_mesh::{DynamicMesh, GpuMeshRef, LineMesh, MeshScale, RenderInfo};
pub use thyllore_model_core::mesh::{
    MeshData, PrimitiveTopology, VertexAttribute, VertexAttributeId, VertexAttributeValues,
    VertexFormat, VertexLayout,
};
