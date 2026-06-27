mod attribute;
mod interleave;
mod mesh_data;
mod normals;
mod subdivision;
mod values;
mod vertex;

pub use attribute::{VertexAttribute, VertexAttributeId, VertexFormat};
pub use interleave::{compute_vertex_layout, create_interleaved_buffer, VertexLayout};
pub use mesh_data::{MeshData, PrimitiveTopology};
pub use normals::compute_smooth_normals;
pub use subdivision::{
    catmull_clark, triangulate_subdivided, Blendable, PolygonMesh, SparseWeights, SubdividedMesh,
};
pub use values::VertexAttributeValues;
pub use vertex::{Vertex, VertexData};
