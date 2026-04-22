mod buffer_handle;
mod ubo;

pub use buffer_handle::{BufferHandle, IndexBufferHandle, VertexBufferHandle};
pub use ubo::{FrameUBO, MaterialUBO, ObjectUBO};

pub type MeshId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferMemoryType {
    DeviceLocal,
    HostVisible,
}
