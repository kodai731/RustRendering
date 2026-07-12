use crate::{IndexBufferHandle, PipelineId, VertexBufferHandle};

#[derive(Clone, Debug, Default)]
pub struct DynamicMesh<V> {
    pub vertices: Vec<V>,
    pub indices: Vec<u32>,
    pub vertex_buffer_handle: VertexBufferHandle,
    pub index_buffer_handle: IndexBufferHandle,
}

impl<V> DynamicMesh<V> {
    pub fn new() -> Self
    where
        V: Default,
    {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer_handle: VertexBufferHandle::default(),
            index_buffer_handle: IndexBufferHandle::default(),
        }
    }

    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuMeshRef {
    pub vertex_buffer_handle: VertexBufferHandle,
    pub index_buffer_handle: IndexBufferHandle,
    pub index_count: u32,
}

impl GpuMeshRef {
    pub fn new(
        vertex_buffer_handle: VertexBufferHandle,
        index_buffer_handle: IndexBufferHandle,
        index_count: u32,
    ) -> Self {
        Self {
            vertex_buffer_handle,
            index_buffer_handle,
            index_count,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderInfo {
    pub pipeline_id: Option<PipelineId>,
    pub object_index: usize,
}

impl RenderInfo {
    pub fn new(pipeline_id: Option<PipelineId>, object_index: usize) -> Self {
        Self {
            pipeline_id,
            object_index,
        }
    }
}

pub type LineMesh = DynamicMesh<crate::ColorVertex>;

#[derive(Clone, Copy, Debug, Default)]
pub struct MeshScale(pub f32);

impl MeshScale {
    pub fn new(scale: f32) -> Self {
        Self(scale)
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}
