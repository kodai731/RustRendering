use crate::core::device::RRDevice;
use crate::resource::{GpuBufferRegistry, GraphicsResources, PipelineStorage};

pub struct FrameRenderContext<'a> {
    pub device: &'a RRDevice,
    pub graphics: &'a GraphicsResources,
    pub buffers: &'a GpuBufferRegistry,
    pub pipelines: &'a PipelineStorage,
    pub image_index: usize,
}
