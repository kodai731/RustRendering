use crate::core::device::RRDevice;
use crate::pipeline::RRPipeline;
use crate::resource::GpuResource;
use thyllore_render_core::PipelineId;

#[derive(Default)]
pub struct PipelineStorage {
    pipelines: Vec<RRPipeline>,
}

impl PipelineStorage {
    pub fn new() -> Self {
        Self {
            pipelines: Vec::new(),
        }
    }

    pub fn register(&mut self, pipeline: RRPipeline) -> PipelineId {
        let id = self.pipelines.len();
        self.pipelines.push(pipeline);
        id
    }

    pub fn get(&self, id: PipelineId) -> Option<&RRPipeline> {
        self.pipelines.get(id)
    }

    pub fn get_mut(&mut self, id: PipelineId) -> Option<&mut RRPipeline> {
        self.pipelines.get_mut(id)
    }

    pub fn count(&self) -> usize {
        self.pipelines.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (PipelineId, &RRPipeline)> {
        self.pipelines.iter().enumerate()
    }

    pub unsafe fn destroy_all(&mut self, rrdevice: &RRDevice) {
        for pipeline in self.pipelines.drain(..) {
            pipeline.destroy(&rrdevice.device);
        }
    }
}

impl GpuResource for PipelineStorage {
    unsafe fn destroy_gpu(&mut self, rrdevice: &RRDevice) {
        self.destroy_all(rrdevice);
    }

    fn resource_name(&self) -> &'static str {
        "PipelineStorage"
    }
}

impl std::fmt::Debug for PipelineStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineStorage")
            .field("pipeline_count", &self.pipelines.len())
            .finish()
    }
}
