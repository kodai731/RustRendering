pub use thyllore_render_core::PipelineId;

#[derive(Clone, Debug)]
pub struct PipelineRef(pub PipelineId);

impl PipelineRef {
    pub fn new(id: PipelineId) -> Self {
        Self(id)
    }

    pub fn id(&self) -> PipelineId {
        self.0
    }
}

#[derive(Default, Debug)]
pub struct PipelineManager {
    count: usize,
}

impl PipelineManager {
    pub fn new() -> Self {
        Self { count: 0 }
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

pub fn pipeline_allocate_id(manager: &mut PipelineManager) -> PipelineId {
    let id = manager.count;
    manager.count += 1;
    id
}
