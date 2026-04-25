use crate::ecs::component::RenderInfo;
use crate::vulkanr::descriptor::RRBillboardDescriptorSet;
use crate::vulkanr::image::RRImage;

pub use thyllore_render_core::{BillboardMesh, BillboardTransform, BillboardVertex};

#[derive(Clone, Debug, Default)]
pub struct BillboardRenderState {
    pub descriptor_set: RRBillboardDescriptorSet,
    pub texture: Option<RRImage>,
}

#[derive(Clone, Debug, Default)]
pub struct BillboardData {
    pub mesh: BillboardMesh,
    pub transform: Option<BillboardTransform>,
    pub render_info: RenderInfo,
    pub render_state: BillboardRenderState,
}

impl BillboardData {
    pub fn transform(&self) -> Option<&BillboardTransform> {
        self.transform.as_ref()
    }

    pub fn transform_mut(&mut self) -> &mut Option<BillboardTransform> {
        &mut self.transform
    }

    pub fn vertices(&self) -> &[BillboardVertex] {
        &self.mesh.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.mesh.indices
    }
}
