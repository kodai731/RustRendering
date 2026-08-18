use anyhow::{anyhow, Result};

use crate::descriptor::auto_exposure::{
    RRAutoExposureAverageDescriptorSet, RRAutoExposureHistogramDescriptorSet,
};
use crate::descriptor::billboard::RRBillboardDescriptorSet;
use crate::descriptor::bloom::RRBloomDescriptorSets;
use crate::descriptor::composite::RRCompositeDescriptorSet;
use crate::descriptor::dof::RRDofDescriptorSet;
use crate::descriptor::flame::RRFlameDescriptorSet;
use crate::descriptor::frame::FrameDescriptorSet;
use crate::descriptor::imgui_layout::imgui_layout_spec;
use crate::descriptor::material::MaterialManager;
use crate::descriptor::object::ObjectDescriptorSet;
use crate::descriptor::onion_skin::onion_skin_composite_layout_spec;
use crate::descriptor::pass_manifest::{PassId, PassShaders, SetRole};
use crate::descriptor::ray_query::RRRayQueryDescriptorSet;
use crate::descriptor::reflected_layout::ReflectedLayoutSpec;
use crate::descriptor::tonemap::RRToneMapDescriptorSet;

pub fn layout_spec_for_role(pass: &PassShaders, role: SetRole) -> Result<ReflectedLayoutSpec> {
    match role {
        SetRole::Frame => Ok(FrameDescriptorSet::layout_spec()),
        SetRole::Material => Ok(MaterialManager::layout_spec()),
        SetRole::Object => Ok(ObjectDescriptorSet::layout_spec()),
        SetRole::Local => local_layout_spec(pass.id).ok_or_else(|| {
            anyhow!(
                "pass `{}` declares a local descriptor set but descriptor/pass_layouts.rs maps no layout for it",
                pass.name()
            )
        }),
    }
}

pub fn layout_specs(pass: &PassShaders) -> Result<Vec<(u32, ReflectedLayoutSpec)>> {
    pass.set_roles
        .iter()
        .map(|(set, role)| Ok((*set, layout_spec_for_role(pass, *role)?)))
        .collect()
}

fn local_layout_spec(pass: PassId) -> Option<ReflectedLayoutSpec> {
    match pass {
        PassId::FlameResolve => Some(RRFlameDescriptorSet::layout_spec()),
        PassId::Tonemap => Some(RRToneMapDescriptorSet::layout_spec()),
        PassId::BloomDownsample | PassId::BloomUpsample => {
            Some(RRBloomDescriptorSets::layout_spec())
        }
        PassId::Dof => Some(RRDofDescriptorSet::layout_spec()),
        PassId::AutoExposureHistogram => Some(RRAutoExposureHistogramDescriptorSet::layout_spec()),
        PassId::AutoExposureAverage => Some(RRAutoExposureAverageDescriptorSet::layout_spec()),
        PassId::RayQueryShadow => Some(RRRayQueryDescriptorSet::layout_spec()),
        PassId::Composite => Some(RRCompositeDescriptorSet::layout_spec()),
        PassId::Billboard => Some(RRBillboardDescriptorSet::layout_spec()),
        PassId::OnionSkinComposite => Some(onion_skin_composite_layout_spec()),
        PassId::Imgui => Some(imgui_layout_spec()),
        PassId::Model
        | PassId::Gbuffer
        | PassId::Grid
        | PassId::Gizmo
        | PassId::Bone
        | PassId::OnionSkinGhost => None,
    }
}
