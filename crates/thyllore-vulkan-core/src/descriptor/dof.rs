use crate::core::device::*;
use crate::descriptor::pass_manifest::DOF;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::dof;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRDofDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRDofDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&DOF)
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let descriptor_set = layout.allocate_set(rrdevice)?;

        Ok(Self {
            layout,
            descriptor_set,
        })
    }

    pub unsafe fn update_image_views(
        &self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
        depth_image_view: vk::ImageView,
        depth_sampler: vk::Sampler,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set)
            .image(
                dof::HDR_SAMPLER,
                hdr_image_view,
                hdr_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .image(
                dof::DEPTH_SAMPLER,
                depth_image_view,
                depth_sampler,
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}
