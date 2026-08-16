use crate::core::device::*;
use crate::descriptor::pass_shaders::DOF_SHADERS;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::vulkan::*;

const HDR_SAMPLER_BINDING: u32 = 0;
const DEPTH_SAMPLER_BINDING: u32 = 1;

#[derive(Clone, Debug, Default)]
pub struct RRDofDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRDofDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::new(DOF_SHADERS.to_vec(), 0)
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let descriptor_pool =
            layout.create_pool(rrdevice, 1, vk::DescriptorPoolCreateFlags::empty())?;
        let descriptor_set = layout.allocate_sets(rrdevice, descriptor_pool, 1)?[0];

        Ok(Self {
            layout,
            descriptor_pool,
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
                HDR_SAMPLER_BINDING,
                hdr_image_view,
                hdr_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .image(
                DEPTH_SAMPLER_BINDING,
                depth_image_view,
                depth_sampler,
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.descriptor_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.descriptor_pool = vk::DescriptorPool::null();
        }
        self.layout.destroy(device);
    }
}
