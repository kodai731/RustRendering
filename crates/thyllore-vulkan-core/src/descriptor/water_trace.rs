use crate::core::device::*;
use crate::descriptor::pass_manifest::WATER_TRACE;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::water_trace;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRWaterTraceDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRWaterTraceDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&WATER_TRACE)
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let descriptor_set = layout.allocate_set(rrdevice)?;

        Ok(Self {
            layout,
            descriptor_set,
        })
    }

    pub unsafe fn write_all(
        &self,
        rrdevice: &RRDevice,
        tlas: vk::AccelerationStructureKHR,
        trace_image_view: vk::ImageView,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set)
            .acceleration_structure(water_trace::TLAS, tlas)?
            .image(
                water_trace::OUT_IMAGE,
                trace_image_view,
                vk::Sampler::null(),
                vk::ImageLayout::GENERAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}
