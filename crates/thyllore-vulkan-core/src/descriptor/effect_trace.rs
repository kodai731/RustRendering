use crate::core::device::*;
use crate::descriptor::pass_manifest::EFFECT_TRACE;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::effect_trace;
use crate::resource::uniform_buffer::UniformBuffer;
use crate::vulkan::*;
use thyllore_effect_core::WaterUBO;

#[derive(Clone, Debug, Default)]
pub struct RREffectTraceDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_set: vk::DescriptorSet,
}

impl RREffectTraceDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&EFFECT_TRACE)
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
        water_ubo: &UniformBuffer<WaterUBO>,
        hit_table: vk::Buffer,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set)
            .acceleration_structure(effect_trace::TLAS, tlas)?
            .image(
                effect_trace::OUT_IMAGE,
                trace_image_view,
                vk::Sampler::null(),
                vk::ImageLayout::GENERAL,
            )?
            .uniform(effect_trace::WATER, water_ubo, 0)?
            .buffer(effect_trace::HIT_TABLE, hit_table, 0, vk::WHOLE_SIZE as u64)?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}
