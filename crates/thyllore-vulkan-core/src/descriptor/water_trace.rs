use crate::core::device::*;
use crate::descriptor::pass_manifest::WATER_TRACE;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::water_trace;
use crate::resource::uniform_buffer::UniformBuffer;
use crate::vulkan::*;
use thyllore_effect_core::WaterUBO;

#[derive(Clone, Debug, Default)]
pub struct RRWaterTraceDescriptorSet {
    pub layout: ReflectedSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl RRWaterTraceDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&WATER_TRACE)
    }

    pub unsafe fn new(rrdevice: &RRDevice, frames_in_flight: usize) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let descriptor_sets = layout.allocate_sets(rrdevice, frames_in_flight.max(1))?;

        Ok(Self {
            layout,
            descriptor_sets,
        })
    }

    pub fn descriptor_set(&self, frame_slot: usize) -> Result<vk::DescriptorSet> {
        self.descriptor_sets
            .get(frame_slot)
            .copied()
            .ok_or_else(|| {
                anyhow!(
                    "water trace descriptor slot {frame_slot} exceeds {} sets",
                    self.descriptor_sets.len()
                )
            })
    }

    pub unsafe fn write_all_at(
        &self,
        rrdevice: &RRDevice,
        frame_slot: usize,
        tlas: vk::AccelerationStructureKHR,
        trace_image_view: vk::ImageView,
        water_ubo: &UniformBuffer<WaterUBO>,
        hit_table: vk::Buffer,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set(frame_slot)?)
            .acceleration_structure(water_trace::TLAS, tlas)?
            .image(
                water_trace::OUT_IMAGE,
                trace_image_view,
                vk::Sampler::null(),
                vk::ImageLayout::GENERAL,
            )?
            .uniform(water_trace::WATER, water_ubo, 0)?
            .buffer(water_trace::HIT_TABLE, hit_table, 0, vk::WHOLE_SIZE as u64)?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}
