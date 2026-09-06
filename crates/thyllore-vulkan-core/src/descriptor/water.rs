use crate::core::device::*;
use crate::descriptor::pass_manifest::WATER_RESOLVE;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::water_resolve;
use crate::resource::uniform_buffer::UniformBuffer;
use crate::vulkan::*;
use thyllore_effect_core::WaterUBO;

const WATER_HISTORY_SET_COUNT: usize = 2;

#[derive(Clone, Debug, Default)]
pub struct RRWaterDescriptorSet {
    pub layout: ReflectedSetLayout,
    frames: Vec<[vk::DescriptorSet; WATER_HISTORY_SET_COUNT]>,
}

impl RRWaterDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&WATER_RESOLVE).with_override(
            water_resolve::WATER,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        )
    }

    pub unsafe fn new(rrdevice: &RRDevice, frames_in_flight: usize) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;

        let mut frames = Vec::with_capacity(frames_in_flight.max(1));
        for _ in 0..frames_in_flight.max(1) {
            let sets = layout.allocate_sets(rrdevice, WATER_HISTORY_SET_COUNT)?;
            frames.push([sets[0], sets[1]]);
        }

        Ok(Self { layout, frames })
    }

    pub fn descriptor_set(
        &self,
        frame_slot: usize,
        history_index: usize,
    ) -> Result<vk::DescriptorSet> {
        let sets = self.frames.get(frame_slot).ok_or_else(|| {
            anyhow!(
                "water descriptor slot {frame_slot} exceeds {} frames",
                self.frames.len()
            )
        })?;
        sets.get(history_index)
            .copied()
            .ok_or_else(|| anyhow!("water history index {history_index} is out of range"))
    }

    pub unsafe fn write_all_at(
        &self,
        rrdevice: &RRDevice,
        frame_slot: usize,
        water_ubo: &UniformBuffer<WaterUBO>,
        scene_color_view: vk::ImageView,
        scene_color_sampler: vk::Sampler,
        history_image_views: [vk::ImageView; 2],
        history_sampler: vk::Sampler,
        trace_image_view: vk::ImageView,
        trace_sampler: vk::Sampler,
        tlas: vk::AccelerationStructureKHR,
        hit_table: vk::Buffer,
    ) -> Result<()> {
        for i in 0..WATER_HISTORY_SET_COUNT {
            let descriptor_set = self.descriptor_set(frame_slot, i)?;
            let previous_history_view = history_image_views[1 - i];
            self.layout
                .writer(descriptor_set)
                .uniform_dynamic(water_resolve::WATER, water_ubo)?
                .image(
                    water_resolve::SCENE_COLOR_SAMPLER,
                    scene_color_view,
                    scene_color_sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )?
                .image(
                    water_resolve::WATER_HISTORY_SAMPLER,
                    previous_history_view,
                    history_sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )?
                .image(
                    water_resolve::WATER_TRACE_SAMPLER,
                    trace_image_view,
                    trace_sampler,
                    vk::ImageLayout::GENERAL,
                )?
                .acceleration_structure(water_resolve::SCENE_TLAS, tlas)?
                .buffer(
                    water_resolve::HIT_TABLE,
                    hit_table,
                    0,
                    vk::WHOLE_SIZE as u64,
                )?
                .apply(rrdevice);
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}
