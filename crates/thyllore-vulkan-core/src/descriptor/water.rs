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
    pub descriptor_sets: [vk::DescriptorSet; WATER_HISTORY_SET_COUNT],
}

impl RRWaterDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&WATER_RESOLVE).with_override(
            water_resolve::WATER,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        )
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let sets = layout.allocate_sets(rrdevice, WATER_HISTORY_SET_COUNT)?;

        Ok(Self {
            layout,
            descriptor_sets: [sets[0], sets[1]],
        })
    }

    pub unsafe fn write_all(
        &self,
        rrdevice: &RRDevice,
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
        for (i, descriptor_set) in self.descriptor_sets.into_iter().enumerate() {
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

    pub fn descriptor_sets(&self) -> [vk::DescriptorSet; 2] {
        self.descriptor_sets
    }

    pub unsafe fn update_scene_color(
        &self,
        rrdevice: &RRDevice,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<()> {
        for descriptor_set in self.descriptor_sets {
            self.layout
                .writer(descriptor_set)
                .image(
                    water_resolve::SCENE_COLOR_SAMPLER,
                    view,
                    sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )?
                .apply(rrdevice);
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}
