use crate::core::device::*;
use crate::descriptor::pass_manifest::WATER_RESOLVE;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::water_resolve;
use crate::resource::uniform_buffer::UniformBuffer;
use crate::vulkan::*;
use thyllore_effect_core::WaterUBO;

#[derive(Clone, Debug, Default)]
pub struct RRWaterDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_set: vk::DescriptorSet,
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
        let descriptor_set = layout.allocate_sets(rrdevice, 1)?[0];

        Ok(Self {
            layout,
            descriptor_set,
        })
    }

    pub unsafe fn write_all(
        &self,
        rrdevice: &RRDevice,
        water_ubo: &UniformBuffer<WaterUBO>,
        scene_color_view: vk::ImageView,
        scene_color_sampler: vk::Sampler,
        tlas: vk::AccelerationStructureKHR,
        hit_table: vk::Buffer,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set)
            .uniform_dynamic(water_resolve::WATER, water_ubo)?
            .image(
                water_resolve::SCENE_COLOR_SAMPLER,
                scene_color_view,
                scene_color_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .acceleration_structure(water_resolve::SCENE_TLAS, tlas)?
            .buffer(
                water_resolve::HIT_TABLE,
                hit_table,
                0,
                vk::WHOLE_SIZE as u64,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn update_scene_color(
        &self,
        rrdevice: &RRDevice,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set)
            .image(
                water_resolve::SCENE_COLOR_SAMPLER,
                view,
                sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}
