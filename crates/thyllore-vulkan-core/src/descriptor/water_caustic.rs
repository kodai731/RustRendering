use crate::core::device::*;
use crate::descriptor::pass_manifest::{WATER_CAUSTIC_APPLY, WATER_CAUSTIC_SPLAT};
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::{water_caustic_apply, water_caustic_splat};
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRWaterCausticDescriptorSet {
    pub splat_layout: ReflectedSetLayout,
    pub splat_descriptor_set: vk::DescriptorSet,
    pub apply_layout: ReflectedSetLayout,
    pub apply_descriptor_set: vk::DescriptorSet,
}

impl RRWaterCausticDescriptorSet {
    pub fn splat_layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&WATER_CAUSTIC_SPLAT)
    }

    pub fn apply_layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&WATER_CAUSTIC_APPLY)
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let splat_layout = ReflectedSetLayout::create(rrdevice, &Self::splat_layout_spec())?;
        let apply_layout = ReflectedSetLayout::create(rrdevice, &Self::apply_layout_spec())?;

        Ok(Self {
            splat_layout,
            splat_descriptor_set: vk::DescriptorSet::null(),
            apply_layout,
            apply_descriptor_set: vk::DescriptorSet::null(),
        })
    }

    pub unsafe fn allocate_and_update(
        &mut self,
        rrdevice: &RRDevice,
        caustic_accum_view: vk::ImageView,
        position_image_view: vk::ImageView,
        tlas: Option<vk::AccelerationStructureKHR>,
        scene_uniform_buffer: vk::Buffer,
        water_ubo: vk::Buffer,
        hdr_color_image_view: vk::ImageView,
    ) -> Result<()> {
        if self.splat_descriptor_set == vk::DescriptorSet::null() {
            self.splat_descriptor_set = self.splat_layout.allocate_set(rrdevice)?;
        }
        if self.apply_descriptor_set == vk::DescriptorSet::null() {
            self.apply_descriptor_set = self.apply_layout.allocate_set(rrdevice)?;
        }

        let mut splat_writer = self
            .splat_layout
            .writer(self.splat_descriptor_set)
            .image(
                water_caustic_splat::CAUSTIC_ACCUM_IMAGE,
                caustic_accum_view,
                vk::Sampler::null(),
                vk::ImageLayout::GENERAL,
            )?
            .image(
                water_caustic_splat::POSITION_IMAGE,
                position_image_view,
                vk::Sampler::null(),
                vk::ImageLayout::GENERAL,
            )?;
        if let Some(tlas) = tlas {
            splat_writer =
                splat_writer.acceleration_structure(water_caustic_splat::TOP_LEVEL_AS, tlas)?;
        }
        splat_writer
            .buffer(
                water_caustic_splat::SCENE_DATA,
                scene_uniform_buffer,
                0,
                std::mem::size_of::<crate::data::SceneUniformData>() as u64,
            )?
            .buffer(
                water_caustic_splat::WATER,
                water_ubo,
                0,
                vk::WHOLE_SIZE as u64,
            )?
            .apply(rrdevice);

        self.apply_layout
            .writer(self.apply_descriptor_set)
            .image(
                water_caustic_apply::CAUSTIC_ACCUM_IMAGE,
                caustic_accum_view,
                vk::Sampler::null(),
                vk::ImageLayout::GENERAL,
            )?
            .image(
                water_caustic_apply::HDR_COLOR_IMAGE,
                hdr_color_image_view,
                vk::Sampler::null(),
                vk::ImageLayout::GENERAL,
            )?
            .buffer(
                water_caustic_apply::SCENE_DATA,
                scene_uniform_buffer,
                0,
                std::mem::size_of::<crate::data::SceneUniformData>() as u64,
            )?
            .buffer(
                water_caustic_apply::WATER,
                water_ubo,
                0,
                vk::WHOLE_SIZE as u64,
            )?
            .apply(rrdevice);

        Ok(())
    }

    pub unsafe fn update_tlas(
        &mut self,
        rrdevice: &RRDevice,
        tlas: vk::AccelerationStructureKHR,
    ) -> Result<()> {
        if self.splat_descriptor_set == vk::DescriptorSet::null() {
            return Ok(());
        }
        self.splat_layout
            .writer(self.splat_descriptor_set)
            .acceleration_structure(water_caustic_splat::TOP_LEVEL_AS, tlas)?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.splat_layout.destroy(device);
        self.apply_layout.destroy(device);
    }
}
