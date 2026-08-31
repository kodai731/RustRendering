use crate::core::device::*;
use crate::descriptor::pass_manifest::RAY_QUERY_SHADOW;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::ray_query_shadow;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRRayQueryDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRRayQueryDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&RAY_QUERY_SHADOW)
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;

        Ok(Self {
            layout,
            descriptor_set: vk::DescriptorSet::null(),
        })
    }

    pub unsafe fn allocate_and_update(
        &mut self,
        rrdevice: &RRDevice,
        position_image_view: vk::ImageView,
        normal_image_view: vk::ImageView,
        shadow_mask_image_view: vk::ImageView,
        tlas: vk::AccelerationStructureKHR,
        scene_uniform_buffer: vk::Buffer,
        hit_shading_table_buffer: vk::Buffer,
    ) -> Result<()> {
        self.descriptor_set = self.layout.allocate_set(rrdevice)?;

        self.update_gbuffer_views(
            rrdevice,
            position_image_view,
            normal_image_view,
            shadow_mask_image_view,
        )?;
        self.update_tlas(rrdevice, tlas)?;
        self.layout
            .writer(self.descriptor_set)
            .buffer(
                ray_query_shadow::SCENE_DATA,
                scene_uniform_buffer,
                0,
                std::mem::size_of::<crate::data::SceneUniformData>() as u64,
            )?
            .buffer(
                ray_query_shadow::HIT_TABLE,
                hit_shading_table_buffer,
                0,
                vk::WHOLE_SIZE as u64,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn update_gbuffer_views(
        &self,
        rrdevice: &RRDevice,
        position_image_view: vk::ImageView,
        normal_image_view: vk::ImageView,
        shadow_mask_image_view: vk::ImageView,
    ) -> Result<()> {
        if self.descriptor_set == vk::DescriptorSet::null() {
            return Ok(());
        }

        let storage_sampler = vk::Sampler::null();
        self.layout
            .writer(self.descriptor_set)
            .image(
                ray_query_shadow::POSITION_IMAGE,
                position_image_view,
                storage_sampler,
                vk::ImageLayout::GENERAL,
            )?
            .image(
                ray_query_shadow::NORMAL_IMAGE,
                normal_image_view,
                storage_sampler,
                vk::ImageLayout::GENERAL,
            )?
            .image(
                ray_query_shadow::SHADOW_MASK_IMAGE,
                shadow_mask_image_view,
                storage_sampler,
                vk::ImageLayout::GENERAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn update_tlas(
        &mut self,
        rrdevice: &RRDevice,
        tlas: vk::AccelerationStructureKHR,
    ) -> Result<()> {
        if self.descriptor_set == vk::DescriptorSet::null() {
            return Ok(());
        }

        self.layout
            .writer(self.descriptor_set)
            .acceleration_structure(ray_query_shadow::TOP_LEVEL_AS, tlas)?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}
