use crate::core::device::*;
use crate::descriptor::pass_shaders::RAY_QUERY_SHADOW_SHADER;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::vulkan::*;

const POSITION_IMAGE_BINDING: u32 = 0;
const NORMAL_IMAGE_BINDING: u32 = 1;
const SHADOW_MASK_IMAGE_BINDING: u32 = 2;
const TLAS_BINDING: u32 = 3;
const SCENE_UBO_BINDING: u32 = 4;

#[derive(Clone, Debug, Default)]
pub struct RRRayQueryDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRRayQueryDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::new(vec![RAY_QUERY_SHADOW_SHADER], 0)
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let descriptor_pool =
            layout.create_pool(rrdevice, 1, vk::DescriptorPoolCreateFlags::empty())?;

        Ok(Self {
            layout,
            descriptor_pool,
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
    ) -> Result<()> {
        self.descriptor_set = self
            .layout
            .allocate_sets(rrdevice, self.descriptor_pool, 1)?[0];

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
                SCENE_UBO_BINDING,
                scene_uniform_buffer,
                0,
                std::mem::size_of::<crate::data::SceneUniformData>() as u64,
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
                POSITION_IMAGE_BINDING,
                position_image_view,
                storage_sampler,
                vk::ImageLayout::GENERAL,
            )?
            .image(
                NORMAL_IMAGE_BINDING,
                normal_image_view,
                storage_sampler,
                vk::ImageLayout::GENERAL,
            )?
            .image(
                SHADOW_MASK_IMAGE_BINDING,
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
            .acceleration_structure(TLAS_BINDING, tlas)?
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
