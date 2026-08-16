use crate::core::device::*;
use crate::descriptor::pass_shaders::{FLAME_DESCRIPTOR_SET, FLAME_RESOLVE_SHADERS};
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::vulkan::*;

const FLAME_HISTORY_SET_COUNT: usize = 2;

const FLAME_UBO_BINDING: u32 = 0;
const HISTORY_SAMPLER_BINDING: u32 = 4;
const SDF_SAMPLER_BINDING: u32 = 5;
const SCENE_DEPTH_SAMPLER_BINDING: u32 = 6;

#[derive(Clone, Copy, Debug)]
pub struct FlameImageBindings {
    pub history_image_views: [vk::ImageView; FLAME_HISTORY_SET_COUNT],
    pub flame_sampler: vk::Sampler,
    pub sdf_image_view: vk::ImageView,
    pub sdf_sampler: vk::Sampler,
    pub scene_depth_view: vk::ImageView,
}

#[derive(Clone, Debug, Default)]
pub struct RRFlameDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: [vk::DescriptorSet; FLAME_HISTORY_SET_COUNT],
    pub scene_depth_sampler: vk::Sampler,
}

impl RRFlameDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::new(FLAME_RESOLVE_SHADERS.to_vec(), FLAME_DESCRIPTOR_SET)
            .with_override(
                FLAME_UBO_BINDING,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            )
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let descriptor_pool = layout.create_pool(
            rrdevice,
            FLAME_HISTORY_SET_COUNT as u32,
            vk::DescriptorPoolCreateFlags::empty(),
        )?;
        let sets = layout.allocate_sets(rrdevice, descriptor_pool, FLAME_HISTORY_SET_COUNT)?;
        let scene_depth_sampler = create_scene_depth_sampler(rrdevice)?;

        Ok(Self {
            layout,
            descriptor_pool,
            descriptor_sets: [sets[0], sets[1]],
            scene_depth_sampler,
        })
    }

    pub unsafe fn write_all(
        &self,
        rrdevice: &RRDevice,
        flame_ubo_buffer: vk::Buffer,
        flame_ubo_size: vk::DeviceSize,
        images: FlameImageBindings,
    ) -> Result<()> {
        for descriptor_set in self.descriptor_sets {
            self.layout
                .writer(descriptor_set)
                .buffer(FLAME_UBO_BINDING, flame_ubo_buffer, 0, flame_ubo_size)?
                .apply(rrdevice);
        }
        self.update_image_views(rrdevice, images)
    }

    pub unsafe fn update_image_views(
        &self,
        rrdevice: &RRDevice,
        images: FlameImageBindings,
    ) -> Result<()> {
        for (history_index, descriptor_set) in self.descriptor_sets.into_iter().enumerate() {
            let previous_history_view = images.history_image_views[1 - history_index];
            self.layout
                .writer(descriptor_set)
                .image(
                    HISTORY_SAMPLER_BINDING,
                    previous_history_view,
                    images.flame_sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )?
                .image(
                    SDF_SAMPLER_BINDING,
                    images.sdf_image_view,
                    images.sdf_sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )?
                .image(
                    SCENE_DEPTH_SAMPLER_BINDING,
                    images.scene_depth_view,
                    self.scene_depth_sampler,
                    vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                )?
                .apply(rrdevice);
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.descriptor_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.descriptor_pool = vk::DescriptorPool::null();
        }
        self.layout.destroy(device);
        device.destroy_sampler(self.scene_depth_sampler, None);
    }
}

// Depth formats must not be sampled with LINEAR filtering.
unsafe fn create_scene_depth_sampler(rrdevice: &RRDevice) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .build();
    Ok(rrdevice.device.create_sampler(&info, None)?)
}
