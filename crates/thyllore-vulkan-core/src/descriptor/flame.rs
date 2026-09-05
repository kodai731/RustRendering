use crate::core::device::*;
use crate::descriptor::pass_manifest::FLAME_RESOLVE;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::flame_resolve;
use crate::resource::uniform_buffer::UniformBuffer;
use crate::vulkan::*;
use thyllore_effect_core::FlameUBO;

const FLAME_HISTORY_SET_COUNT: usize = 2;

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
    pub descriptor_sets: [vk::DescriptorSet; FLAME_HISTORY_SET_COUNT],
    pub scene_depth_sampler: vk::Sampler,
}

impl RRFlameDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&FLAME_RESOLVE).with_override(
            flame_resolve::FLAME,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        )
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let sets = layout.allocate_sets(rrdevice, FLAME_HISTORY_SET_COUNT)?;
        let scene_depth_sampler = create_scene_depth_sampler(rrdevice)?;

        Ok(Self {
            layout,
            descriptor_sets: [sets[0], sets[1]],
            scene_depth_sampler,
        })
    }

    pub unsafe fn write_all(
        &self,
        rrdevice: &RRDevice,
        flame_ubo: &UniformBuffer<FlameUBO>,
        images: FlameImageBindings,
    ) -> Result<()> {
        for descriptor_set in self.descriptor_sets {
            self.layout
                .writer(descriptor_set)
                .uniform_dynamic(flame_resolve::FLAME, flame_ubo)?
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
                    flame_resolve::FLAME_HISTORY_SAMPLER,
                    previous_history_view,
                    images.flame_sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )?
                .image(
                    flame_resolve::FLAME_SDF_SAMPLER,
                    images.sdf_image_view,
                    images.sdf_sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )?
                .image(
                    flame_resolve::SCENE_DEPTH_SAMPLER,
                    images.scene_depth_view,
                    self.scene_depth_sampler,
                    vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                )?
                .apply(rrdevice);
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
        device.destroy_sampler(self.scene_depth_sampler, None);
    }
}

// Depth formats must not be sampled with LINEAR filtering.
pub(crate) unsafe fn create_scene_depth_sampler(rrdevice: &RRDevice) -> Result<vk::Sampler> {
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
