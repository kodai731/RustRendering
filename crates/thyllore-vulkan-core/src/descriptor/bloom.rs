use crate::core::device::*;
use crate::descriptor::pass_shaders::bloom_shaders;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::vulkan::*;

const INPUT_SAMPLER_BINDING: u32 = 0;

#[derive(Clone, Debug, Default)]
pub struct RRBloomDescriptorSets {
    pub layout: ReflectedSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub downsample_sets: Vec<vk::DescriptorSet>,
    pub upsample_sets: Vec<vk::DescriptorSet>,
}

impl RRBloomDescriptorSets {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::new(bloom_shaders(), 0)
    }

    pub unsafe fn new(rrdevice: &RRDevice, mip_count: usize) -> Result<Self> {
        let downsample_count = mip_count;
        let upsample_count = mip_count.saturating_sub(1);
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let descriptor_pool = layout.create_pool(
            rrdevice,
            (downsample_count + upsample_count) as u32,
            vk::DescriptorPoolCreateFlags::empty(),
        )?;
        let downsample_sets = layout.allocate_sets(rrdevice, descriptor_pool, downsample_count)?;
        let upsample_sets = layout.allocate_sets(rrdevice, descriptor_pool, upsample_count)?;

        Ok(Self {
            layout,
            descriptor_pool,
            downsample_sets,
            upsample_sets,
        })
    }

    unsafe fn write_input(
        &self,
        rrdevice: &RRDevice,
        descriptor_set: vk::DescriptorSet,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<()> {
        self.layout
            .writer(descriptor_set)
            .image(
                INPUT_SAMPLER_BINDING,
                image_view,
                sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn update_image_views(
        &self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        mip_image_views: &[vk::ImageView],
        sampler: vk::Sampler,
    ) -> Result<()> {
        for (mip_index, set) in self.downsample_sets.iter().enumerate() {
            let input_view = match mip_index {
                0 => hdr_image_view,
                _ => mip_image_views[mip_index - 1],
            };
            self.write_input(rrdevice, *set, input_view, sampler)?;
        }

        for (pass_index, set) in self.upsample_sets.iter().enumerate() {
            let source_view_index = mip_image_views.len() - 1 - pass_index;
            self.write_input(rrdevice, *set, mip_image_views[source_view_index], sampler)?;
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.descriptor_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.descriptor_pool = vk::DescriptorPool::null();
        }
        self.layout.destroy(device);
        self.downsample_sets.clear();
        self.upsample_sets.clear();
    }
}
