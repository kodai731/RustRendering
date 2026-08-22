use crate::core::device::*;
use crate::descriptor::pass_manifest::{SetRole, BLOOM_DOWNSAMPLE, BLOOM_UPSAMPLE};
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::bloom_downsample;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRBloomDescriptorSets {
    pub layout: ReflectedSetLayout,
    pub downsample_sets: Vec<vk::DescriptorSet>,
    pub upsample_sets: Vec<vk::DescriptorSet>,
}

impl RRBloomDescriptorSets {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::new(vec![&BLOOM_DOWNSAMPLE, &BLOOM_UPSAMPLE], SetRole::Local)
    }

    pub unsafe fn new(rrdevice: &RRDevice, mip_count: usize) -> Result<Self> {
        let downsample_count = mip_count;
        let upsample_count = mip_count.saturating_sub(1);
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let downsample_sets = layout.allocate_sets(rrdevice, downsample_count)?;
        let upsample_sets = layout.allocate_sets(rrdevice, upsample_count)?;

        Ok(Self {
            layout,
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
                bloom_downsample::INPUT_SAMPLER,
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
        self.layout.destroy(device);
        self.downsample_sets.clear();
        self.upsample_sets.clear();
    }
}
