use crate::core::device::*;
use crate::descriptor::pass_manifest::{AUTO_EXPOSURE_AVERAGE, AUTO_EXPOSURE_HISTOGRAM};
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::{auto_exposure_average, auto_exposure_histogram};
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRAutoExposureHistogramDescriptorSet {
    pub layout: ReflectedSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl RRAutoExposureHistogramDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&AUTO_EXPOSURE_HISTOGRAM)
    }

    pub unsafe fn new(rrdevice: &RRDevice, set_count: usize) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let descriptor_sets = layout.allocate_sets(rrdevice, set_count.max(1))?;

        Ok(Self {
            layout,
            descriptor_sets,
        })
    }

    pub fn descriptor_set(&self, frame_slot: usize) -> Result<vk::DescriptorSet> {
        self.descriptor_sets
            .get(frame_slot)
            .copied()
            .ok_or_else(|| {
                anyhow!(
                    "auto exposure histogram descriptor slot {frame_slot} exceeds {} sets",
                    self.descriptor_sets.len()
                )
            })
    }

    pub unsafe fn update_bindings(
        &self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
        histogram_buffer: vk::Buffer,
        histogram_buffer_size: u64,
    ) -> Result<()> {
        for descriptor_set in &self.descriptor_sets {
            self.layout
                .writer(*descriptor_set)
                .image(
                    auto_exposure_histogram::HDR_IMAGE,
                    hdr_image_view,
                    hdr_sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )?
                .buffer(
                    auto_exposure_histogram::HISTOGRAM,
                    histogram_buffer,
                    0,
                    histogram_buffer_size,
                )?
                .apply(rrdevice);
        }
        Ok(())
    }

    pub unsafe fn update_hdr_image_at(
        &self,
        rrdevice: &RRDevice,
        frame_slot: usize,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set(frame_slot)?)
            .image(
                auto_exposure_histogram::HDR_IMAGE,
                hdr_image_view,
                hdr_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}

#[derive(Clone, Debug, Default)]
pub struct RRAutoExposureAverageDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRAutoExposureAverageDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&AUTO_EXPOSURE_AVERAGE)
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;
        let descriptor_set = layout.allocate_set(rrdevice)?;

        Ok(Self {
            layout,
            descriptor_set,
        })
    }

    pub unsafe fn update_bindings(
        &self,
        rrdevice: &RRDevice,
        histogram_buffer: vk::Buffer,
        histogram_buffer_size: u64,
        luminance_buffer: vk::Buffer,
        luminance_buffer_size: u64,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set)
            .buffer(
                auto_exposure_average::HISTOGRAM,
                histogram_buffer,
                0,
                histogram_buffer_size,
            )?
            .buffer(
                auto_exposure_average::RESULT,
                luminance_buffer,
                0,
                luminance_buffer_size,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.layout.destroy(device);
    }
}
