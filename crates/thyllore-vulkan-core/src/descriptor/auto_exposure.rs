use crate::core::device::*;
use crate::descriptor::pass_manifest::{AUTO_EXPOSURE_AVERAGE, AUTO_EXPOSURE_HISTOGRAM};
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::vulkan::*;

const HISTOGRAM_HDR_SAMPLER_BINDING: u32 = 0;
const HISTOGRAM_BUFFER_BINDING: u32 = 1;
const AVERAGE_HISTOGRAM_BUFFER_BINDING: u32 = 0;
const AVERAGE_LUMINANCE_BUFFER_BINDING: u32 = 1;

#[derive(Clone, Debug, Default)]
pub struct RRAutoExposureHistogramDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRAutoExposureHistogramDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&AUTO_EXPOSURE_HISTOGRAM)
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
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
        histogram_buffer: vk::Buffer,
        histogram_buffer_size: u64,
    ) -> Result<()> {
        self.layout
            .writer(self.descriptor_set)
            .image(
                HISTOGRAM_HDR_SAMPLER_BINDING,
                hdr_image_view,
                hdr_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .buffer(
                HISTOGRAM_BUFFER_BINDING,
                histogram_buffer,
                0,
                histogram_buffer_size,
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
                AVERAGE_HISTOGRAM_BUFFER_BINDING,
                histogram_buffer,
                0,
                histogram_buffer_size,
            )?
            .buffer(
                AVERAGE_LUMINANCE_BUFFER_BINDING,
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
