use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::vulkanr::core::RRDevice;
use crate::vulkanr::descriptor::{imgui_layout_spec, ReflectedSetLayout, IMGUI_TEXTURE_BINDING};
use crate::vulkanr::resource::{
    AutoExposureBuffers, BloomChain, DofBuffer, FlameBuffer, HdrBuffer, OffscreenFramebuffer,
};

#[derive(Debug, Default)]
pub struct ViewportState {
    pub offscreen: Option<OffscreenFramebuffer>,
    pub hdr_buffer: Option<HdrBuffer>,
    pub bloom_chain: Option<BloomChain>,
    pub dof_buffer: Option<DofBuffer>,
    pub auto_exposure_buffers: Option<AutoExposureBuffers>,
    pub flame_buffer: Option<FlameBuffer>,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set_layout: ReflectedSetLayout,
    pub descriptor_set: vk::DescriptorSet,
    pub width: u32,
    pub height: u32,
    pub focused: bool,
    pub hovered: bool,
    pub hdr_grid_pipeline_id: Option<usize>,
}

impl ViewportState {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
        msaa_samples: vk::SampleCountFlags,
        swapchain_format: vk::Format,
    ) -> Result<Self> {
        let offscreen = OffscreenFramebuffer::new(
            instance,
            rrdevice,
            command_pool,
            width,
            height,
            msaa_samples,
            swapchain_format,
        )?;

        let hdr_buffer = HdrBuffer::new(instance, rrdevice, width, height)?;

        let bloom_chain = BloomChain::new(instance, rrdevice, width, height, 5, command_pool)?;

        let dof_buffer = DofBuffer::new(instance, rrdevice, width, height, command_pool)?;

        let auto_exposure_buffers = AutoExposureBuffers::new(instance, rrdevice, width, height)?;

        let flame_buffer = FlameBuffer::new(
            instance,
            rrdevice,
            command_pool,
            width,
            height,
            hdr_buffer.color_image_view,
        )?;

        let (descriptor_pool, descriptor_set_layout, descriptor_set) =
            Self::create_imgui_descriptor(rrdevice, &offscreen)?;

        Ok(Self {
            offscreen: Some(offscreen),
            hdr_buffer: Some(hdr_buffer),
            bloom_chain: Some(bloom_chain),
            dof_buffer: Some(dof_buffer),
            auto_exposure_buffers: Some(auto_exposure_buffers),
            flame_buffer: Some(flame_buffer),
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            width,
            height,
            focused: false,
            hovered: false,
            hdr_grid_pipeline_id: None,
        })
    }

    unsafe fn create_imgui_descriptor(
        rrdevice: &RRDevice,
        offscreen: &OffscreenFramebuffer,
    ) -> Result<(vk::DescriptorPool, ReflectedSetLayout, vk::DescriptorSet)> {
        let layout = ReflectedSetLayout::create(rrdevice, &imgui_layout_spec())?;
        let descriptor_pool =
            layout.create_pool(rrdevice, 1, vk::DescriptorPoolCreateFlags::empty())?;
        let descriptor_set = layout.allocate_sets(rrdevice, descriptor_pool, 1)?[0];

        Self::update_descriptor_set(rrdevice, &layout, descriptor_set, offscreen)?;

        Ok((descriptor_pool, layout, descriptor_set))
    }

    unsafe fn update_descriptor_set(
        rrdevice: &RRDevice,
        layout: &ReflectedSetLayout,
        descriptor_set: vk::DescriptorSet,
        offscreen: &OffscreenFramebuffer,
    ) -> Result<()> {
        layout
            .writer(descriptor_set)
            .image(
                IMGUI_TEXTURE_BINDING,
                offscreen.resolve_image_view(),
                offscreen.sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn resize(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        command_pool: vk::CommandPool,
        new_width: u32,
        new_height: u32,
    ) -> Result<()> {
        if new_width == self.width && new_height == self.height {
            return Ok(());
        }

        if new_width == 0 || new_height == 0 {
            return Ok(());
        }

        if let Some(ref mut offscreen) = self.offscreen {
            offscreen.resize(instance, rrdevice, command_pool, new_width, new_height)?;
            Self::update_descriptor_set(
                rrdevice,
                &self.descriptor_set_layout,
                self.descriptor_set,
                offscreen,
            )?;
        }

        if let Some(ref mut hdr_buffer) = self.hdr_buffer {
            hdr_buffer.resize(instance, rrdevice, new_width, new_height)?;
        }

        if let Some(ref mut bloom_chain) = self.bloom_chain {
            bloom_chain.resize(instance, rrdevice, new_width, new_height, command_pool)?;
        }

        if let Some(ref mut dof_buffer) = self.dof_buffer {
            dof_buffer.resize(instance, rrdevice, new_width, new_height, command_pool)?;
        }

        if let Some(ref mut ae_buffers) = self.auto_exposure_buffers {
            ae_buffers.resize(instance, rrdevice, new_width, new_height)?;
        }

        if let (Some(ref mut flame_buffer), Some(ref hdr_buffer)) =
            (&mut self.flame_buffer, &self.hdr_buffer)
        {
            flame_buffer.resize(
                instance,
                rrdevice,
                command_pool,
                new_width,
                new_height,
                hdr_buffer.color_image_view,
            )?;
        }

        self.width = new_width;
        self.height = new_height;

        log!("Viewport resized to: {}x{}", new_width, new_height);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        self.descriptor_set_layout.destroy(device);

        if let Some(ref mut offscreen) = self.offscreen {
            offscreen.destroy(device);
        }

        if let Some(ref mut hdr_buffer) = self.hdr_buffer {
            hdr_buffer.destroy(device);
        }

        if let Some(ref mut bloom_chain) = self.bloom_chain {
            bloom_chain.destroy(device);
        }

        if let Some(ref mut dof_buffer) = self.dof_buffer {
            dof_buffer.destroy(device);
        }

        if let Some(ref mut ae_buffers) = self.auto_exposure_buffers {
            ae_buffers.destroy(device);
        }

        if let Some(ref mut flame_buffer) = self.flame_buffer {
            flame_buffer.destroy(device);
        }

        log!("Destroyed viewport state");
    }

    pub fn texture_id(&self) -> usize {
        self.descriptor_set.as_raw() as usize
    }
}
