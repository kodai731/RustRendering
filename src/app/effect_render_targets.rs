use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::vulkanr::core::RRDevice;
use crate::vulkanr::resource::{FlameBuffer, RenderTargetRegistry, WaterBuffer};

#[derive(Debug, Default)]
pub struct EffectRenderTargets {
    pub flame: Option<FlameBuffer>,
    pub water: Option<WaterBuffer>,
}

impl EffectRenderTargets {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        registry: &mut RenderTargetRegistry,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
        hdr_image_view: vk::ImageView,
    ) -> Result<Self> {
        let flame = FlameBuffer::new(
            instance,
            rrdevice,
            registry,
            command_pool,
            width,
            height,
            hdr_image_view,
        )?;

        Ok(Self {
            flame: Some(flame),
            water: None,
        })
    }

    pub unsafe fn resize(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        registry: &mut RenderTargetRegistry,
        command_pool: vk::CommandPool,
        new_width: u32,
        new_height: u32,
        hdr_image_view: vk::ImageView,
    ) -> Result<()> {
        if let Some(mut water) = self.water.take() {
            water.destroy(&rrdevice.device);
        }

        if let Some(ref mut flame) = self.flame {
            flame.resize(
                instance,
                rrdevice,
                registry,
                command_pool,
                new_width,
                new_height,
                hdr_image_view,
            )?;
        }

        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if let Some(mut flame) = self.flame.take() {
            flame.destroy(device);
        }

        if let Some(mut water) = self.water.take() {
            water.destroy(device);
        }
    }
}
