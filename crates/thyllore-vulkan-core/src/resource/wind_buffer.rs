use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::core::RRDevice;
use crate::resource::hdr_buffer::HDR_FORMAT;

/// Render pass and framebuffer that blend the wind resolve pass onto the HDR color image.
#[derive(Clone, Debug, Default)]
pub struct WindBuffer {
    pub render_pass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    pub width: u32,
    pub height: u32,
}

impl WindBuffer {
    pub unsafe fn new(
        rrdevice: &RRDevice,
        width: u32,
        height: u32,
        hdr_image_view: vk::ImageView,
    ) -> Result<Self> {
        let render_pass = Self::create_render_pass(rrdevice)?;
        let attachments = [hdr_image_view];
        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);
        let framebuffer = rrdevice
            .device
            .create_framebuffer(&framebuffer_info, None)?;

        log!("Created wind buffer: {}x{}", width, height);
        Ok(Self {
            render_pass,
            framebuffer,
            width,
            height,
        })
    }

    unsafe fn create_render_pass(rrdevice: &RRDevice) -> Result<vk::RenderPass> {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(HDR_FORMAT)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let color_attachments = [color_attachment_ref];

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachments);

        let dependency_in = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build();

        let dependency_out = vk::SubpassDependency::builder()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();

        let attachments = [color_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency_in, dependency_out];
        let info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        Ok(rrdevice.device.create_render_pass(&info, None)?)
    }

    pub unsafe fn resize(
        &mut self,
        rrdevice: &RRDevice,
        new_width: u32,
        new_height: u32,
        hdr_image_view: vk::ImageView,
    ) -> Result<()> {
        self.destroy(&rrdevice.device);
        *self = Self::new(rrdevice, new_width, new_height, hdr_image_view)?;
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.framebuffer != vk::Framebuffer::null() {
            device.destroy_framebuffer(self.framebuffer, None);
            self.framebuffer = vk::Framebuffer::null();
        }
        if self.render_pass != vk::RenderPass::null() {
            device.destroy_render_pass(self.render_pass, None);
            self.render_pass = vk::RenderPass::null();
        }
    }

    pub fn extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }
}

impl Drop for WindBuffer {
    fn drop(&mut self) {
        if self.render_pass != vk::RenderPass::null() {
            log_warn!("WindBuffer dropped without calling destroy()");
        }
    }
}
