use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::core::RRDevice;
use crate::render::pass::get_depth_format;
use crate::resource::image::{create_image, create_image_view, transition_image_layout};

#[derive(Clone, Debug, Default)]
pub struct OffscreenFramebuffer {
    pub msaa_color_image: vk::Image,
    pub msaa_color_image_memory: vk::DeviceMemory,
    pub msaa_color_image_view: vk::ImageView,

    pub resolve_color_image: vk::Image,
    pub resolve_color_image_memory: vk::DeviceMemory,
    pub resolve_color_image_view: vk::ImageView,

    pub msaa_depth_image: vk::Image,
    pub msaa_depth_image_memory: vk::DeviceMemory,
    pub msaa_depth_image_view: vk::ImageView,

    pub framebuffer: vk::Framebuffer,
    pub render_pass: vk::RenderPass,
    pub sampler: vk::Sampler,

    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub msaa_samples: vk::SampleCountFlags,
}

impl OffscreenFramebuffer {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
        msaa_samples: vk::SampleCountFlags,
        format: vk::Format,
    ) -> Result<Self> {
        let (msaa_color_image, msaa_color_image_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            msaa_samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let msaa_color_image_view = create_image_view(
            rrdevice,
            msaa_color_image,
            format,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        let (resolve_color_image, resolve_color_image_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            vk::SampleCountFlags::_1,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let resolve_color_image_view = create_image_view(
            rrdevice,
            resolve_color_image,
            format,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        transition_image_layout(
            rrdevice,
            rrdevice.graphics_queue,
            command_pool,
            resolve_color_image,
            format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            1,
        )?;

        let depth_format = get_depth_format(instance, rrdevice)?;
        let (msaa_depth_image, msaa_depth_image_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            msaa_samples,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let msaa_depth_image_view = create_image_view(
            rrdevice,
            msaa_depth_image,
            depth_format,
            vk::ImageAspectFlags::DEPTH,
            1,
        )?;

        let render_pass = Self::create_render_pass(instance, rrdevice, format, msaa_samples)?;

        let attachments = [
            msaa_color_image_view,
            msaa_depth_image_view,
            resolve_color_image_view,
        ];
        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);
        let framebuffer = rrdevice
            .device
            .create_framebuffer(&framebuffer_info, None)?;

        let sampler = Self::create_sampler(&rrdevice.device)?;

        log!(
            "Created offscreen framebuffer: {}x{} with MSAA {:?}",
            width,
            height,
            msaa_samples
        );

        Ok(Self {
            msaa_color_image,
            msaa_color_image_memory,
            msaa_color_image_view,
            resolve_color_image,
            resolve_color_image_memory,
            resolve_color_image_view,
            msaa_depth_image,
            msaa_depth_image_memory,
            msaa_depth_image_view,
            framebuffer,
            render_pass,
            sampler,
            width,
            height,
            format,
            msaa_samples,
        })
    }

    unsafe fn create_render_pass(
        instance: &Instance,
        rrdevice: &RRDevice,
        color_format: vk::Format,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<vk::RenderPass> {
        let msaa_color_attachment = vk::AttachmentDescription::builder()
            .format(color_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_format = get_depth_format(instance, rrdevice)?;
        let depth_attachment = vk::AttachmentDescription::builder()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let resolve_color_attachment = vk::AttachmentDescription::builder()
            .format(color_format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let resolve_attachment_ref = vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachments = [color_attachment_ref];
        let resolve_attachments = [resolve_attachment_ref];

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachments)
            .depth_stencil_attachment(&depth_attachment_ref)
            .resolve_attachments(&resolve_attachments);

        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let attachments = [
            msaa_color_attachment,
            depth_attachment,
            resolve_color_attachment,
        ];
        let subpasses = [subpass];
        let dependencies = [dependency];

        let info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = rrdevice.device.create_render_pass(&info, None)?;
        Ok(render_pass)
    }

    unsafe fn create_sampler(device: &vulkanalia::Device) -> Result<vk::Sampler> {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(1.0);

        let sampler = device.create_sampler(&sampler_info, None)?;
        Ok(sampler)
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

        let msaa_samples = self.msaa_samples;
        let format = self.format;
        self.destroy(&rrdevice.device);

        let new_fb = Self::new(
            instance,
            rrdevice,
            command_pool,
            new_width,
            new_height,
            msaa_samples,
            format,
        )?;
        *self = new_fb;

        log!(
            "Resized offscreen framebuffer to: {}x{}",
            new_width,
            new_height
        );
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        device.destroy_sampler(self.sampler, None);
        device.destroy_framebuffer(self.framebuffer, None);
        device.destroy_render_pass(self.render_pass, None);

        device.destroy_image_view(self.msaa_color_image_view, None);
        device.destroy_image(self.msaa_color_image, None);
        device.free_memory(self.msaa_color_image_memory, None);

        device.destroy_image_view(self.resolve_color_image_view, None);
        device.destroy_image(self.resolve_color_image, None);
        device.free_memory(self.resolve_color_image_memory, None);

        device.destroy_image_view(self.msaa_depth_image_view, None);
        device.destroy_image(self.msaa_depth_image, None);
        device.free_memory(self.msaa_depth_image_memory, None);

        log!("Destroyed offscreen framebuffer");
    }

    pub fn extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }

    pub fn resolve_image_view(&self) -> vk::ImageView {
        self.resolve_color_image_view
    }
}
