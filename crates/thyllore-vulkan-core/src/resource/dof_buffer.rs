use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::command::{begin_single_time_commands, end_single_time_commands};
use crate::core::RRDevice;
use crate::resource::hdr_buffer::HDR_FORMAT;
use crate::resource::image::{create_image, create_image_view};

#[derive(Clone, Debug, Default)]
pub struct DofBuffer {
    pub output_image: vk::Image,
    pub output_image_memory: vk::DeviceMemory,
    pub output_image_view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
    pub render_pass: vk::RenderPass,
    pub sampler: vk::Sampler,
    pub width: u32,
    pub height: u32,
}

impl DofBuffer {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        width: u32,
        height: u32,
        command_pool: vk::CommandPool,
    ) -> Result<Self> {
        let (output_image, output_image_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            vk::SampleCountFlags::_1,
            HDR_FORMAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let output_image_view = create_image_view(
            rrdevice,
            output_image,
            HDR_FORMAT,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        let render_pass = Self::create_render_pass(rrdevice)?;

        let attachments = [output_image_view];
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

        Self::transition_initial_layout(rrdevice, command_pool, output_image)?;

        log!("Created DOF buffer: {}x{}", width, height);

        Ok(Self {
            output_image,
            output_image_memory,
            output_image_view,
            framebuffer,
            render_pass,
            sampler,
            width,
            height,
        })
    }

    unsafe fn create_render_pass(rrdevice: &RRDevice) -> Result<vk::RenderPass> {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(HDR_FORMAT)
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

        let color_attachments = [color_attachment_ref];

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachments);

        let dependency_in = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let dependency_out = vk::SubpassDependency::builder()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            )
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        let attachments = [color_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency_in, dependency_out];

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

    unsafe fn transition_initial_layout(
        rrdevice: &RRDevice,
        command_pool: vk::CommandPool,
        image: vk::Image,
    ) -> Result<()> {
        let command_buffer = begin_single_time_commands(rrdevice, command_pool)?;

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        rrdevice.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        end_single_time_commands(
            rrdevice,
            rrdevice.graphics_queue,
            command_pool,
            command_buffer,
        )?;
        Ok(())
    }

    pub unsafe fn resize(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        new_width: u32,
        new_height: u32,
        command_pool: vk::CommandPool,
    ) -> Result<()> {
        if new_width == self.width && new_height == self.height {
            return Ok(());
        }

        let render_pass = self.render_pass;
        self.render_pass = vk::RenderPass::null();
        self.destroy(&rrdevice.device);

        let new_buf = Self::new(instance, rrdevice, new_width, new_height, command_pool)?;
        *self = new_buf;
        rrdevice.device.destroy_render_pass(render_pass, None);

        log!("Resized DOF buffer to: {}x{}", new_width, new_height);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.framebuffer != vk::Framebuffer::null() {
            device.destroy_framebuffer(self.framebuffer, None);
            self.framebuffer = vk::Framebuffer::null();
        }
        if self.output_image_view != vk::ImageView::null() {
            device.destroy_image_view(self.output_image_view, None);
            self.output_image_view = vk::ImageView::null();
        }
        if self.output_image != vk::Image::null() {
            device.destroy_image(self.output_image, None);
            self.output_image = vk::Image::null();
        }
        if self.output_image_memory != vk::DeviceMemory::null() {
            device.free_memory(self.output_image_memory, None);
            self.output_image_memory = vk::DeviceMemory::null();
        }
        if self.sampler != vk::Sampler::null() {
            device.destroy_sampler(self.sampler, None);
            self.sampler = vk::Sampler::null();
        }
        if self.render_pass != vk::RenderPass::null() {
            device.destroy_render_pass(self.render_pass, None);
            self.render_pass = vk::RenderPass::null();
        }

        log!("Destroyed DOF buffer");
    }

    pub fn extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }
}

impl Drop for DofBuffer {
    fn drop(&mut self) {
        if self.output_image != vk::Image::null() {
            log_warn!("DofBuffer dropped without calling destroy()");
        }
    }
}
