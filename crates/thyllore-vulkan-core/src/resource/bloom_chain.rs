use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::command::{begin_single_time_commands, end_single_time_commands};
use crate::core::RRDevice;
use crate::resource::hdr_buffer::HDR_FORMAT;
use crate::resource::image::{create_image, create_image_view};

#[derive(Clone, Debug, Default)]
pub struct BloomMipLevel {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub image_view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Default)]
pub struct BloomChain {
    pub mip_levels: Vec<BloomMipLevel>,
    pub downsample_render_pass: vk::RenderPass,
    pub upsample_render_pass: vk::RenderPass,
    pub sampler: vk::Sampler,
    pub mip_count: u32,
}

impl BloomChain {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        base_width: u32,
        base_height: u32,
        mip_count: u32,
        command_pool: vk::CommandPool,
    ) -> Result<Self> {
        let downsample_render_pass =
            Self::create_render_pass(rrdevice, vk::AttachmentLoadOp::DONT_CARE)?;
        let upsample_render_pass = Self::create_render_pass(rrdevice, vk::AttachmentLoadOp::LOAD)?;
        let sampler = Self::create_sampler(&rrdevice.device)?;

        let mut mip_levels = Vec::with_capacity(mip_count as usize);
        let mut width = base_width / 2;
        let mut height = base_height / 2;

        for i in 0..mip_count {
            width = width.max(1);
            height = height.max(1);

            let mip =
                Self::create_mip_level(instance, rrdevice, downsample_render_pass, width, height)?;

            log!("Created bloom mip {}: {}x{}", i, width, height);
            mip_levels.push(mip);

            width /= 2;
            height /= 2;
        }

        Self::transition_mip_layouts(rrdevice, command_pool, &mip_levels)?;

        Ok(Self {
            mip_levels,
            downsample_render_pass,
            upsample_render_pass,
            sampler,
            mip_count,
        })
    }

    unsafe fn create_mip_level(
        instance: &Instance,
        rrdevice: &RRDevice,
        render_pass: vk::RenderPass,
        width: u32,
        height: u32,
    ) -> Result<BloomMipLevel> {
        let (image, memory) = create_image(
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

        let image_view =
            create_image_view(rrdevice, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR, 1)?;

        let attachments = [image_view];
        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);
        let framebuffer = rrdevice
            .device
            .create_framebuffer(&framebuffer_info, None)?;

        Ok(BloomMipLevel {
            image,
            memory,
            image_view,
            framebuffer,
            width,
            height,
        })
    }

    unsafe fn create_render_pass(
        rrdevice: &RRDevice,
        load_op: vk::AttachmentLoadOp,
    ) -> Result<vk::RenderPass> {
        let initial_layout = if load_op == vk::AttachmentLoadOp::LOAD {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };

        let color_attachment = vk::AttachmentDescription::builder()
            .format(HDR_FORMAT)
            .samples(vk::SampleCountFlags::_1)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial_layout)
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

    pub unsafe fn resize(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        new_width: u32,
        new_height: u32,
        command_pool: vk::CommandPool,
    ) -> Result<()> {
        let current_width = self.mip_levels.first().map(|m| m.width * 2).unwrap_or(0);
        let current_height = self.mip_levels.first().map(|m| m.height * 2).unwrap_or(0);
        if new_width == current_width && new_height == current_height {
            return Ok(());
        }

        self.destroy_mip_levels(&rrdevice.device);

        let mip_count = self.mip_count;
        let mut width = new_width / 2;
        let mut height = new_height / 2;

        for i in 0..mip_count {
            width = width.max(1);
            height = height.max(1);

            let mip = Self::create_mip_level(
                instance,
                rrdevice,
                self.downsample_render_pass,
                width,
                height,
            )?;

            log!("Resized bloom mip {}: {}x{}", i, width, height);
            self.mip_levels.push(mip);

            width /= 2;
            height /= 2;
        }

        Self::transition_mip_layouts(rrdevice, command_pool, &self.mip_levels)?;

        log!("Resized bloom chain for {}x{}", new_width, new_height);
        Ok(())
    }

    unsafe fn transition_mip_layouts(
        rrdevice: &RRDevice,
        command_pool: vk::CommandPool,
        mip_levels: &[BloomMipLevel],
    ) -> Result<()> {
        let command_buffer = begin_single_time_commands(rrdevice, command_pool)?;

        for mip in mip_levels {
            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(mip.image)
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
        }

        end_single_time_commands(
            rrdevice,
            rrdevice.graphics_queue,
            command_pool,
            command_buffer,
        )?;
        Ok(())
    }

    unsafe fn destroy_mip_levels(&mut self, device: &vulkanalia::Device) {
        for mip in &self.mip_levels {
            device.destroy_framebuffer(mip.framebuffer, None);
            device.destroy_image_view(mip.image_view, None);
            device.destroy_image(mip.image, None);
            device.free_memory(mip.memory, None);
        }
        self.mip_levels.clear();
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.destroy_mip_levels(device);
        device.destroy_sampler(self.sampler, None);
        device.destroy_render_pass(self.downsample_render_pass, None);
        device.destroy_render_pass(self.upsample_render_pass, None);
        log!("Destroyed bloom chain");
    }
}
