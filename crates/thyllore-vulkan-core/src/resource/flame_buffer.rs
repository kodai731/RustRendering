use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::command::{begin_single_time_commands, end_single_time_commands};
use crate::core::RRDevice;
use crate::resource::hdr_buffer::HDR_FORMAT;
use crate::resource::image::{create_image, create_image_view};

pub const FLAME_HISTORY_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

#[derive(Clone, Debug, Default)]
pub struct FlameBuffer {
    pub history_images: [vk::Image; 2],
    pub history_image_memories: [vk::DeviceMemory; 2],
    pub history_image_views: [vk::ImageView; 2],
    pub sampler: vk::Sampler,
    pub shading_render_pass: vk::RenderPass,
    pub shading_framebuffers: [vk::Framebuffer; 2],
    pub width: u32,
    pub height: u32,
}

impl FlameBuffer {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
        hdr_image_view: vk::ImageView,
    ) -> Result<Self> {
        // Create two history images for ping-pong temporal accumulation
        let mut history_images = [vk::Image::null(); 2];
        let mut history_image_memories = [vk::DeviceMemory::null(); 2];
        let mut history_image_views = [vk::ImageView::null(); 2];
        for i in 0..2 {
            let (img, mem) = create_image(
                instance,
                rrdevice,
                width,
                height,
                1,
                vk::SampleCountFlags::_1,
                FLAME_HISTORY_FORMAT,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            history_images[i] = img;
            history_image_memories[i] = mem;
            history_image_views[i] = create_image_view(
                rrdevice,
                img,
                FLAME_HISTORY_FORMAT,
                vk::ImageAspectFlags::COLOR,
                1,
            )?;
        }

        // Clear both history images to zero using a single-time command buffer
        let cmd = begin_single_time_commands(rrdevice, command_pool)?;
        for i in 0..2 {
            // Barrier: UNDEFINED -> TRANSFER_DST_OPTIMAL
            let barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(history_images[i])
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            rrdevice.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[barrier],
            );
            // Clear with float32 [0,0,0,0]
            let clear_value = vk::ClearColorValue {
                float32: [0.0f32; 4],
            };
            rrdevice.device.cmd_clear_color_image(
                cmd,
                history_images[i],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_value,
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            );
            // Barrier: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
            let barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(history_images[i])
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            rrdevice.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[barrier],
            );
        }
        end_single_time_commands(rrdevice, rrdevice.graphics_queue, command_pool, cmd)?;

        let shading_render_pass = Self::create_shading_render_pass(rrdevice)?;

        // F2 render pass: A0 = HDR (LOAD + premultiplied blend), A1 = history (CLEAR/DONT_CARE + no blend)
        // Create two framebuffers for ping-pong swap: framebuffer i writes to history[i], reads from history[1-i]
        let mut shading_framebuffers = [vk::Framebuffer::null(); 2];
        for i in 0..2 {
            let shading_attachments = [hdr_image_view, history_image_views[i]];
            let shading_framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(shading_render_pass)
                .attachments(&shading_attachments)
                .width(width)
                .height(height)
                .layers(1);
            shading_framebuffers[i] = rrdevice
                .device
                .create_framebuffer(&shading_framebuffer_info, None)?;
        }

        let sampler = Self::create_sampler(&rrdevice.device)?;

        log!(
            "Created flame buffer: {}x{} history format {:?}",
            width,
            height,
            FLAME_HISTORY_FORMAT
        );

        Ok(Self {
            history_images,
            history_image_memories,
            history_image_views,
            sampler,
            shading_render_pass,
            shading_framebuffers,
            width,
            height,
        })
    }

    unsafe fn create_shading_render_pass(rrdevice: &RRDevice) -> Result<vk::RenderPass> {
        // A0 = HDR (LOAD + premultiplied blend, existing behavior)
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

        // A1 = history (CLEAR/DONT_CARE + no blend)
        let history_attachment = vk::AttachmentDescription::builder()
            .format(FLAME_HISTORY_FORMAT)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();

        let history_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();

        let color_attachments = [color_attachment_ref, history_attachment_ref];

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

        let attachments = [color_attachment, history_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency_in, dependency_out];

        let info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        Ok(rrdevice.device.create_render_pass(&info, None)?)
    }

    unsafe fn create_sampler(device: &vulkanalia::Device) -> Result<vk::Sampler> {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        Ok(device.create_sampler(&sampler_info, None)?)
    }

    pub unsafe fn resize(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        command_pool: vk::CommandPool,
        new_width: u32,
        new_height: u32,
        hdr_image_view: vk::ImageView,
    ) -> Result<()> {
        self.destroy(&rrdevice.device);
        *self = Self::new(
            instance,
            rrdevice,
            command_pool,
            new_width,
            new_height,
            hdr_image_view,
        )?;

        log!("Resized flame buffer to: {}x{}", new_width, new_height);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.sampler != vk::Sampler::null() {
            device.destroy_sampler(self.sampler, None);
            self.sampler = vk::Sampler::null();
        }
        for i in 0..2 {
            if self.shading_framebuffers[i] != vk::Framebuffer::null() {
                device.destroy_framebuffer(self.shading_framebuffers[i], None);
                self.shading_framebuffers[i] = vk::Framebuffer::null();
            }
        }
        if self.shading_render_pass != vk::RenderPass::null() {
            device.destroy_render_pass(self.shading_render_pass, None);
            self.shading_render_pass = vk::RenderPass::null();
        }

        // Destroy history images (ping-pong temporal accumulation)
        for i in 0..2 {
            if self.history_image_views[i] != vk::ImageView::null() {
                device.destroy_image_view(self.history_image_views[i], None);
                self.history_image_views[i] = vk::ImageView::null();
            }
            if self.history_images[i] != vk::Image::null() {
                device.destroy_image(self.history_images[i], None);
                self.history_images[i] = vk::Image::null();
            }
            if self.history_image_memories[i] != vk::DeviceMemory::null() {
                device.free_memory(self.history_image_memories[i], None);
                self.history_image_memories[i] = vk::DeviceMemory::null();
            }
        }

        log!("Destroyed flame buffer");
    }

    pub fn extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }
}

impl Drop for FlameBuffer {
    fn drop(&mut self) {
        if self.history_images[0] != vk::Image::null() {
            log_warn!("FlameBuffer dropped without calling destroy()");
        }
    }
}
