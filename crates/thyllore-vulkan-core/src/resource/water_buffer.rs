use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::command::{begin_single_time_commands, end_single_time_commands};
use crate::core::RRDevice;
use crate::resource::hdr_buffer::HDR_FORMAT;
use crate::resource::image::{create_image, create_image_view};

#[derive(Clone, Debug, Default)]
pub struct WaterBuffer {
    pub render_pass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    pub framebuffers: [vk::Framebuffer; 2],
    pub width: u32,
    pub height: u32,
    pub scene_color_image: vk::Image,
    pub scene_color_image_memory: vk::DeviceMemory,
    pub scene_color_image_view: vk::ImageView,
    pub scene_color_sampler: vk::Sampler,
    pub history_images: [vk::Image; 2],
    pub history_image_memories: [vk::DeviceMemory; 2],
    pub history_image_views: [vk::ImageView; 2],
    pub history_sampler: vk::Sampler,
    pub trace_image: vk::Image,
    pub trace_memory: vk::DeviceMemory,
    pub trace_image_view: vk::ImageView,
    pub caustic_accum_image: vk::Image,
    pub caustic_accum_memory: vk::DeviceMemory,
    pub caustic_accum_view: vk::ImageView,
}

impl WaterBuffer {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
        hdr_image_view: vk::ImageView,
        depth_image_view: vk::ImageView,
    ) -> Result<Self> {
        let render_pass = Self::create_shading_render_pass(rrdevice)?;

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
                HDR_FORMAT,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            history_images[i] = img;
            history_image_memories[i] = mem;
            history_image_views[i] =
                create_image_view(rrdevice, img, HDR_FORMAT, vk::ImageAspectFlags::COLOR, 1)?;
        }

        // Create two framebuffers (one per history image)
        let mut framebuffers = [vk::Framebuffer::null(); 2];
        for i in 0..2 {
            let attachments = [hdr_image_view, history_image_views[i], depth_image_view];
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(width)
                .height(height)
                .layers(1);
            framebuffers[i] = rrdevice
                .device
                .create_framebuffer(&framebuffer_info, None)?;
        }

        // Keep framebuffer field for compatibility (same as framebuffers[0])
        let framebuffer = framebuffers[0];

        // Create scene color image (TRANSFER_DST | SAMPLED)
        let (scene_color_image, scene_color_image_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            vk::SampleCountFlags::_1,
            HDR_FORMAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let scene_color_image_view = create_image_view(
            rrdevice,
            scene_color_image,
            HDR_FORMAT,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        let scene_color_sampler = Self::create_scene_color_sampler(rrdevice)?;

        // Create trace image (rgba16f, STORAGE | SAMPLED)
        let (trace_image, trace_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            vk::SampleCountFlags::_1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let trace_image_view = create_image_view(
            rrdevice,
            trace_image,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        // Create caustic accum image (r32uint, STORAGE | TRANSFER_DST | TRANSFER_SRC)
        let (caustic_accum_image, caustic_accum_memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            vk::SampleCountFlags::_1,
            vk::Format::R32_UINT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let caustic_accum_view = create_image_view(
            rrdevice,
            caustic_accum_image,
            vk::Format::R32_UINT,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        // Clear both history images so the first frame samples zero in SHADER_READ_ONLY layout
        let cmd = begin_single_time_commands(rrdevice, command_pool)?;
        let color_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        for &image in &history_images {
            let to_transfer = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(color_range)
                .build();
            rrdevice.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[to_transfer],
            );

            let clear_value = vk::ClearColorValue {
                float32: [0.0f32; 4],
            };
            rrdevice.device.cmd_clear_color_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_value,
                &[color_range],
            );

            let to_sampled = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(color_range)
                .build();
            rrdevice.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[to_sampled],
            );
        }

        // Transition trace image to GENERAL layout
        let barrier = vk::ImageMemoryBarrier::builder()
            .image(trace_image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
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
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        // Transition caustic accum image to GENERAL layout
        let barrier = vk::ImageMemoryBarrier::builder()
            .image(caustic_accum_image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
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
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );
        end_single_time_commands(rrdevice, rrdevice.graphics_queue, command_pool, cmd)?;

        log!("Created water buffer: {}x{}", width, height);

        Ok(Self {
            render_pass,
            framebuffer,
            framebuffers,
            width,
            height,
            scene_color_image,
            scene_color_image_memory,
            scene_color_image_view,
            scene_color_sampler,
            history_images,
            history_image_memories,
            history_image_views,
            history_sampler: Self::create_scene_color_sampler(rrdevice)?,
            trace_image,
            trace_memory,
            trace_image_view,
            caustic_accum_image,
            caustic_accum_memory,
            caustic_accum_view,
        })
    }

    unsafe fn create_shading_render_pass(rrdevice: &RRDevice) -> Result<vk::RenderPass> {
        // A0 = HDR (LOAD + no blend, same as hdr_buffer.rs composite pass)
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

        // A1 = history (DONT_CARE load + STORE, for temporal accumulation)
        let history_attachment = vk::AttachmentDescription::builder()
            .format(HDR_FORMAT)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        // A2 = depth D32_SFLOAT (LOAD/STORE, same as hdr_buffer.rs lines 114-122)
        let depth_attachment = vk::AttachmentDescription::builder()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
            .build();

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();

        let history_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();

        let depth_attachment_ref = vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let color_attachments = [color_attachment_ref, history_attachment_ref];

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachments)
            .depth_stencil_attachment(&depth_attachment_ref);

        let dependency_in = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .build();

        let dependency_out = vk::SubpassDependency::builder()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .dst_stage_mask(
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();

        let attachments = [color_attachment, history_attachment, depth_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency_in, dependency_out];

        let info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        Ok(rrdevice.device.create_render_pass(&info, None)?)
    }

    unsafe fn create_scene_color_sampler(rrdevice: &RRDevice) -> Result<vk::Sampler> {
        let address_mode = vk::SamplerAddressMode::CLAMP_TO_EDGE;
        let info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(address_mode)
            .address_mode_v(address_mode)
            .address_mode_w(address_mode)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
            .anisotropy_enable(false)
            .max_anisotropy(1.0);

        Ok(rrdevice.device.create_sampler(&info, None)?)
    }

    pub unsafe fn resize(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        command_pool: vk::CommandPool,
        new_width: u32,
        new_height: u32,
        hdr_image_view: vk::ImageView,
        depth_image_view: vk::ImageView,
    ) -> Result<()> {
        self.destroy(&rrdevice.device);
        *self = Self::new(
            instance,
            rrdevice,
            command_pool,
            new_width,
            new_height,
            hdr_image_view,
            depth_image_view,
        )?;

        log!("Resized water buffer to: {}x{}", new_width, new_height);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        // Destroy both framebuffers (framebuffer field is same as framebuffers[0])
        for i in 0..2 {
            if self.framebuffers[i] != vk::Framebuffer::null() {
                device.destroy_framebuffer(self.framebuffers[i], None);
                self.framebuffers[i] = vk::Framebuffer::null();
            }
        }
        self.framebuffer = vk::Framebuffer::null();

        if self.render_pass != vk::RenderPass::null() {
            device.destroy_render_pass(self.render_pass, None);
            self.render_pass = vk::RenderPass::null();
        }

        // Destroy scene color resources
        if self.scene_color_sampler != vk::Sampler::null() {
            device.destroy_sampler(self.scene_color_sampler, None);
            self.scene_color_sampler = vk::Sampler::null();
        }
        if self.scene_color_image_view != vk::ImageView::null() {
            device.destroy_image_view(self.scene_color_image_view, None);
            self.scene_color_image_view = vk::ImageView::null();
        }
        if self.scene_color_image != vk::Image::null() {
            device.destroy_image(self.scene_color_image, None);
            self.scene_color_image = vk::Image::null();
        }
        if self.scene_color_image_memory != vk::DeviceMemory::null() {
            device.free_memory(self.scene_color_image_memory, None);
            self.scene_color_image_memory = vk::DeviceMemory::null();
        }

        // Destroy history resources
        if self.history_sampler != vk::Sampler::null() {
            device.destroy_sampler(self.history_sampler, None);
            self.history_sampler = vk::Sampler::null();
        }
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

        // Destroy trace image resources
        // Destroy trace image resources
        if self.trace_image_view != vk::ImageView::null() {
            device.destroy_image_view(self.trace_image_view, None);
            self.trace_image_view = vk::ImageView::null();
        }
        if self.trace_image != vk::Image::null() {
            device.destroy_image(self.trace_image, None);
            self.trace_image = vk::Image::null();
        }
        if self.trace_memory != vk::DeviceMemory::null() {
            device.free_memory(self.trace_memory, None);
            self.trace_memory = vk::DeviceMemory::null();
        }

        // Destroy caustic accum image resources
        if self.caustic_accum_view != vk::ImageView::null() {
            device.destroy_image_view(self.caustic_accum_view, None);
            self.caustic_accum_view = vk::ImageView::null();
        }
        if self.caustic_accum_image != vk::Image::null() {
            device.destroy_image(self.caustic_accum_image, None);
            self.caustic_accum_image = vk::Image::null();
        }
        if self.caustic_accum_memory != vk::DeviceMemory::null() {
            device.free_memory(self.caustic_accum_memory, None);
            self.caustic_accum_memory = vk::DeviceMemory::null();
        }

        log!("Destroyed water buffer");
    }

    pub fn extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }

    pub fn scene_color_binding(&self) -> (vk::ImageView, vk::Sampler) {
        (self.scene_color_image_view, self.scene_color_sampler)
    }
}

impl Drop for WaterBuffer {
    fn drop(&mut self) {
        let mut has_resources = false;
        if self.framebuffer != vk::Framebuffer::null() {
            has_resources = true;
        }
        for i in 0..2 {
            if self.framebuffers[i] != vk::Framebuffer::null() {
                has_resources = true;
                break;
            }
        }
        if has_resources {
            log_warn!("WaterBuffer dropped without calling destroy()");
        }
    }
}
