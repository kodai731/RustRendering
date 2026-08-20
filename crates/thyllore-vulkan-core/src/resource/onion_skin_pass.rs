use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::core::RRDevice;
use crate::descriptor::shader_bindings::onion_skin_composite;
use crate::descriptor::{onion_skin_composite_layout_spec, ReflectedSetLayout};
use crate::pipeline::RRPipeline;
use crate::resource::image::{create_image, create_image_view};

pub const GHOST_BUFFER_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

#[derive(Clone, Debug, Default)]
pub struct OnionSkinPassResources {
    pub ghost_image: vk::Image,
    pub ghost_image_memory: vk::DeviceMemory,
    pub ghost_image_view: vk::ImageView,
    pub ghost_sampler: vk::Sampler,

    pub ghost_render_pass: vk::RenderPass,
    pub ghost_framebuffer: vk::Framebuffer,
    pub ghost_pipeline: RRPipeline,

    pub composite_render_pass: vk::RenderPass,
    pub composite_framebuffer: vk::Framebuffer,
    pub composite_pipeline: RRPipeline,
    pub composite_descriptor_layout: ReflectedSetLayout,
    pub composite_descriptor_set: vk::DescriptorSet,

    pub width: u32,
    pub height: u32,
}

impl OnionSkinPassResources {
    pub unsafe fn create_ghost_buffer(
        instance: &Instance,
        rrdevice: &RRDevice,
        width: u32,
        height: u32,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler)> {
        let (image, memory) = create_image(
            instance,
            rrdevice,
            width,
            height,
            1,
            vk::SampleCountFlags::_1,
            GHOST_BUFFER_FORMAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let image_view = create_image_view(
            rrdevice,
            image,
            GHOST_BUFFER_FORMAT,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        let sampler = Self::create_ghost_sampler(&rrdevice.device)?;

        Ok((image, memory, image_view, sampler))
    }

    unsafe fn create_ghost_sampler(device: &vulkanalia::Device) -> Result<vk::Sampler> {
        let info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(1.0);

        let sampler = device.create_sampler(&info, None)?;
        Ok(sampler)
    }

    pub unsafe fn create_ghost_render_pass(rrdevice: &RRDevice) -> Result<vk::RenderPass> {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(GHOST_BUFFER_FORMAT)
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
            .dst_stage_mask(
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            )
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();

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

    pub unsafe fn create_composite_render_pass(
        rrdevice: &RRDevice,
        offscreen_format: vk::Format,
    ) -> Result<vk::RenderPass> {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(offscreen_format)
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
            .dst_stage_mask(
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            )
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();

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

    pub unsafe fn create_composite_descriptor(
        rrdevice: &RRDevice,
        ghost_image_view: vk::ImageView,
        ghost_sampler: vk::Sampler,
    ) -> Result<(ReflectedSetLayout, vk::DescriptorSet)> {
        let layout = ReflectedSetLayout::create(rrdevice, &onion_skin_composite_layout_spec())?;
        let descriptor_set = layout.allocate_set(rrdevice)?;

        Self::update_composite_descriptor(
            rrdevice,
            &layout,
            descriptor_set,
            ghost_image_view,
            ghost_sampler,
        )?;

        Ok((layout, descriptor_set))
    }

    pub unsafe fn update_composite_descriptor(
        rrdevice: &RRDevice,
        layout: &ReflectedSetLayout,
        descriptor_set: vk::DescriptorSet,
        ghost_image_view: vk::ImageView,
        ghost_sampler: vk::Sampler,
    ) -> Result<()> {
        layout
            .writer(descriptor_set)
            .image(
                onion_skin_composite::GHOST_SAMPLER,
                ghost_image_view,
                ghost_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn create_single_framebuffer(
        rrdevice: &RRDevice,
        render_pass: vk::RenderPass,
        image_view: vk::ImageView,
        width: u32,
        height: u32,
    ) -> Result<vk::Framebuffer> {
        let attachments = [image_view];

        let info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);

        let framebuffer = rrdevice.device.create_framebuffer(&info, None)?;
        Ok(framebuffer)
    }

    pub unsafe fn destroy(&self, device: &vulkanalia::Device) {
        device.destroy_framebuffer(self.composite_framebuffer, None);
        self.composite_pipeline.destroy(device);
        device.destroy_render_pass(self.composite_render_pass, None);
        device.destroy_descriptor_set_layout(self.composite_descriptor_layout.handle, None);

        device.destroy_framebuffer(self.ghost_framebuffer, None);
        self.ghost_pipeline.destroy(device);
        device.destroy_render_pass(self.ghost_render_pass, None);

        device.destroy_sampler(self.ghost_sampler, None);
        device.destroy_image_view(self.ghost_image_view, None);
        device.destroy_image(self.ghost_image, None);
        device.free_memory(self.ghost_image_memory, None);

        log!("Destroyed onion skin pass resources");
    }

    pub unsafe fn recreate_on_resize(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        offscreen_resolve_image_view: vk::ImageView,
        width: u32,
        height: u32,
    ) -> Result<()> {
        rrdevice
            .device
            .destroy_framebuffer(self.ghost_framebuffer, None);
        rrdevice
            .device
            .destroy_framebuffer(self.composite_framebuffer, None);

        rrdevice.device.destroy_sampler(self.ghost_sampler, None);
        rrdevice
            .device
            .destroy_image_view(self.ghost_image_view, None);
        rrdevice.device.destroy_image(self.ghost_image, None);
        rrdevice.device.free_memory(self.ghost_image_memory, None);

        let (image, memory, view, sampler) =
            Self::create_ghost_buffer(instance, rrdevice, width, height)?;
        self.ghost_image = image;
        self.ghost_image_memory = memory;
        self.ghost_image_view = view;
        self.ghost_sampler = sampler;

        self.ghost_framebuffer =
            Self::create_single_framebuffer(rrdevice, self.ghost_render_pass, view, width, height)?;

        self.composite_framebuffer = Self::create_single_framebuffer(
            rrdevice,
            self.composite_render_pass,
            offscreen_resolve_image_view,
            width,
            height,
        )?;

        Self::update_composite_descriptor(
            rrdevice,
            &self.composite_descriptor_layout,
            self.composite_descriptor_set,
            self.ghost_image_view,
            self.ghost_sampler,
        )?;

        self.width = width;
        self.height = height;

        log!("Recreated onion skin resources: {}x{}", width, height);
        Ok(())
    }
}
