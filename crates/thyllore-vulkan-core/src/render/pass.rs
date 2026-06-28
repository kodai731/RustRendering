use crate::command::*;
use crate::core::device::*;
use crate::core::swapchain::*;
use crate::render::framebuffer::{create_color_objects, create_framebuffers};
use crate::resource::image::*;
use crate::vulkan::*;
#[derive(Clone, Debug, Default)]
pub struct RRRender {
    // Main render pass (for final presentation)
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    pub color_image: vk::Image,
    pub color_image_view: vk::ImageView,
    pub color_image_memory: vk::DeviceMemory,

    // G-Buffer render pass (for deferred rendering)
    pub gbuffer_render_pass: vk::RenderPass,
    pub gbuffer_framebuffer: vk::Framebuffer,
    pub gbuffer_depth_image: vk::Image,
    pub gbuffer_depth_image_memory: vk::DeviceMemory,
    pub gbuffer_depth_image_view: vk::ImageView,
}

impl RRRender {
    pub unsafe fn destroy_size_dependent(&self, device: &crate::core::device::Device) {
        for &fb in &self.framebuffers {
            device.destroy_framebuffer(fb, None);
        }

        device.destroy_image_view(self.depth_image_view, None);
        device.free_memory(self.depth_image_memory, None);
        device.destroy_image(self.depth_image, None);

        device.destroy_image_view(self.color_image_view, None);
        device.free_memory(self.color_image_memory, None);
        device.destroy_image(self.color_image, None);
    }

    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        rrcommand_pool: &RRCommandPool,
    ) -> Self {
        println!("start to create render ...");
        let mut rrrender = RRRender::default();
        if let Err(e) = create_render_pass(instance, rrdevice, rrswapchain, &mut rrrender) {
            eprintln!("Create render pass failed {:?}", e);
        }
        if let Err(e) = create_depth_objects(
            instance,
            rrdevice,
            rrswapchain,
            rrcommand_pool,
            &mut rrrender,
        ) {
            eprintln!("Create render pass failed {:?}", e);
        }
        if let Err(e) = create_color_objects(instance, rrdevice, rrswapchain, &mut rrrender) {
            eprintln!("Create color objects failed {:?}", e);
        }
        if let Err(e) = create_framebuffers(rrdevice, rrswapchain, &mut rrrender) {
            eprintln!("Create framebuffers failed {:?}", e);
        }
        if let Err(e) = create_gbuffer_render_pass(instance, rrdevice, &mut rrrender) {
            eprintln!("Create G-Buffer render pass failed {:?}", e);
        }
        println!("created render pass {:?}", rrrender);
        rrrender
    }
}

unsafe fn create_render_pass(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrswapchain: &RRSwapchain,
    rrrender: &mut RRRender,
) -> Result<()> {
    let final_color_layout = if rrdevice.present_queue.is_null() {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::PRESENT_SRC_KHR
    };

    let color_attachment = vk::AttachmentDescription::builder()
        .format(rrswapchain.swapchain_format)
        .samples(rrdevice.msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(final_color_layout);

    let color_resolve_attachment = vk::AttachmentDescription::builder()
        .format(rrswapchain.swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(final_color_layout);

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_resolve_attachement_ref = vk::AttachmentReference::builder()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
    let resolve_attachments = &[color_resolve_attachement_ref];

    let depth_stencil_attachment = vk::AttachmentDescription::builder()
        .format(get_depth_format(instance, rrdevice)?)
        .samples(rrdevice.msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments)
        .depth_stencil_attachment(&depth_stencil_attachment_ref)
        .resolve_attachments(resolve_attachments);

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

    let attachments = &[
        color_attachment,
        depth_stencil_attachment,
        color_resolve_attachment,
    ];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    rrrender.render_pass = rrdevice.device.create_render_pass(&info, None)?;
    println!("render pass {:?}", rrrender);
    Ok(())
}

pub unsafe fn get_depth_format(instance: &Instance, rrdevice: &RRDevice) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    get_suppoted_format(
        instance,
        rrdevice,
        candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

unsafe fn get_suppoted_format(
    instance: &Instance,
    rrdevice: &RRDevice,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    candidates
        .iter()
        .cloned()
        .find(|f| {
            let properties =
                instance.get_physical_device_format_properties(rrdevice.physical_device, *f);
            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .ok_or_else(|| anyhow!("Failed to find supported format"))
}

pub unsafe fn create_depth_objects(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrswapchain: &RRSwapchain,
    rrcommand_buffer: &RRCommandPool,
    rrrender: &mut RRRender,
) -> Result<()> {
    let format = get_depth_format(instance, rrdevice)?;
    let (depth_image, depth_image_memory) = create_image(
        instance,
        rrdevice,
        rrswapchain.swapchain_extent.width,
        rrswapchain.swapchain_extent.height,
        1,
        rrdevice.msaa_samples,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    rrrender.depth_image = depth_image;
    rrrender.depth_image_memory = depth_image_memory;
    rrrender.depth_image_view = create_image_view(
        rrdevice,
        rrrender.depth_image,
        format,
        vk::ImageAspectFlags::DEPTH,
        1,
    )?;

    transition_image_layout(
        rrdevice,
        rrdevice.graphics_queue,
        rrcommand_buffer.command_pool,
        rrrender.depth_image,
        format,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        1,
    )?;
    println!("created depth object");
    Ok(())
}

pub unsafe fn create_gbuffer_render_pass(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrrender: &mut RRRender,
) -> Result<()> {
    let position_attachment = vk::AttachmentDescription::builder()
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::GENERAL)
        .build();

    let normal_attachment = vk::AttachmentDescription::builder()
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::GENERAL)
        .build();

    let albedo_attachment = vk::AttachmentDescription::builder()
        .format(vk::Format::R8G8B8A8_UNORM)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .build();

    let object_id_attachment = vk::AttachmentDescription::builder()
        .format(vk::Format::R32_UINT)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .build();

    let depth_attachment = vk::AttachmentDescription::builder()
        .format(get_depth_format(instance, rrdevice)?)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        .build();

    let position_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();

    let normal_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();

    let albedo_attachment_ref = vk::AttachmentReference::builder()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();

    let object_id_attachment_ref = vk::AttachmentReference::builder()
        .attachment(3)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();

    let color_attachments = [
        position_attachment_ref,
        normal_attachment_ref,
        albedo_attachment_ref,
        object_id_attachment_ref,
    ];

    let depth_attachment_ref = vk::AttachmentReference::builder()
        .attachment(4)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachments)
        .depth_stencil_attachment(&depth_attachment_ref);

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
        position_attachment,
        normal_attachment,
        albedo_attachment,
        object_id_attachment,
        depth_attachment,
    ];
    let subpasses = [subpass];
    let dependencies = [dependency];

    let info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    rrrender.gbuffer_render_pass = rrdevice.device.create_render_pass(&info, None)?;

    log::info!("Created G-Buffer render pass with ObjectID attachment");
    Ok(())
}
