use super::command::*;
use super::device::*;
use super::image::*;
use super::swapchain::*;
use super::vulkan::*;
#[derive(Clone, Debug, Default)]
pub struct RRRender {
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    pub color_image: vk::Image,
    pub color_image_view: vk::ImageView,
    pub color_image_memory: vk::DeviceMemory,
}

impl RRRender {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        rrcommand_pool: &RRCommandPool,
    ) -> Self {
        let mut rrrender = RRRender::default();
        let _ = create_render_pass(instance, rrdevice, rrswapchain, &mut rrrender);
        let _ = create_depth_objects(
            instance,
            rrdevice,
            rrswapchain,
            rrcommand_pool,
            &mut rrrender,
        );
        let _ = create_framebuffers(rrdevice, rrswapchain, &mut rrrender);
        let _ = create_color_objects(instance, rrdevice, rrswapchain, &mut rrrender);
        rrrender
    }
}

unsafe fn create_render_pass(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrswapchain: &RRSwapchain,
    rrrender: &mut RRRender,
) -> Result<()> {
    // we need to tell Vulkan about the framebuffer attachments that will be used while rendering.
    // We need to specify how many color and depth buffers there will be, how many samples to use for each of them and how their contents should be handled throughout the rendering operations.
    // All of this information is wrapped in a render pass object
    let color_attachment = vk::AttachmentDescription::builder()
        .format(rrswapchain.swapchain_format)
        .samples(rrdevice.msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE) // for stencil buffer
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE) // for stencil buffer
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // That's because multisampled images cannot be presented directly.
    // We first need to resolve them to a regular image.
    let color_resolve_attachment = vk::AttachmentDescription::builder()
        .format(rrswapchain.swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    //  Subpasses are subsequent rendering operations that depend on the contents of framebuffers in previous passes
    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_resolve_attachement_ref = vk::AttachmentReference::builder()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    // The index of the attachment in this array is directly referenced from the fragment shader with the layout(location = 0) out vec4 outColor directive!
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

    // The subpasses in a render pass automatically take care of image layout transitions.
    // These transitions are controlled by subpass dependencies, which specify memory and execution dependencies between subpasses
    // The depth image is first accessed in the early fragment test pipeline stage
    // and because we have a load operation that clears, we should specify the access mask for writes.
    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        ) //  wait for the swapchain to finish reading from the image before we can access it
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
    Ok(())
}

unsafe fn get_depth_format(instance: &Instance, rrdevice: &RRDevice) -> Result<vk::Format> {
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

unsafe fn create_depth_objects(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrswapchain: &RRSwapchain,
    rrcommand_buffer: &RRCommandPool,
    rrrender: &mut RRRender,
) -> Result<()> {
    // The stencil component is used for stencil tests, which is an additional test that can be combined with depth testing.
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

    Ok(())
}

pub unsafe fn create_framebuffers(
    rrdevice: &RRDevice,
    rrswapchain: &RRSwapchain,
    rrrender: &mut RRRender,
) -> Result<()> {
    rrrender.framebuffers = rrswapchain
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[rrrender.color_image_view, rrrender.depth_image_view, *i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(rrrender.render_pass) // they use the same number and type of attachments.
                .attachments(attachments)
                .width(rrswapchain.swapchain_extent.width)
                .height(rrswapchain.swapchain_extent.height)
                .layers(1);
            rrdevice.device.create_framebuffer(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

pub unsafe fn create_color_objects(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrswapchain: &RRSwapchain,
    rrrender: &mut RRRender,
) -> Result<()> {
    //  this color buffer doesn't need mipmaps since it's not going to be used as a texture:
    let (color_image, color_image_memory) = create_image(
        instance,
        rrdevice,
        rrswapchain.swapchain_extent.width,
        rrswapchain.swapchain_extent.height,
        1,
        rrdevice.msaa_samples,
        rrswapchain.swapchain_format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    rrrender.color_image = color_image;
    rrrender.color_image_memory = color_image_memory;

    rrrender.color_image_view = create_image_view(
        rrdevice,
        rrrender.color_image,
        rrswapchain.swapchain_format,
        vk::ImageAspectFlags::COLOR,
        1,
    )?;

    Ok(())
}
