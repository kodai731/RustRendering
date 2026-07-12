use crate::core::device::*;
use crate::core::swapchain::*;
use crate::render::pass::RRRender;
use crate::resource::image::*;
use crate::vulkan::*;
use anyhow::Result;

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
    println!("created framebuffers");
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
    println!("created color objects");
    Ok(())
}
