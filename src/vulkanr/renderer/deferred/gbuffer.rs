use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::vulkanr::core::RRDevice;
use crate::vulkanr::render::pass::get_depth_format;
use crate::vulkanr::render::RRRender;
use crate::vulkanr::resource::{create_image, create_image_view, RRGBuffer};

pub use thyllore_vulkan_core::renderer::push_constants::{
    GBufferPushConstants, OnionSkinPushConstants,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbuffer_push_constants_size() {
        assert_eq!(std::mem::size_of::<GBufferPushConstants>(), 8);
    }

    #[test]
    fn test_gbuffer_push_constants_as_bytes() {
        let pc = GBufferPushConstants::new(42, 1);
        let bytes = pc.as_bytes();
        assert_eq!(bytes.len(), 8);
        let object_id = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(object_id, 42);
        let heatmap_mode = u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(heatmap_mode, 1);
    }

    #[test]
    fn test_onion_skin_push_constants_size() {
        assert_eq!(std::mem::size_of::<OnionSkinPushConstants>(), 32);
    }

    #[test]
    fn test_onion_skin_push_constants_values() {
        let pc = OnionSkinPushConstants::new([0.2, 0.4, 1.0], 0.5);
        assert!((pc.ghost_tint_r - 0.2).abs() < f32::EPSILON);
        assert!((pc.ghost_tint_g - 0.4).abs() < f32::EPSILON);
        assert!((pc.ghost_tint_b - 1.0).abs() < f32::EPSILON);
        assert!((pc.ghost_opacity - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_onion_skin_push_constants_as_bytes() {
        let pc = OnionSkinPushConstants::new([1.0, 2.0, 3.0], 4.0);
        let bytes = pc.as_bytes();
        assert_eq!(bytes.len(), 32);
        let tint_r = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert!((tint_r - 1.0).abs() < f32::EPSILON);
    }
}

pub unsafe fn create_gbuffer_framebuffer(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrrender: &mut RRRender,
    gbuffer: &RRGBuffer,
) -> Result<()> {
    let (depth_image, depth_image_memory) = create_image(
        instance,
        rrdevice,
        gbuffer.width,
        gbuffer.height,
        1,
        vk::SampleCountFlags::_1,
        get_depth_format(instance, rrdevice)?,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let depth_image_view = create_image_view(
        rrdevice,
        depth_image,
        get_depth_format(instance, rrdevice)?,
        vk::ImageAspectFlags::DEPTH,
        1,
    )?;

    rrrender.gbuffer_depth_image = depth_image;
    rrrender.gbuffer_depth_image_memory = depth_image_memory;
    rrrender.gbuffer_depth_image_view = depth_image_view;

    let attachments = [
        gbuffer.position_image_view,
        gbuffer.normal_image_view,
        gbuffer.albedo_image_view,
        gbuffer.object_id_image_view,
        depth_image_view,
    ];

    let info = vk::FramebufferCreateInfo::builder()
        .render_pass(rrrender.gbuffer_render_pass)
        .attachments(&attachments)
        .width(gbuffer.width)
        .height(gbuffer.height)
        .layers(1);

    rrrender.gbuffer_framebuffer = rrdevice.device.create_framebuffer(&info, None)?;

    log!(
        "Created G-Buffer framebuffer: {}x{}",
        gbuffer.width,
        gbuffer.height
    );
    Ok(())
}
