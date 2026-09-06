use vulkanalia::prelude::v1_0::*;

pub(crate) fn full_extent_scissor(extent: vk::Extent2D) -> vk::Rect2D {
    vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(extent)
        .build()
}
