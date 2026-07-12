use crate::ecs::resource::billboard::BillboardData;
use crate::ecs::resource::{Camera, LightState};
use crate::ecs::systems::camera_systems::{
    compute_camera_direction, compute_camera_position, compute_camera_up,
};
use crate::ecs::world::World;
use crate::vulkanr::context::SwapchainState;
use crate::vulkanr::descriptor::RRBillboardDescriptorSet;
use crate::vulkanr::swapchain::RRSwapchain;
use cgmath::{Deg, Vector3};
use thyllore_vulkan_core::resource::raytracing_data::RayTracingData;
use vulkanalia::prelude::v1_0::*;

pub fn collect_and_log_billboard_debug(world: &World, raytracing: &RayTracingData) {
    let camera = world.resource::<Camera>();
    let light = world.resource::<LightState>();

    let info = BillboardDebugInfo {
        light_position: light.light_position,
        camera_position: compute_camera_position(&camera),
        camera_direction: compute_camera_direction(&camera),
        camera_up: compute_camera_up(&camera),
        near_plane: camera.near_plane,
        fov_y: camera.fov_y,
    };

    let gbuffer_debug_info = raytracing.gbuffer.as_ref().map(|gb| GBufferDebugInfo {
        position_image_view: gb.position_image_view,
        extent_width: gb.width,
        extent_height: gb.height,
    });

    let swapchain = &world.resource::<SwapchainState>().swapchain;
    let billboard = world.resource::<BillboardData>();

    log_billboard_debug_info(
        &info,
        swapchain,
        &billboard.render_state.descriptor_set,
        gbuffer_debug_info.as_ref(),
        raytracing.gbuffer_sampler,
    );
}

pub struct BillboardDebugInfo {
    pub light_position: Vector3<f32>,
    pub camera_position: Vector3<f32>,
    pub camera_direction: Vector3<f32>,
    pub camera_up: Vector3<f32>,
    pub near_plane: f32,
    pub fov_y: Deg<f32>,
}

pub struct GBufferDebugInfo {
    pub position_image_view: vk::ImageView,
    pub extent_width: u32,
    pub extent_height: u32,
}

pub fn log_billboard_debug_info(
    info: &BillboardDebugInfo,
    swapchain: &RRSwapchain,
    billboard_descriptor_set: &RRBillboardDescriptorSet,
    gbuffer_info: Option<&GBufferDebugInfo>,
    gbuffer_sampler: Option<vk::Sampler>,
) {
    log!("=== Billboard Depth Debug Info ===");

    log_basic_info(
        info,
        swapchain,
        billboard_descriptor_set,
        gbuffer_info,
        gbuffer_sampler,
    );
    log_view_to_screen_transforms(info, swapchain);

    log!("=================================");
}

fn log_basic_info(
    info: &BillboardDebugInfo,
    swapchain: &RRSwapchain,
    billboard_descriptor_set: &RRBillboardDescriptorSet,
    gbuffer_info: Option<&GBufferDebugInfo>,
    gbuffer_sampler: Option<vk::Sampler>,
) {
    log!(
        "Light position: ({:.2}, {:.2}, {:.2})",
        info.light_position.x,
        info.light_position.y,
        info.light_position.z
    );
    log!(
        "Camera position: ({:.2}, {:.2}, {:.2})",
        info.camera_position.x,
        info.camera_position.y,
        info.camera_position.z
    );
    log!(
        "Camera direction: ({:.3}, {:.3}, {:.3})",
        info.camera_direction.x,
        info.camera_direction.y,
        info.camera_direction.z
    );

    let swapchain_extent = swapchain.swapchain_extent;
    log!(
        "Swapchain extent: {}x{}",
        swapchain_extent.width,
        swapchain_extent.height
    );

    if let Some(gbuffer) = gbuffer_info {
        log!("GBuffer:");
        log!("  position_image_view: {:?}", gbuffer.position_image_view);
        log!(
            "  extent: {}x{}",
            gbuffer.extent_width,
            gbuffer.extent_height
        );

        let gbuffer_matches_swapchain = gbuffer.extent_width == swapchain_extent.width
            && gbuffer.extent_height == swapchain_extent.height;
        log!(
            "  GBuffer matches swapchain extent: {}",
            gbuffer_matches_swapchain
        );
    } else {
        log!("WARNING: No GBuffer!");
    }

    if let Some(sampler) = gbuffer_sampler {
        log!("GBuffer sampler: {:?}", sampler);
    } else {
        log!("WARNING: No GBuffer sampler!");
    }

    log!("Billboard Descriptor Set:");
    log!("  rrdata count: {}", billboard_descriptor_set.rrdata.len());
    log!(
        "  descriptor_sets count: {}",
        billboard_descriptor_set.descriptor_sets.len()
    );
}

fn log_view_to_screen_transforms(info: &BillboardDebugInfo, swapchain: &RRSwapchain) {
    use crate::math::{coordinate_system::perspective_infinite_reverse, view};

    let swapchain_extent = swapchain.swapchain_extent;

    let view_matrix = unsafe { view(info.camera_position, info.camera_direction, info.camera_up) };
    let light_view_pos = view_matrix
        * cgmath::Vector4::new(
            info.light_position.x,
            info.light_position.y,
            info.light_position.z,
            1.0,
        );
    let billboard_view_depth = -light_view_pos.z;

    log!("Billboard (light) in view space:");
    log!(
        "  view_pos: ({:.2}, {:.2}, {:.2})",
        light_view_pos.x,
        light_view_pos.y,
        light_view_pos.z
    );
    log!("  view_depth (=-view_pos.z): {:.4}", billboard_view_depth);

    let aspect = swapchain_extent.width as f32 / swapchain_extent.height as f32;
    let proj = perspective_infinite_reverse(info.fov_y, aspect, info.near_plane);

    let light_clip_pos = proj * light_view_pos;
    let light_ndc = if light_clip_pos.w.abs() > 0.0001 {
        cgmath::Vector3::new(
            light_clip_pos.x / light_clip_pos.w,
            light_clip_pos.y / light_clip_pos.w,
            light_clip_pos.z / light_clip_pos.w,
        )
    } else {
        cgmath::Vector3::new(0.0, 0.0, 0.0)
    };
    log!(
        "Billboard NDC: ({:.4}, {:.4}, {:.4})",
        light_ndc.x,
        light_ndc.y,
        light_ndc.z
    );

    let screen_x = (light_ndc.x * 0.5 + 0.5) * swapchain_extent.width as f32;
    let screen_y = (light_ndc.y * 0.5 + 0.5) * swapchain_extent.height as f32;
    log!(
        "Billboard screen position: ({:.1}, {:.1})",
        screen_x,
        screen_y
    );

    let screen_uv_x = screen_x / swapchain_extent.width as f32;
    let screen_uv_y = screen_y / swapchain_extent.height as f32;
    log!(
        "Screen UV at billboard center: ({:.4}, {:.4})",
        screen_uv_x,
        screen_uv_y
    );

    log!("Near plane: {:.4}", info.near_plane);
}
