use std::ffi::CStr;

use crate::app::App;
use crate::ecs::systems::{
    build_water_debug_record, current_unix_time, water_debug_screenshot_path,
    write_water_debug_dump, WaterDebugRenderInfo,
};
use crate::vulkanr::context::SwapchainState;
use crate::vulkanr::vulkan::*;

impl App {
    pub fn dump_water_debug(&self) {
        let unix_time = current_unix_time();
        let mut render_info = self.collect_water_debug_render_info();

        let screenshot_path = water_debug_screenshot_path(unix_time);
        let image_index = self.frame % crate::app::init::MAX_FRAMES_IN_FLIGHT;
        match unsafe { self.save_screenshot_to(image_index, &screenshot_path) } {
            Ok(_) => render_info.screenshot_path = Some(screenshot_path.display().to_string()),
            Err(error) => log_warn!("water debug screenshot failed: {:?}", error),
        }

        let record = build_water_debug_record(&self.data.ecs_world, &render_info, unix_time);
        match write_water_debug_dump(&record, unix_time) {
            Ok(path) => msg_info!("Water debug dumped: {}", path.display()),
            Err(error) => log_error!("water debug dump failed: {}", error),
        }
    }

    fn collect_water_debug_render_info(&self) -> WaterDebugRenderInfo {
        let properties = unsafe {
            self.instance
                .get_physical_device_properties(self.rrdevice.physical_device)
        };
        let gpu_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .to_string();

        let swapchain_extent = self.resource::<SwapchainState>().swapchain.swapchain_extent;
        let viewport = &self.data.viewport;
        let acceleration = self.data.raytracing.acceleration_structure.as_ref();

        WaterDebugRenderInfo {
            gpu_name,
            driver_version: format_vulkan_version(properties.driver_version),
            api_version: format_vulkan_version(properties.api_version),
            swapchain_size: [swapchain_extent.width, swapchain_extent.height],
            hdr_buffer_size: viewport.hdr_buffer.as_ref().map(|b| [b.width, b.height]),
            water_buffer_size: viewport.water_buffer.as_ref().map(|b| [b.width, b.height]),
            mesh_count: self.data.graphics_resources.meshes.len(),
            mesh_blas_count: acceleration.map(|a| a.blas_list.len()).unwrap_or(0),
            water_blas_count: acceleration.map(|a| a.water_blas.len()).unwrap_or(0),
            hit_shading_table_capacity: acceleration
                .and_then(|a| a.hit_shading_table.as_ref())
                .map(|table| table.capacity),
            screenshot_path: None,
        }
    }
}

fn format_vulkan_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::version_major(version),
        vk::version_minor(version),
        vk::version_patch(version)
    )
}
