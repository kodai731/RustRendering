use crate::app::App;
use crate::vulkanr::context::{CommandState, SwapchainState};
use crate::vulkanr::vulkan::*;

use anyhow::Result;
use thyllore_vulkan_core::copy_image_to_host_buffer;

impl App {
    pub unsafe fn save_screenshot(&self, image_index: usize) -> Result<String> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)?
            .as_secs();
        let path = std::path::PathBuf::from(format!("log/screenshot_{}.png", timestamp));
        self.save_screenshot_to(image_index, &path)
    }

    pub unsafe fn save_screenshot_to(
        &self,
        image_index: usize,
        path: &std::path::Path,
    ) -> Result<String> {
        let device = &self.rrdevice.device;
        let swapchain = &self.resource::<SwapchainState>().swapchain;
        let swapchain_image = swapchain.swapchain_images[image_index];
        let extent = swapchain.swapchain_extent;
        let width = extent.width;
        let height = extent.height;
        let image_size = (width * height * 4) as vk::DeviceSize;

        let (buffer, buffer_memory) = self.copy_image_to_buffer(
            swapchain_image,
            extent.width,
            extent.height,
            image_size,
            vk::ImageLayout::PRESENT_SRC_KHR,
        )?;

        let saved_path =
            Self::encode_and_save_png(device, buffer_memory, image_size, width, height, path)?;

        device.free_memory(buffer_memory, None);
        device.destroy_buffer(buffer, None);

        Ok(saved_path)
    }

    pub unsafe fn copy_image_to_buffer(
        &self,
        image: vk::Image,
        width: u32,
        height: u32,
        image_size: vk::DeviceSize,
        layout: vk::ImageLayout,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let command_pool = self.resource::<CommandState>().pool.command_pool;
        copy_image_to_host_buffer(
            &self.instance,
            &self.rrdevice,
            self.rrdevice.graphics_queue,
            command_pool,
            image,
            width,
            height,
            image_size,
            layout,
        )
    }

    unsafe fn encode_and_save_png(
        device: &crate::vulkanr::core::device::Device,
        buffer_memory: vk::DeviceMemory,
        image_size: vk::DeviceSize,
        width: u32,
        height: u32,
        path: &std::path::Path,
    ) -> Result<String> {
        use std::fs::File;
        use std::io::BufWriter;

        let data = device.map_memory(buffer_memory, 0, image_size, vk::MemoryMapFlags::empty())?;
        let slice = std::slice::from_raw_parts(data as *const u8, image_size as usize);

        let mut rgba_data = vec![0u8; (width * height * 4) as usize];
        for i in (0..rgba_data.len()).step_by(4) {
            rgba_data[i] = slice[i + 2];
            rgba_data[i + 1] = slice[i + 1];
            rgba_data[i + 2] = slice[i];
            rgba_data[i + 3] = slice[i + 3];
        }

        device.unmap_memory(buffer_memory);

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let mut encoder = png::Encoder::new(writer, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut png_writer = encoder.write_header()?;
        png_writer.write_image_data(&rgba_data)?;

        let absolute_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let path_str = absolute_path.to_string_lossy().to_string();

        log!("Screenshot saved to: {}", path_str);

        Ok(path_str)
    }
}
