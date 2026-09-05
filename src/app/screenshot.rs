use crate::app::App;
use crate::vulkanr::context::{CommandState, SwapchainState};
use crate::vulkanr::vulkan::*;

use anyhow::Result;

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
        let command_pool = self.resource::<CommandState>().pool.command_pool;

        let (buffer, buffer_memory, command_buffer) = self.copy_image_to_buffer(
            swapchain_image,
            extent.width,
            extent.height,
            image_size,
            command_pool,
            vk::ImageLayout::PRESENT_SRC_KHR,
        )?;

        let saved_path =
            Self::encode_and_save_png(device, buffer_memory, image_size, width, height, path)?;

        device.free_command_buffers(command_pool, &[command_buffer]);
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
        command_pool: vk::CommandPool,
        layout: vk::ImageLayout,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, vk::CommandBuffer)> {
        let device = &self.rrdevice.device;

        let (buffer, buffer_memory) = self.allocate_transfer_buffer(device, image_size)?;

        let command_buffer = allocate_one_time_command_buffer(device, command_pool)?;

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(command_buffer, &begin_info)?;

        record_image_to_buffer_copy(
            device,
            command_buffer,
            image,
            buffer,
            width,
            height,
            layout,
            layout,
        );

        device.end_command_buffer(command_buffer)?;

        let command_buffers_slice = [command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers_slice);
        device.queue_submit(
            self.rrdevice.graphics_queue,
            &[submit_info.build()],
            vk::Fence::null(),
        )?;
        device.queue_wait_idle(self.rrdevice.graphics_queue)?;

        Ok((buffer, buffer_memory, command_buffer))
    }

    unsafe fn allocate_transfer_buffer(
        &self,
        device: &crate::vulkanr::core::device::Device,
        image_size: vk::DeviceSize,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(image_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device.create_buffer(&buffer_info, None)?;

        let mem_requirements = device.get_buffer_memory_requirements(buffer);
        let memory_type_index = self.get_memory_type_index(
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);
        let buffer_memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_buffer_memory(buffer, buffer_memory, 0)?;

        Ok((buffer, buffer_memory))
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

pub(crate) unsafe fn allocate_one_time_command_buffer(
    device: &crate::vulkanr::core::device::Device,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let cmd_alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffers = device.allocate_command_buffers(&cmd_alloc_info)?;
    Ok(command_buffers[0])
}

unsafe fn record_image_to_buffer_copy(
    device: &crate::vulkanr::core::device::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    buffer: vk::Buffer,
    width: u32,
    height: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    let barrier_to_transfer = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier_to_transfer.build()],
    );

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    device.cmd_copy_image_to_buffer(
        command_buffer,
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        buffer,
        &[region.build()],
    );

    let barrier_back = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ);

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier_back.build()],
    );
}

pub(crate) fn append_jsonl(path: &str, line: &str) {
    use std::fs::OpenOptions;
    use std::io::Write;
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(line.as_bytes());
    }
}
