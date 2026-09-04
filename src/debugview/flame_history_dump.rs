use crate::app::App;
use crate::vulkanr::context::CommandState;
use crate::vulkanr::vulkan::*;

use anyhow::Result;

impl App {
    pub unsafe fn save_flame_history_npy(&self, path: &std::path::Path) -> Result<()> {
        use thyllore_math_core::{f16_to_f32, write_npy_f32};

        let device = &self.rrdevice.device;
        let flame_buffer = self
            .data
            .viewport
            .flame_buffer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("flame buffer not initialized"))?;

        let flames = self.data.ecs_world.query_flames();
        let history_index = if let Some(first) = flames.first() {
            if let Some(temporal) = self
                .data
                .ecs_world
                .get_component::<crate::ecs::component::FlameTemporalAccum>(*first)
            {
                (temporal.frame_index as usize) & 1
            } else {
                0
            }
        } else {
            0
        };

        let history_image = flame_buffer.history_images[history_index];
        let width = flame_buffer.width;
        let height = flame_buffer.height;
        let image_size = (width * height * 8) as vk::DeviceSize;
        let command_pool = self.resource::<CommandState>().pool.command_pool;

        let (buffer, buffer_memory, command_buffer) = self.copy_image_to_buffer(
            history_image,
            width,
            height,
            image_size,
            command_pool,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;

        let data_ptr =
            device.map_memory(buffer_memory, 0, image_size, vk::MemoryMapFlags::empty())?;
        let slice = std::slice::from_raw_parts(data_ptr as *const u8, image_size as usize);

        let mut f32_data = Vec::with_capacity((width * height * 4) as usize);
        for chunk in slice.chunks_exact(8) {
            let r_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let g_bits = u16::from_le_bytes([chunk[2], chunk[3]]);
            let b_bits = u16::from_le_bytes([chunk[4], chunk[5]]);
            let a_bits = u16::from_le_bytes([chunk[6], chunk[7]]);
            f32_data.push(f16_to_f32(r_bits));
            f32_data.push(f16_to_f32(g_bits));
            f32_data.push(f16_to_f32(b_bits));
            f32_data.push(f16_to_f32(a_bits));
        }

        device.unmap_memory(buffer_memory);
        device.free_command_buffers(command_pool, &[command_buffer]);
        device.free_memory(buffer_memory, None);
        device.destroy_buffer(buffer, None);

        write_npy_f32(path, &[height as usize, width as usize, 4], &f32_data)?;

        Ok(())
    }
}
