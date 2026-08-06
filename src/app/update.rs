use crate::app::{App, AppData, FrameContext};
use crate::ecs::run_frame;
use crate::vulkanr::device::RRDevice;
use crate::vulkanr::vulkan::*;

use anyhow::Result;

impl App {
    pub unsafe fn update(&mut self, image_index: usize) -> Result<()> {
        let (time, delta_time) = if self.get_resource::<crate::ecs::resource::BatchRun>().is_some()
        {
            let time = self.last_update_time + 1.0 / 60.0;
            let delta_time = 1.0 / 60.0;
            (time, delta_time)
        } else {
            let time = self.start.elapsed().as_secs_f32();
            let delta_time = time - self.last_update_time;
            (time, delta_time)
        };
        self.last_update_time = time;

        let viewport_extent = (
            self.data.viewport.width.max(1),
            self.data.viewport.height.max(1),
        );
        let command_pool = self
            .data
            .ecs_world
            .resource::<crate::vulkanr::context::CommandState>()
            .pool
            .clone();

        {
            let mut ctx = FrameContext {
                instance: &self.instance,
                device: &self.rrdevice,
                command_pool,
                time,
                delta_time,
                image_index,
                frame_slot: self.frame % crate::app::init::MAX_FRAMES_IN_FLIGHT,
                swapchain_extent: viewport_extent,
                graphics: &mut self.data.graphics_resources,
                raytracing: &mut self.data.raytracing,
                buffer_registry: &mut self.data.buffer_registry,
                world: &mut self.data.ecs_world,
                assets: &mut self.data.ecs_assets,
                onion_skin_gpu: &mut self.data.onion_skin_gpu,
            };

            run_frame(&mut ctx)?;
        }

        Ok(())
    }

    pub unsafe fn update_imgui_buffers(
        instance: &Instance,
        rrdevice: &RRDevice,
        data: &mut AppData,
        draw_data: &imgui::DrawData,
        frame_slot: usize,
    ) -> Result<()> {
        if draw_data.total_vtx_count == 0 || draw_data.total_idx_count == 0 {
            return Ok(());
        }

        let vtx_buffer_size = (draw_data.total_vtx_count as usize
            * std::mem::size_of::<imgui::DrawVert>())
            as vk::DeviceSize;
        let idx_buffer_size = (draw_data.total_idx_count as usize
            * std::mem::size_of::<imgui::DrawIdx>())
            as vk::DeviceSize;

        let needs_vertex_resize = data.imgui.vertex_buffers[frame_slot].is_none()
            || vtx_buffer_size > data.imgui.vertex_buffer_sizes[frame_slot];
        let needs_index_resize = data.imgui.index_buffers[frame_slot].is_none()
            || idx_buffer_size > data.imgui.index_buffer_sizes[frame_slot];

        if needs_vertex_resize || needs_index_resize {
            rrdevice.device.device_wait_idle()?;
        }

        if needs_vertex_resize {
            resize_imgui_vertex_buffer(instance, rrdevice, data, vtx_buffer_size, frame_slot)?;
        }

        if needs_index_resize {
            resize_imgui_index_buffer(instance, rrdevice, data, idx_buffer_size, frame_slot)?;
        }

        upload_imgui_vertex_data(rrdevice, data, draw_data, vtx_buffer_size, frame_slot)?;
        upload_imgui_index_data(rrdevice, data, draw_data, idx_buffer_size, frame_slot)?;

        Ok(())
    }
}

unsafe fn resize_imgui_vertex_buffer(
    instance: &Instance,
    rrdevice: &RRDevice,
    data: &mut AppData,
    vtx_buffer_size: vk::DeviceSize,
    frame_slot: usize,
) -> Result<()> {
    if let Some(buffer) = data.imgui.vertex_buffers[frame_slot] {
        rrdevice.device.destroy_buffer(buffer, None);
    }
    if let Some(memory) = data.imgui.vertex_buffer_memories[frame_slot] {
        rrdevice.device.free_memory(memory, None);
    }

    let buffer_info = vk::BufferCreateInfo::builder()
        .size(vtx_buffer_size)
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let vertex_buffer = rrdevice.device.create_buffer(&buffer_info, None)?;
    let mem_requirements = rrdevice
        .device
        .get_buffer_memory_requirements(vertex_buffer);

    let mem_alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(crate::vulkanr::vulkan::get_memory_type_index(
            instance,
            rrdevice.physical_device,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            mem_requirements,
        )?);

    let vertex_buffer_memory = rrdevice.device.allocate_memory(&mem_alloc_info, None)?;
    rrdevice
        .device
        .bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0)?;

    data.imgui.vertex_buffers[frame_slot] = Some(vertex_buffer);
    data.imgui.vertex_buffer_memories[frame_slot] = Some(vertex_buffer_memory);
    data.imgui.vertex_buffer_sizes[frame_slot] = vtx_buffer_size;

    Ok(())
}

unsafe fn resize_imgui_index_buffer(
    instance: &Instance,
    rrdevice: &RRDevice,
    data: &mut AppData,
    idx_buffer_size: vk::DeviceSize,
    frame_slot: usize,
) -> Result<()> {
    if let Some(buffer) = data.imgui.index_buffers[frame_slot] {
        rrdevice.device.destroy_buffer(buffer, None);
    }
    if let Some(memory) = data.imgui.index_buffer_memories[frame_slot] {
        rrdevice.device.free_memory(memory, None);
    }

    let buffer_info = vk::BufferCreateInfo::builder()
        .size(idx_buffer_size)
        .usage(vk::BufferUsageFlags::INDEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let index_buffer = rrdevice.device.create_buffer(&buffer_info, None)?;
    let mem_requirements = rrdevice.device.get_buffer_memory_requirements(index_buffer);

    let mem_alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(crate::vulkanr::vulkan::get_memory_type_index(
            instance,
            rrdevice.physical_device,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            mem_requirements,
        )?);

    let index_buffer_memory = rrdevice.device.allocate_memory(&mem_alloc_info, None)?;
    rrdevice
        .device
        .bind_buffer_memory(index_buffer, index_buffer_memory, 0)?;

    data.imgui.index_buffers[frame_slot] = Some(index_buffer);
    data.imgui.index_buffer_memories[frame_slot] = Some(index_buffer_memory);
    data.imgui.index_buffer_sizes[frame_slot] = idx_buffer_size;

    Ok(())
}

unsafe fn upload_imgui_vertex_data(
    rrdevice: &RRDevice,
    data: &AppData,
    draw_data: &imgui::DrawData,
    vtx_buffer_size: vk::DeviceSize,
    frame_slot: usize,
) -> Result<()> {
    let Some(vertex_buffer_memory) = data.imgui.vertex_buffer_memories[frame_slot] else {
        return Ok(());
    };

    let ptr = rrdevice.device.map_memory(
        vertex_buffer_memory,
        0,
        vtx_buffer_size,
        vk::MemoryMapFlags::empty(),
    )?;

    let mut offset = 0;
    for draw_list in draw_data.draw_lists() {
        let vtx_buffer = draw_list.vtx_buffer();
        let vtx_size = (vtx_buffer.len() * std::mem::size_of::<imgui::DrawVert>()) as usize;
        std::ptr::copy_nonoverlapping(
            vtx_buffer.as_ptr() as *const u8,
            (ptr as *mut u8).add(offset),
            vtx_size,
        );
        offset += vtx_size;
    }

    rrdevice.device.unmap_memory(vertex_buffer_memory);
    Ok(())
}

unsafe fn upload_imgui_index_data(
    rrdevice: &RRDevice,
    data: &AppData,
    draw_data: &imgui::DrawData,
    idx_buffer_size: vk::DeviceSize,
    frame_slot: usize,
) -> Result<()> {
    let Some(index_buffer_memory) = data.imgui.index_buffer_memories[frame_slot] else {
        return Ok(());
    };

    let ptr = rrdevice.device.map_memory(
        index_buffer_memory,
        0,
        idx_buffer_size,
        vk::MemoryMapFlags::empty(),
    )?;

    let mut offset = 0;
    for draw_list in draw_data.draw_lists() {
        let idx_buffer = draw_list.idx_buffer();
        let idx_size = (idx_buffer.len() * std::mem::size_of::<imgui::DrawIdx>()) as usize;
        std::ptr::copy_nonoverlapping(
            idx_buffer.as_ptr() as *const u8,
            (ptr as *mut u8).add(offset),
            idx_size,
        );
        offset += idx_size;
    }

    rrdevice.device.unmap_memory(index_buffer_memory);
    Ok(())
}
