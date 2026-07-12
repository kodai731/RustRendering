use anyhow::Result;
use std::ffi::c_void;
use vulkanalia::prelude::v1_0::*;

use crate::core::RRDevice;
use crate::data::Vertex;
use crate::resource::create_buffer;

#[derive(Debug)]
pub struct OnionSkinGhostBuffer {
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub vertex_count: u32,
    pub tint_color: [f32; 3],
    pub opacity: f32,
    capacity_bytes: u64,
}

impl OnionSkinGhostBuffer {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        vertex_capacity: usize,
    ) -> Result<Self> {
        let capacity_bytes = (vertex_capacity * std::mem::size_of::<Vertex>()) as u64;

        let (buffer, memory) = create_buffer(
            instance,
            rrdevice,
            capacity_bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        Ok(Self {
            vertex_buffer: buffer,
            vertex_buffer_memory: memory,
            vertex_count: 0,
            tint_color: [1.0, 1.0, 1.0],
            opacity: 0.3,
            capacity_bytes,
        })
    }

    pub unsafe fn update_vertices(
        &mut self,
        rrdevice: &RRDevice,
        vertices: &[Vertex],
        tint_color: [f32; 3],
        opacity: f32,
    ) -> Result<()> {
        let data_size = (vertices.len() * std::mem::size_of::<Vertex>()) as u64;
        if data_size > self.capacity_bytes {
            return Err(anyhow::anyhow!(
                "Ghost buffer capacity exceeded: {} > {}",
                data_size,
                self.capacity_bytes
            ));
        }

        let memory = rrdevice.device.map_memory(
            self.vertex_buffer_memory,
            0,
            data_size,
            vk::MemoryMapFlags::empty(),
        )?;

        std::ptr::copy_nonoverlapping(
            vertices.as_ptr() as *const c_void,
            memory,
            data_size as usize,
        );

        rrdevice.device.unmap_memory(self.vertex_buffer_memory);

        self.vertex_count = vertices.len() as u32;
        self.tint_color = tint_color;
        self.opacity = opacity;

        Ok(())
    }

    pub unsafe fn destroy(&self, rrdevice: &RRDevice) {
        rrdevice.device.destroy_buffer(self.vertex_buffer, None);
        rrdevice.device.free_memory(self.vertex_buffer_memory, None);
    }
}

#[derive(Debug, Default)]
pub struct OnionSkinGpuState {
    pub ghost_buffers: Vec<OnionSkinGhostBuffer>,
    pub source_index_buffer: vk::Buffer,
    pub source_index_count: u32,
    pub source_mesh_index: Option<usize>,
}

impl OnionSkinGpuState {
    pub unsafe fn ensure_capacity(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        ghost_count: usize,
        vertex_capacity: usize,
    ) -> Result<()> {
        while self.ghost_buffers.len() < ghost_count {
            let buffer = OnionSkinGhostBuffer::new(instance, rrdevice, vertex_capacity)?;
            self.ghost_buffers.push(buffer);
        }

        while self.ghost_buffers.len() > ghost_count {
            if let Some(buffer) = self.ghost_buffers.pop() {
                buffer.destroy(rrdevice);
            }
        }

        Ok(())
    }

    pub unsafe fn destroy(&mut self, rrdevice: &RRDevice) {
        for buffer in self.ghost_buffers.drain(..) {
            buffer.destroy(rrdevice);
        }
    }

    pub fn active_ghost_count(&self) -> usize {
        self.ghost_buffers
            .iter()
            .filter(|b| b.vertex_count > 0)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_state_default() {
        let state = OnionSkinGpuState::default();
        assert!(state.ghost_buffers.is_empty());
        assert_eq!(state.source_index_count, 0);
        assert!(state.source_mesh_index.is_none());
    }

    #[test]
    fn test_active_ghost_count_empty() {
        let state = OnionSkinGpuState::default();
        assert_eq!(state.active_ghost_count(), 0);
    }
}
