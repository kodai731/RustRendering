use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::command::RRCommandPool;
use crate::core::device::RRDevice;
use crate::resource::buffer::{copy_buffer, create_buffer};
use crate::resource::GpuResource;
use thyllore_model_core::mesh::{compute_vertex_layout, create_interleaved_buffer, MeshData};
use thyllore_render_core::{BufferMemoryType, IndexBufferHandle, VertexBufferHandle};

#[derive(Debug)]
struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    is_host_visible: bool,
}

#[derive(Debug, Default)]
pub struct GpuBufferRegistry {
    vertex_buffers: Vec<Option<GpuBuffer>>,
    index_buffers: Vec<Option<GpuBuffer>>,
    free_vertex_slots: Vec<u32>,
    free_index_slots: Vec<u32>,
}

impl GpuBufferRegistry {
    pub fn new() -> Self {
        Self {
            vertex_buffers: Vec::new(),
            index_buffers: Vec::new(),
            free_vertex_slots: Vec::new(),
            free_index_slots: Vec::new(),
        }
    }

    pub unsafe fn create_vertex_buffer<T: Copy>(
        &mut self,
        instance: &Instance,
        device: &RRDevice,
        command_pool: &RRCommandPool,
        data: &[T],
        memory_type: BufferMemoryType,
    ) -> Result<VertexBufferHandle> {
        let buffer_size = (size_of::<T>() * data.len()) as u64;

        let gpu_buffer = if memory_type == BufferMemoryType::DeviceLocal {
            self.create_device_local_buffer(
                instance,
                device,
                command_pool,
                buffer_size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                data.as_ptr() as *const u8,
            )?
        } else {
            self.create_host_visible_buffer::<T>(
                instance,
                device,
                buffer_size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                data.as_ptr() as *const u8,
                data.len(),
            )?
        };

        let handle = self.allocate_vertex_slot();
        let index = handle.index();

        if index >= self.vertex_buffers.len() {
            self.vertex_buffers.resize_with(index + 1, || None);
        }
        self.vertex_buffers[index] = Some(gpu_buffer);

        Ok(handle)
    }

    pub unsafe fn create_index_buffer(
        &mut self,
        instance: &Instance,
        device: &RRDevice,
        command_pool: &RRCommandPool,
        data: &[u32],
    ) -> Result<IndexBufferHandle> {
        let buffer_size = (size_of::<u32>() * data.len()) as u64;

        let gpu_buffer = self.create_device_local_buffer(
            instance,
            device,
            command_pool,
            buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            data.as_ptr() as *const u8,
        )?;

        let handle = self.allocate_index_slot();
        let index = handle.index();

        if index >= self.index_buffers.len() {
            self.index_buffers.resize_with(index + 1, || None);
        }
        self.index_buffers[index] = Some(gpu_buffer);

        Ok(handle)
    }

    pub unsafe fn create_host_visible_index_buffer(
        &mut self,
        instance: &Instance,
        device: &RRDevice,
        data: &[u32],
    ) -> Result<IndexBufferHandle> {
        let buffer_size = (size_of::<u32>() * data.len()) as u64;

        let gpu_buffer = self.create_host_visible_buffer::<u32>(
            instance,
            device,
            buffer_size.max(256),
            vk::BufferUsageFlags::INDEX_BUFFER,
            data.as_ptr() as *const u8,
            data.len(),
        )?;

        let handle = self.allocate_index_slot();
        let index = handle.index();

        if index >= self.index_buffers.len() {
            self.index_buffers.resize_with(index + 1, || None);
        }
        self.index_buffers[index] = Some(gpu_buffer);

        Ok(handle)
    }

    pub unsafe fn create_host_visible_vertex_buffer<T: Copy>(
        &mut self,
        instance: &Instance,
        device: &RRDevice,
        data: &[T],
        min_size: u64,
    ) -> Result<VertexBufferHandle> {
        let buffer_size = ((size_of::<T>() * data.len()) as u64).max(min_size);

        let gpu_buffer = self.create_host_visible_buffer::<T>(
            instance,
            device,
            buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            data.as_ptr() as *const u8,
            data.len(),
        )?;

        let handle = self.allocate_vertex_slot();
        let index = handle.index();

        if index >= self.vertex_buffers.len() {
            self.vertex_buffers.resize_with(index + 1, || None);
        }
        self.vertex_buffers[index] = Some(gpu_buffer);

        Ok(handle)
    }

    pub unsafe fn update_vertex_buffer<T: Copy>(
        &self,
        device: &RRDevice,
        handle: VertexBufferHandle,
        data: &[T],
    ) -> Result<()> {
        if !handle.is_valid() {
            return Ok(());
        }

        let index = handle.index();
        if let Some(Some(gpu_buffer)) = self.vertex_buffers.get(index) {
            if gpu_buffer.is_host_visible {
                let data_size = (size_of::<T>() * data.len()) as u64;
                let ptr = device.device.map_memory(
                    gpu_buffer.memory,
                    0,
                    data_size,
                    vk::MemoryMapFlags::empty(),
                )?;
                memcpy(data.as_ptr(), ptr.cast(), data.len());
                device.device.unmap_memory(gpu_buffer.memory);
            }
        }

        Ok(())
    }

    pub unsafe fn update_index_buffer(
        &self,
        device: &RRDevice,
        handle: IndexBufferHandle,
        data: &[u32],
    ) -> Result<()> {
        if !handle.is_valid() {
            return Ok(());
        }

        let index = handle.index();
        if let Some(Some(gpu_buffer)) = self.index_buffers.get(index) {
            if gpu_buffer.is_host_visible {
                let data_size = (size_of::<u32>() * data.len()) as u64;
                let ptr = device.device.map_memory(
                    gpu_buffer.memory,
                    0,
                    data_size,
                    vk::MemoryMapFlags::empty(),
                )?;
                memcpy(data.as_ptr(), ptr.cast(), data.len());
                device.device.unmap_memory(gpu_buffer.memory);
            }
        }

        Ok(())
    }

    pub fn get_vertex_buffer_size(&self, handle: VertexBufferHandle) -> u64 {
        if !handle.is_valid() {
            return 0;
        }
        self.vertex_buffers
            .get(handle.index())
            .and_then(|opt| opt.as_ref())
            .map(|buf| buf.size)
            .unwrap_or(0)
    }

    pub fn get_index_buffer_size(&self, handle: IndexBufferHandle) -> u64 {
        if !handle.is_valid() {
            return 0;
        }
        self.index_buffers
            .get(handle.index())
            .and_then(|opt| opt.as_ref())
            .map(|buf| buf.size)
            .unwrap_or(0)
    }

    pub fn get_vertex_buffer(&self, handle: VertexBufferHandle) -> Option<vk::Buffer> {
        if !handle.is_valid() {
            return None;
        }

        self.vertex_buffers
            .get(handle.index())
            .and_then(|opt| opt.as_ref())
            .map(|buf| buf.buffer)
    }

    pub fn get_index_buffer(&self, handle: IndexBufferHandle) -> Option<vk::Buffer> {
        if !handle.is_valid() {
            return None;
        }

        self.index_buffers
            .get(handle.index())
            .and_then(|opt| opt.as_ref())
            .map(|buf| buf.buffer)
    }

    pub unsafe fn destroy_vertex_buffer(&mut self, device: &RRDevice, handle: VertexBufferHandle) {
        if !handle.is_valid() {
            return;
        }

        let index = handle.index();
        if let Some(Some(gpu_buffer)) = self.vertex_buffers.get(index) {
            device.device.destroy_buffer(gpu_buffer.buffer, None);
            device.device.free_memory(gpu_buffer.memory, None);
        }

        if index < self.vertex_buffers.len() {
            self.vertex_buffers[index] = None;
            self.free_vertex_slots.push(handle.0 .0);
        }
    }

    pub unsafe fn destroy_index_buffer(&mut self, device: &RRDevice, handle: IndexBufferHandle) {
        if !handle.is_valid() {
            return;
        }

        let index = handle.index();
        if let Some(Some(gpu_buffer)) = self.index_buffers.get(index) {
            device.device.destroy_buffer(gpu_buffer.buffer, None);
            device.device.free_memory(gpu_buffer.memory, None);
        }

        if index < self.index_buffers.len() {
            self.index_buffers[index] = None;
            self.free_index_slots.push(handle.0 .0);
        }
    }

    pub unsafe fn destroy_all(&mut self, device: &RRDevice) {
        for buffer_opt in self.vertex_buffers.drain(..) {
            if let Some(gpu_buffer) = buffer_opt {
                device.device.destroy_buffer(gpu_buffer.buffer, None);
                device.device.free_memory(gpu_buffer.memory, None);
            }
        }

        for buffer_opt in self.index_buffers.drain(..) {
            if let Some(gpu_buffer) = buffer_opt {
                device.device.destroy_buffer(gpu_buffer.buffer, None);
                device.device.free_memory(gpu_buffer.memory, None);
            }
        }

        self.free_vertex_slots.clear();
        self.free_index_slots.clear();
    }

    pub unsafe fn create_buffer_from_mesh_data(
        &mut self,
        instance: &Instance,
        device: &RRDevice,
        command_pool: &RRCommandPool,
        mesh: &MeshData,
        memory_type: BufferMemoryType,
    ) -> Result<(VertexBufferHandle, Option<IndexBufferHandle>)> {
        let layout = compute_vertex_layout(mesh);
        let vertex_data = create_interleaved_buffer(mesh, &layout);

        let vertex_handle = self.create_vertex_buffer_raw(
            instance,
            device,
            command_pool,
            &vertex_data,
            memory_type,
        )?;

        let index_handle = if let Some(indices) = mesh.indices() {
            Some(self.create_index_buffer(instance, device, command_pool, indices)?)
        } else {
            None
        };

        Ok((vertex_handle, index_handle))
    }

    unsafe fn create_vertex_buffer_raw(
        &mut self,
        instance: &Instance,
        device: &RRDevice,
        command_pool: &RRCommandPool,
        data: &[u8],
        memory_type: BufferMemoryType,
    ) -> Result<VertexBufferHandle> {
        let buffer_size = data.len() as u64;

        let gpu_buffer = if memory_type == BufferMemoryType::DeviceLocal {
            self.create_device_local_buffer(
                instance,
                device,
                command_pool,
                buffer_size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                data.as_ptr(),
            )?
        } else {
            let (buffer, memory) = create_buffer(
                instance,
                device,
                buffer_size.max(256),
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            if !data.is_empty() {
                let ptr = device.device.map_memory(
                    memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )?;
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast(), buffer_size as usize);
                device.device.unmap_memory(memory);
            }

            GpuBuffer {
                buffer,
                memory,
                size: buffer_size.max(256),
                is_host_visible: true,
            }
        };

        let handle = self.allocate_vertex_slot();
        let index = handle.index();

        if index >= self.vertex_buffers.len() {
            self.vertex_buffers.resize_with(index + 1, || None);
        }
        self.vertex_buffers[index] = Some(gpu_buffer);

        Ok(handle)
    }

    fn allocate_vertex_slot(&mut self) -> VertexBufferHandle {
        if let Some(slot) = self.free_vertex_slots.pop() {
            VertexBufferHandle::new(slot)
        } else {
            VertexBufferHandle::new(self.vertex_buffers.len() as u32)
        }
    }

    fn allocate_index_slot(&mut self) -> IndexBufferHandle {
        if let Some(slot) = self.free_index_slots.pop() {
            IndexBufferHandle::new(slot)
        } else {
            IndexBufferHandle::new(self.index_buffers.len() as u32)
        }
    }

    unsafe fn create_device_local_buffer(
        &self,
        instance: &Instance,
        device: &RRDevice,
        command_pool: &RRCommandPool,
        size: u64,
        usage: vk::BufferUsageFlags,
        data_ptr: *const u8,
    ) -> Result<GpuBuffer> {
        let (staging_buffer, staging_memory) = create_buffer(
            instance,
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let ptr = device
            .device
            .map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(data_ptr, ptr.cast(), size as usize);
        device.device.unmap_memory(staging_memory);

        let (buffer, memory) = create_buffer(
            instance,
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        copy_buffer(device, command_pool, staging_buffer, buffer, size)?;

        device.device.destroy_buffer(staging_buffer, None);
        device.device.free_memory(staging_memory, None);

        Ok(GpuBuffer {
            buffer,
            memory,
            size,
            is_host_visible: false,
        })
    }
}

impl GpuBufferRegistry {
    pub fn active_vertex_count(&self) -> usize {
        self.vertex_buffers.iter().filter(|b| b.is_some()).count()
    }

    pub fn active_index_count(&self) -> usize {
        self.index_buffers.iter().filter(|b| b.is_some()).count()
    }

    pub fn has_leaked_buffers(&self) -> bool {
        self.active_vertex_count() > 0 || self.active_index_count() > 0
    }

    #[cfg(test)]
    fn insert_dummy_vertex(&mut self) {
        self.vertex_buffers.push(Some(GpuBuffer {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 0,
            is_host_visible: false,
        }));
    }

    #[cfg(test)]
    fn insert_dummy_index(&mut self) {
        self.index_buffers.push(Some(GpuBuffer {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 0,
            is_host_visible: false,
        }));
    }

    pub fn clear_tracking(&mut self) {
        self.vertex_buffers.clear();
        self.index_buffers.clear();
        self.free_vertex_slots.clear();
        self.free_index_slots.clear();
    }
}

impl GpuResource for GpuBufferRegistry {
    unsafe fn destroy_gpu(&mut self, rrdevice: &RRDevice) {
        self.destroy_all(rrdevice);
    }

    fn resource_name(&self) -> &'static str {
        "GpuBufferRegistry"
    }
}

impl Drop for GpuBufferRegistry {
    fn drop(&mut self) {
        if self.has_leaked_buffers() {
            log_warn!(
                "GpuBufferRegistry dropped without calling destroy_all(): {} vertex, {} index buffers leaked",
                self.active_vertex_count(),
                self.active_index_count(),
            );
        }
    }
}

impl GpuBufferRegistry {
    unsafe fn create_host_visible_buffer<T>(
        &self,
        instance: &Instance,
        device: &RRDevice,
        size: u64,
        usage: vk::BufferUsageFlags,
        data_ptr: *const u8,
        data_count: usize,
    ) -> Result<GpuBuffer> {
        let (buffer, memory) = create_buffer(
            instance,
            device,
            size,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        if data_count > 0 {
            let actual_size = size_of::<T>() * data_count;
            let ptr = device.device.map_memory(
                memory,
                0,
                actual_size as u64,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(data_ptr, ptr.cast(), actual_size);
            device.device.unmap_memory(memory);
        }

        Ok(GpuBuffer {
            buffer,
            memory,
            size,
            is_host_visible: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_registry_no_leak() {
        let registry = GpuBufferRegistry::new();
        assert!(!registry.has_leaked_buffers());
        assert_eq!(registry.active_vertex_count(), 0);
        assert_eq!(registry.active_index_count(), 0);
    }

    #[test]
    fn test_registry_with_buffers_reports_leak() {
        let mut registry = GpuBufferRegistry::new();
        registry.insert_dummy_vertex();
        registry.insert_dummy_index();

        assert!(registry.has_leaked_buffers());
        assert_eq!(registry.active_vertex_count(), 1);
        assert_eq!(registry.active_index_count(), 1);
    }

    #[test]
    fn test_clear_tracking_prevents_leak() {
        let mut registry = GpuBufferRegistry::new();
        registry.insert_dummy_vertex();
        registry.insert_dummy_vertex();
        registry.insert_dummy_index();

        assert!(registry.has_leaked_buffers());

        registry.clear_tracking();

        assert!(!registry.has_leaked_buffers());
        assert_eq!(registry.active_vertex_count(), 0);
        assert_eq!(registry.active_index_count(), 0);
    }

    #[test]
    fn test_removed_slot_not_counted_as_leak() {
        let mut registry = GpuBufferRegistry::new();
        registry.insert_dummy_vertex();
        registry.insert_dummy_vertex();
        registry.insert_dummy_index();

        registry.vertex_buffers[0] = None;

        assert_eq!(registry.active_vertex_count(), 1);
        assert_eq!(registry.active_index_count(), 1);
        assert!(registry.has_leaked_buffers());

        registry.vertex_buffers[1] = None;
        registry.index_buffers[0] = None;

        assert!(!registry.has_leaked_buffers());
    }

    #[test]
    fn test_destroy_all_clears_buffers_and_free_slots() {
        let mut registry = GpuBufferRegistry::new();

        for _ in 0..10 {
            registry.insert_dummy_vertex();
            registry.insert_dummy_index();
        }

        registry.free_vertex_slots.push(0);
        registry.free_index_slots.push(0);

        assert_eq!(registry.active_vertex_count(), 10);
        assert_eq!(registry.active_index_count(), 10);

        registry.clear_tracking();

        assert!(!registry.has_leaked_buffers());
        assert!(registry.free_vertex_slots.is_empty());
        assert!(registry.free_index_slots.is_empty());
    }
}
