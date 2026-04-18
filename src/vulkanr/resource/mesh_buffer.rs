use cgmath::Vector4;
use vulkanalia::prelude::v1_0::*;

use crate::animation::{SkeletonId, SkinData};
use crate::vulkanr::buffer::{RRIndexBuffer, RRVertexBuffer};
use crate::vulkanr::core::device::RRDevice;
use crate::vulkanr::data::{Vertex, VertexData};

#[derive(Clone, Debug)]
pub struct MeshBuffer {
    pub vertex_buffer: RRVertexBuffer,
    pub index_buffer: RRIndexBuffer,
    pub vertex_data: VertexData,
    pub image: vk::Image,
    pub image_memory: vk::DeviceMemory,
    pub mip_level: u32,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub render_to_gbuffer: bool,
    pub object_index: usize,
    pub skin_data: Option<SkinData>,
    pub skeleton_id: Option<SkeletonId>,
    pub node_index: Option<usize>,
    pub base_vertices: Vec<Vertex>,
    pub base_colors: Option<Vec<Vector4<f32>>>,
}

impl Default for MeshBuffer {
    fn default() -> Self {
        Self {
            vertex_buffer: RRVertexBuffer::default(),
            index_buffer: RRIndexBuffer::default(),
            vertex_data: VertexData::default(),
            image: vk::Image::null(),
            image_memory: vk::DeviceMemory::null(),
            mip_level: 0,
            image_view: vk::ImageView::null(),
            sampler: vk::Sampler::null(),
            render_to_gbuffer: true,
            object_index: 0,
            skin_data: None,
            skeleton_id: None,
            node_index: None,
            base_vertices: Vec::new(),
            base_colors: None,
        }
    }
}

impl MeshBuffer {
    pub unsafe fn destroy(&mut self, rrdevice: &RRDevice) {
        if self.image_view != vk::ImageView::null() {
            rrdevice.device.destroy_image_view(self.image_view, None);
            self.image_view = vk::ImageView::null();
        }
        if self.sampler != vk::Sampler::null() {
            rrdevice.device.destroy_sampler(self.sampler, None);
            self.sampler = vk::Sampler::null();
        }
        if self.vertex_buffer.buffer != vk::Buffer::null() {
            rrdevice
                .device
                .destroy_buffer(self.vertex_buffer.buffer, None);
            self.vertex_buffer.buffer = vk::Buffer::null();
        }
        if self.vertex_buffer.buffer_memory != vk::DeviceMemory::null() {
            rrdevice
                .device
                .free_memory(self.vertex_buffer.buffer_memory, None);
            self.vertex_buffer.buffer_memory = vk::DeviceMemory::null();
        }
        if self.index_buffer.buffer != vk::Buffer::null() {
            rrdevice
                .device
                .destroy_buffer(self.index_buffer.buffer, None);
            self.index_buffer.buffer = vk::Buffer::null();
        }
        if self.index_buffer.buffer_memory != vk::DeviceMemory::null() {
            rrdevice
                .device
                .free_memory(self.index_buffer.buffer_memory, None);
            self.index_buffer.buffer_memory = vk::DeviceMemory::null();
        }
        if self.image != vk::Image::null() {
            rrdevice.device.destroy_image(self.image, None);
            self.image = vk::Image::null();
        }
        if self.image_memory != vk::DeviceMemory::null() {
            rrdevice.device.free_memory(self.image_memory, None);
            self.image_memory = vk::DeviceMemory::null();
        }
    }
}

impl Drop for MeshBuffer {
    fn drop(&mut self) {
        if self.vertex_buffer.buffer != vk::Buffer::null() {
            log_warn!("MeshBuffer dropped without calling destroy()");
        }
    }
}
