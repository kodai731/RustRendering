use crate::core::device::*;
use crate::core::swapchain::*;
use crate::resource::buffer::*;
use crate::vulkan::*;
use thyllore_math_core::*;
use thyllore_spirv_reflect::declare_gpu_block;

pub use thyllore_model_core::mesh::{Vertex, VertexData};

// TODO: implement iterator
#[derive(Clone, Debug)]
pub struct RRData {
    pub rruniform_buffers: Vec<RRUniformBuffer>,
    pub image: vk::Image,
    pub image_memory: vk::DeviceMemory,
    pub mip_level: u32,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub vertex_data: VertexData,
    pub vertex_buffer: RRVertexBuffer,
    pub index_buffer: RRIndexBuffer,
    pub render_to_gbuffer: bool,
}

impl Default for RRData {
    fn default() -> Self {
        Self {
            rruniform_buffers: Vec::new(),
            image: vk::Image::null(),
            image_memory: vk::DeviceMemory::null(),
            mip_level: 0,
            image_view: vk::ImageView::null(),
            sampler: vk::Sampler::null(),
            vertex_data: VertexData::default(),
            vertex_buffer: RRVertexBuffer::default(),
            index_buffer: RRIndexBuffer::default(),
            render_to_gbuffer: true,
        }
    }
}

impl RRData {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        name: &str,
    ) -> anyhow::Result<Self> {
        let mut rrdata = RRData::default();
        Self::create_uniform_buffers(&mut rrdata, instance, rrdevice, rrswapchain, name)?;
        Ok(rrdata)
    }

    pub unsafe fn create_uniform_buffers(
        rrdata: &mut RRData,
        instance: &Instance,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        name: &str,
    ) -> anyhow::Result<()> {
        for i in 0..rrswapchain.swapchain_images.len() {
            let ubo = UniformBufferObject::default();
            let buffer_name = format!("{}[{}]", name, i);
            let rruniform_buffer = RRUniformBuffer::new(instance, rrdevice, ubo, &buffer_name)?;
            rrdata.rruniform_buffers.push(rruniform_buffer);
        }
        Ok(())
    }

    pub unsafe fn delete_buffers(&mut self, rrdevice: &RRDevice) {
        for uniform_buffer in &mut self.rruniform_buffers {
            uniform_buffer.destroy(rrdevice);
        }
        self.rruniform_buffers.clear();

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
    }

    pub unsafe fn delete(&mut self, rrdevice: &RRDevice) {
        self.delete_buffers(rrdevice);

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

declare_gpu_block! {
    #[derive(Copy, Clone, Debug)]
    pub struct UniformBufferObject {
        pub model: Mat4,
        pub view: Mat4,
        pub proj: Mat4,
    }
}

impl Default for UniformBufferObject {
    fn default() -> Self {
        let identity = Mat4::identity();
        Self {
            model: identity,
            view: identity,
            proj: identity,
        }
    }
}

declare_gpu_block! {
    #[derive(Copy, Clone, Debug)]
    pub struct SceneUniformData {
        pub light_position: Vec4,
        pub light_color: Vec4,
        pub view: Mat4,
        pub proj: Mat4,
        pub debug_mode: i32,
        pub shadow_strength: f32,
        pub enable_distance_attenuation: i32,
        pub exposure_value: f32,
    }
}

impl Default for SceneUniformData {
    fn default() -> Self {
        let identity = Mat4::identity();
        Self {
            light_position: Vec4::new(5.0, 5.0, 5.0, 1.0),
            light_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            view: identity,
            proj: identity,
            debug_mode: 0,
            shadow_strength: 1.0,
            enable_distance_attenuation: 0,
            exposure_value: 1.0,
        }
    }
}

pub fn vertex_binding_description() -> vk::VertexInputBindingDescription {
    vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(size_of::<Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build()
}

pub fn vertex_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
    let pos = vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(0)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(0)
        .build();

    let color = vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(1)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .offset(size_of::<Vec3>() as u32)
        .build();

    let tex_coord = vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(2)
        .format(vk::Format::R32G32_SFLOAT)
        .offset((size_of::<Vec3>() + size_of::<Vec4>()) as u32)
        .build();

    let normal = vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(3)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset((size_of::<Vec3>() + size_of::<Vec4>() + size_of::<Vec2>()) as u32)
        .build();

    [pos, color, tex_coord, normal]
}
