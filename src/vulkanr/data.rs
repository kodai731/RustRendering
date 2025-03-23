use super::buffer::*;
use super::device::*;
use super::swapchain::*;
use super::vulkan::*;
use std::cmp::PartialEq;
use std::hash::{Hash, Hasher};
use AqooleEngineR::math::math::{Mat4, Vec2, Vec3, Vec4};

#[repr(C)] // for compatibility of C struct
#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {
    pub pos: Vec3,
    pub color: Vec4,
    pub tex_coord: Vec2,
}

#[derive(Clone, Debug, Default)]
pub struct RRData {
    pub rruniform_buffers: Vec<RRUniformBuffer>,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
}

impl RRData {
    pub unsafe fn create_uniform_buffers(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
    ) -> Self {
        let mut rrdata = RRData::default();
        for _ in 0..rrswapchain.swapchain_images.len() {
            let ubo = UniformBufferObject::default();
            let rruniform_buffer = RRUniformBuffer::new(instance, rrdevice, ubo);
            rrdata.rruniform_buffers.push(rruniform_buffer);
        }
        rrdata
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

impl Default for UniformBufferObject {
    fn default() -> Self {
        let identity = Mat4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        Self {
            model: identity,
            view: identity,
            proj: identity,
        }
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.color && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}
impl Vertex {
    pub const fn new(pos: Vec3, color: Vec4, tex_coord: Vec2) -> Self {
        Self {
            pos,
            color,
            tex_coord,
        }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        //  at which rate to load data from memory throughout the vertices
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        // how to extract a vertex attribute from a chunk of vertex data originating from a binding description
        // two attributes, position and color
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0) // directive of the input in the vertex shader
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

        [pos, color, tex_coord]
    }
}
