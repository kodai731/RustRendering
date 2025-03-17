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
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffer_memories: Vec<vk::DeviceMemory>,
    pub uniform_buffer_objects: Vec<UniformBufferObject>,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
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
    const fn new(pos: Vec3, color: Vec4, tex_coord: Vec2) -> Self {
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
