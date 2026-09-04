use vulkanalia::vk;

#[derive(Clone, Debug)]
pub enum BlasGeometry<'a> {
    Triangles {
        vertex_buffer: &'a vk::Buffer,
        vertex_count: u32,
        vertex_stride: u32,
        index_buffer: &'a vk::Buffer,
        index_count: u32,
    },
    ProceduralAabb {
        aabb: vk::AabbPositionsKHR,
    },
}

#[derive(Clone, Debug)]
pub struct GpuPrimitive<'a> {
    pub geometry: BlasGeometry<'a>,
    pub model: cgmath::Matrix4<f32>,
    pub base_color: [f32; 4],
    pub params: [f32; 4],
}
