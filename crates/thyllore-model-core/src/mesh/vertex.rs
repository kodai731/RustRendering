use std::hash::{Hash, Hasher};

use thyllore_math_core::{Vec2, Vec3, Vec4};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {
    pub pos: Vec3,
    pub color: Vec4,
    pub tex_coord: Vec2,
    pub normal: Vec3,
}

impl Vertex {
    pub fn new(pos: Vec3, color: Vec4, tex_coord: Vec2) -> Self {
        Self {
            pos,
            color,
            tex_coord,
            normal: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    pub fn new_with_normal(pos: Vec3, color: Vec4, tex_coord: Vec2, normal: Vec3) -> Self {
        Self {
            pos,
            color,
            tex_coord,
            normal,
        }
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
            && self.color == other.color
            && self.tex_coord == other.tex_coord
            && self.normal == other.normal
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
        self.normal[0].to_bits().hash(state);
        self.normal[1].to_bits().hash(state);
        self.normal[2].to_bits().hash(state);
    }
}

#[derive(Clone, Debug, Default)]
pub struct VertexData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}
