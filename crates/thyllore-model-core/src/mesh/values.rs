use super::VertexFormat;

#[derive(Clone, Debug)]
pub enum VertexAttributeValues {
    Float32x2(Vec<[f32; 2]>),
    Float32x3(Vec<[f32; 3]>),
    Float32x4(Vec<[f32; 4]>),
}

impl VertexAttributeValues {
    pub fn len(&self) -> usize {
        match self {
            VertexAttributeValues::Float32x2(v) => v.len(),
            VertexAttributeValues::Float32x3(v) => v.len(),
            VertexAttributeValues::Float32x4(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn format(&self) -> VertexFormat {
        match self {
            VertexAttributeValues::Float32x2(_) => VertexFormat::Float32x2,
            VertexAttributeValues::Float32x3(_) => VertexFormat::Float32x3,
            VertexAttributeValues::Float32x4(_) => VertexFormat::Float32x4,
        }
    }

    pub fn get_float32x2(&self, index: usize) -> Option<[f32; 2]> {
        match self {
            VertexAttributeValues::Float32x2(v) => v.get(index).copied(),
            _ => None,
        }
    }

    pub fn get_float32x3(&self, index: usize) -> Option<[f32; 3]> {
        match self {
            VertexAttributeValues::Float32x3(v) => v.get(index).copied(),
            _ => None,
        }
    }

    pub fn get_float32x4(&self, index: usize) -> Option<[f32; 4]> {
        match self {
            VertexAttributeValues::Float32x4(v) => v.get(index).copied(),
            _ => None,
        }
    }
}

impl From<Vec<[f32; 2]>> for VertexAttributeValues {
    fn from(v: Vec<[f32; 2]>) -> Self {
        VertexAttributeValues::Float32x2(v)
    }
}

impl From<Vec<[f32; 3]>> for VertexAttributeValues {
    fn from(v: Vec<[f32; 3]>) -> Self {
        VertexAttributeValues::Float32x3(v)
    }
}

impl From<Vec<[f32; 4]>> for VertexAttributeValues {
    fn from(v: Vec<[f32; 4]>) -> Self {
        VertexAttributeValues::Float32x4(v)
    }
}
