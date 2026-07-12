#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexFormat {
    Float32x2,
    Float32x3,
    Float32x4,
}

impl VertexFormat {
    pub fn size(&self) -> u32 {
        match self {
            VertexFormat::Float32x2 => 8,
            VertexFormat::Float32x3 => 12,
            VertexFormat::Float32x4 => 16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VertexAttributeId {
    Position,
    Normal,
    Tangent,
    Color,
    TexCoord0,
    TexCoord1,
}

impl VertexAttributeId {
    pub fn default_format(&self) -> VertexFormat {
        match self {
            VertexAttributeId::Position => VertexFormat::Float32x3,
            VertexAttributeId::Normal => VertexFormat::Float32x3,
            VertexAttributeId::Tangent => VertexFormat::Float32x4,
            VertexAttributeId::Color => VertexFormat::Float32x3,
            VertexAttributeId::TexCoord0 => VertexFormat::Float32x2,
            VertexAttributeId::TexCoord1 => VertexFormat::Float32x2,
        }
    }

    pub fn default_location(&self) -> u32 {
        match self {
            VertexAttributeId::Position => 0,
            VertexAttributeId::Normal => 1,
            VertexAttributeId::Tangent => 2,
            VertexAttributeId::Color => 3,
            VertexAttributeId::TexCoord0 => 4,
            VertexAttributeId::TexCoord1 => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    pub id: VertexAttributeId,
    pub format: VertexFormat,
    pub shader_location: u32,
}

impl VertexAttribute {
    pub const fn new(id: VertexAttributeId, format: VertexFormat, shader_location: u32) -> Self {
        Self {
            id,
            format,
            shader_location,
        }
    }

    pub fn from_id(id: VertexAttributeId) -> Self {
        Self {
            id,
            format: id.default_format(),
            shader_location: id.default_location(),
        }
    }
}
