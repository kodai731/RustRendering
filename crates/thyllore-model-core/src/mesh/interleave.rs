use super::VertexAttribute;

#[derive(Clone, Debug)]
pub struct VertexLayout {
    pub attributes: Vec<VertexAttribute>,
    pub stride: u32,
}

impl VertexLayout {
    pub fn from_attributes(attrs: &[VertexAttribute]) -> Self {
        let stride: u32 = attrs.iter().map(|a| a.format.size()).sum();
        Self {
            attributes: attrs.to_vec(),
            stride,
        }
    }
}
