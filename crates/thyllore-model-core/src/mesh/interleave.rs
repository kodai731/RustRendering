use super::{MeshData, VertexAttribute, VertexAttributeValues, VertexFormat};

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

pub fn compute_vertex_layout(mesh: &MeshData) -> VertexLayout {
    let mut attributes = Vec::new();
    let mut offset = 0u32;

    for id in mesh.attribute_ids() {
        if let Some(values) = mesh.attribute(*id) {
            let format = values.format();
            let attr = VertexAttribute::new(*id, format, id.default_location());
            attributes.push(attr);
            offset += format.size();
        }
    }

    VertexLayout {
        attributes,
        stride: offset,
    }
}

pub fn create_interleaved_buffer(mesh: &MeshData, layout: &VertexLayout) -> Vec<u8> {
    let vertex_count = mesh.vertex_count();
    if vertex_count == 0 {
        return Vec::new();
    }

    let buffer_size = vertex_count * layout.stride as usize;
    let mut buffer = vec![0u8; buffer_size];

    for vertex_idx in 0..vertex_count {
        let mut offset = 0usize;
        for attr in &layout.attributes {
            if let Some(values) = mesh.attribute(attr.id) {
                let dst_offset = vertex_idx * layout.stride as usize + offset;
                write_attribute_value(&mut buffer, dst_offset, values, vertex_idx, attr.format);
            }
            offset += attr.format.size() as usize;
        }
    }

    buffer
}

fn write_attribute_value(
    buffer: &mut [u8],
    offset: usize,
    values: &VertexAttributeValues,
    index: usize,
    format: VertexFormat,
) {
    match (values, format) {
        (VertexAttributeValues::Float32x2(data), VertexFormat::Float32x2) => {
            write_f32_array(&mut buffer[offset..], &data[index]);
        }
        (VertexAttributeValues::Float32x3(data), VertexFormat::Float32x3) => {
            write_f32_array(&mut buffer[offset..], &data[index]);
        }
        (VertexAttributeValues::Float32x4(data), VertexFormat::Float32x4) => {
            write_f32_array(&mut buffer[offset..], &data[index]);
        }
        _ => unreachable!(
            "Format mismatch in write_attribute_value: values={:?}, format={:?}",
            std::mem::discriminant(values),
            format
        ),
    }
}

fn write_f32_array<const N: usize>(buffer: &mut [u8], values: &[f32; N]) {
    for (i, &value) in values.iter().enumerate() {
        let bytes = value.to_le_bytes();
        let start = i * 4;
        buffer[start..start + 4].copy_from_slice(&bytes);
    }
}
