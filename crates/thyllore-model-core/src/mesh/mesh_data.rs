use std::collections::BTreeMap;

use super::{VertexAttribute, VertexAttributeId, VertexAttributeValues};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrimitiveTopology {
    #[default]
    TriangleList,
    LineList,
}

#[derive(Clone, Debug, Default)]
pub struct MeshData {
    attributes: BTreeMap<VertexAttributeId, VertexAttributeValues>,
    indices: Option<Vec<u32>>,
    topology: PrimitiveTopology,
}

impl MeshData {
    pub fn new(topology: PrimitiveTopology) -> Self {
        Self {
            attributes: BTreeMap::new(),
            indices: None,
            topology,
        }
    }

    pub fn insert_attribute(
        &mut self,
        attr: VertexAttribute,
        values: impl Into<VertexAttributeValues>,
    ) {
        let values = values.into();
        assert_eq!(
            attr.format,
            values.format(),
            "Attribute format mismatch: expected {:?}, got {:?}",
            attr.format,
            values.format()
        );
        self.attributes.insert(attr.id, values);
    }

    pub fn with_inserted_attribute(
        mut self,
        attr: VertexAttribute,
        values: impl Into<VertexAttributeValues>,
    ) -> Self {
        self.insert_attribute(attr, values);
        self
    }

    pub fn set_indices(&mut self, indices: Vec<u32>) {
        self.indices = Some(indices);
    }

    pub fn with_indices(mut self, indices: Vec<u32>) -> Self {
        self.set_indices(indices);
        self
    }

    pub fn attribute(&self, id: VertexAttributeId) -> Option<&VertexAttributeValues> {
        self.attributes.get(&id)
    }

    pub fn attribute_ids(&self) -> impl Iterator<Item = &VertexAttributeId> {
        self.attributes.keys()
    }

    pub fn indices(&self) -> Option<&[u32]> {
        self.indices.as_deref()
    }

    pub fn topology(&self) -> PrimitiveTopology {
        self.topology
    }

    pub fn vertex_count(&self) -> usize {
        self.attributes
            .values()
            .next()
            .map(|v| v.len())
            .unwrap_or(0)
    }

    pub fn index_count(&self) -> usize {
        self.indices.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    pub fn has_attribute(&self, id: VertexAttributeId) -> bool {
        self.attributes.contains_key(&id)
    }
}
