pub struct GltfModelCache {
    pub source_path: Option<String>,
    pub glb_data: Option<Vec<u8>>,
}

impl GltfModelCache {
    pub fn new(source_path: String) -> Self {
        Self {
            source_path: Some(source_path),
            glb_data: None,
        }
    }

    pub fn from_glb_data(glb_data: Vec<u8>) -> Self {
        Self {
            source_path: None,
            glb_data: Some(glb_data),
        }
    }

    pub fn empty() -> Self {
        Self {
            source_path: None,
            glb_data: None,
        }
    }

    pub fn has_model(&self) -> bool {
        self.source_path.is_some() || self.glb_data.is_some()
    }

    pub fn clear(&mut self) {
        self.source_path = None;
        self.glb_data = None;
    }
}
