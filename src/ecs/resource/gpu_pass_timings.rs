use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPassTimings {
    pub frame: u64,
    pub passes: Vec<(String, f32)>,
}

impl Default for GpuPassTimings {
    fn default() -> Self {
        Self {
            frame: 0,
            passes: Vec::new(),
        }
    }
}
