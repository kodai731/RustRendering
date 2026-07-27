use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuFrameTimings {
    pub frame: u64,
    pub dt_ms: f32,
    pub stages: Vec<(String, f32)>,
    pub imgui_vtx: u32,
    pub imgui_idx: u32,
}
