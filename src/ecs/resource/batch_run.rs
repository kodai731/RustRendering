use std::path::PathBuf;

use crate::animation::editable::SourceClipId;

#[derive(Clone)]
pub enum BatchRunState {
    WaitingForFrame,
    ScreenshotRequested,
    Completed { result: Result<String, String> },
}

#[derive(Clone)]
pub struct BatchRun {
    pub output: PathBuf,
    pub screenshot_frame: u64,
    pub frames_rendered: u64,
    pub state: BatchRunState,
    pub flame_set: Vec<(String, f32)>,
    pub dump_wall_probe: bool,
    pub dump_water_debug: bool,
    pub captures_remaining: u32,
    pub stride: u32,
    pub sequence_dir: Option<PathBuf>,
    pub total_count: u32,
    pub flame_trace_path: Option<PathBuf>,
    pub wall_probe_path: Option<PathBuf>,
    pub water_probe_path: Option<PathBuf>,
    pub play_requested: bool,
    pub play_clip_id: Option<SourceClipId>,
}

impl BatchRun {
    pub fn new(output: PathBuf, screenshot_frame: u64, flame_set: Vec<(String, f32)>) -> Self {
        Self {
            output,
            screenshot_frame,
            frames_rendered: 0,
            state: BatchRunState::WaitingForFrame,
            flame_set,
            dump_wall_probe: false,
            dump_water_debug: false,
            captures_remaining: 0,
            stride: 1,
            sequence_dir: None,
            total_count: 0,
            flame_trace_path: None,
            wall_probe_path: None,
            water_probe_path: None,
            play_requested: false,
            play_clip_id: None,
        }
    }

    pub fn is_completed(&self) -> bool {
        matches!(self.state, BatchRunState::Completed { .. })
    }
}
