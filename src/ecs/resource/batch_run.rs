use std::path::PathBuf;

#[derive(Clone)]
pub enum BatchRunState {
    WaitingForFrame,
    ScreenshotRequested,
    Completed { result: Result<String, String> },
}

pub struct BatchRun {
    pub output: PathBuf,
    pub screenshot_frame: u64,
    pub frames_rendered: u64,
    pub state: BatchRunState,
    pub flame_set: Vec<(String, f32)>,
    pub dump_wall_probe: bool,
    pub captures_remaining: u32,
    pub stride: u32,
    pub sequence_dir: Option<PathBuf>,
    pub total_count: u32,
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
            captures_remaining: 0,
            stride: 1,
            sequence_dir: None,
            total_count: 0,
        }
    }

    pub fn is_completed(&self) -> bool {
        matches!(self.state, BatchRunState::Completed { .. })
    }
}
