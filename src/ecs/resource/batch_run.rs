use std::path::PathBuf;

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
}

impl BatchRun {
    pub fn new(output: PathBuf, screenshot_frame: u64) -> Self {
        Self {
            output,
            screenshot_frame,
            frames_rendered: 0,
            state: BatchRunState::WaitingForFrame,
        }
    }

    pub fn is_completed(&self) -> bool {
        matches!(self.state, BatchRunState::Completed { .. })
    }
}
