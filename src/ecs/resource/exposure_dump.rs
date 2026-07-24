#[derive(Clone)]
pub struct ExposureDumpSink {
    pub path: String,
    pub last_frame: u64,
}

impl ExposureDumpSink {
    pub fn new(path: String) -> Self {
        Self { path, last_frame: 0 }
    }
}
