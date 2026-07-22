pub struct GpuTimingsSink {
    pub path: String,
    pub last_frame: u64,
}

impl GpuTimingsSink {
    pub fn new(path: String) -> Self {
        Self { path, last_frame: 0 }
    }
}
