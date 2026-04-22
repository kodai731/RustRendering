#[derive(Clone, Debug)]
pub struct SnapSettings {
    pub snap_to_frame: bool,
    pub snap_to_key: bool,
    pub frame_rate: f32,
    pub snap_threshold_px: f32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            snap_to_frame: false,
            snap_to_key: false,
            frame_rate: 30.0,
            snap_threshold_px: 8.0,
        }
    }
}
