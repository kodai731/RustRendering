/// A viewport click requested from the command line, so headless runs can exercise the picking
/// readback path. Fires once, on the first frame that reaches the input phase.
#[derive(Clone, Copy, Debug)]
pub struct BatchPickRequest {
    pub pixel: (u32, u32),
    pub fired: bool,
}

impl BatchPickRequest {
    pub fn new(pixel: (u32, u32)) -> Self {
        Self {
            pixel,
            fired: false,
        }
    }
}
