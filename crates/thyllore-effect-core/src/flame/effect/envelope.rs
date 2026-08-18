/// Height envelope of the emission: peak height, base level and tail length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameEnvelope {
    pub peak: f32,
    pub base: f32,
    pub tail: f32,
}

impl Default for FlameEnvelope {
    fn default() -> Self {
        Self {
            peak: 0.25,
            base: 0.05,
            tail: 1.25,
        }
    }
}
