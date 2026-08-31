/// Per-frame accumulation state for temporal history reuse. Lives outside
/// `WaterTorusEffect` so the appearance snapshot comparison never has to strip
/// per-frame fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterTemporalAccum {
    pub weight: f32,
    pub frame_index: u64,
}
