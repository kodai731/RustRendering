/// Per-frame accumulation state for temporal history reuse. Lives outside
/// `FlameEffect` so the appearance snapshot comparison never has to strip
/// per-frame fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlameTemporalAccum {
    pub weight: f32,
    pub frame_index: u64,
}
