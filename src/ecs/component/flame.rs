pub use thyllore_effect_core::flame_plume::HeatPlume;
pub use thyllore_effect_core::{FlameBaked, FlameEffect, FlameTemporalAccum};

/// Provenance of the last style applied to this flame. Values are baked into
/// FlameEffect on apply; this records only which style they came from, so a
/// saved scene names its look without depending on the style file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppliedFlameStyle {
    pub name: String,
    pub version: u32,
}
