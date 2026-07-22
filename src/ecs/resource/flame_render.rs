pub use thyllore_render_core::{FlameEffect, FlameRenderSettings, FlameShadingMode};

#[derive(Clone, Copy, Debug)]
pub struct SelectedFlameInstance(pub usize);

impl Default for SelectedFlameInstance {
    fn default() -> Self {
        SelectedFlameInstance(0)
    }
}
