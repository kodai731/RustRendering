use cgmath::Matrix4;

use super::FlameRenderSettings;
use crate::ecs::component::FlameEffect;

/// The frame state that history reuse depends on. Any difference between two
/// consecutive frames invalidates the accumulated history.
#[derive(Clone, PartialEq)]
pub struct FlameTemporalSnapshot {
    pub view: Matrix4<f32>,
    pub appearance: FlameEffect,
    pub settings: FlameRenderSettings,
}

#[derive(Default)]
pub struct FlameTemporalState {
    pub previous: Option<FlameTemporalSnapshot>,
}
