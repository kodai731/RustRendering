use cgmath::Matrix4;

use super::WaterRenderSettings;
use crate::ecs::component::WaterTorusEffect;

/// The frame state that history reuse depends on. Any difference between two
/// consecutive frames invalidates the accumulated history.
#[derive(Clone, PartialEq)]
pub struct WaterTemporalSnapshot {
    pub view: Matrix4<f32>,
    pub effect: WaterTorusEffect,
    pub settings: WaterRenderSettings,
}

#[derive(Default)]
pub struct WaterTemporalState {
    pub previous: Option<WaterTemporalSnapshot>,
}
