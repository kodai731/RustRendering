#[derive(Clone, Debug)]
pub struct FlameTrail {
    pub state: thyllore_effect_core::FlameTrailState,
    pub last_timeline_time: Option<f32>,
}

impl Default for FlameTrail {
    fn default() -> Self {
        Self {
            state: thyllore_effect_core::FlameTrailState::default(),
            last_timeline_time: None,
        }
    }
}
