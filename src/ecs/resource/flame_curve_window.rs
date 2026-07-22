use std::collections::HashSet;

use crate::ecs::component::FlameParam;

#[derive(Clone, Debug)]
pub struct FlameCurveWindowState {
    pub open: bool,
    pub hidden_params: HashSet<FlameParam>,
}

impl Default for FlameCurveWindowState {
    fn default() -> Self {
        Self {
            open: false,
            hidden_params: HashSet::new(),
        }
    }
}
