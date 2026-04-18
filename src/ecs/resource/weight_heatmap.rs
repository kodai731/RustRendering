use crate::animation::BoneId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeatmapApplication {
    NotApplied,
    Applied(Option<BoneId>),
}

impl Default for HeatmapApplication {
    fn default() -> Self {
        Self::NotApplied
    }
}

#[derive(Clone, Debug, Default)]
pub struct WeightHeatmapState {
    pub enabled: bool,
    last_applied: HeatmapApplication,
}

impl WeightHeatmapState {
    pub fn last_applied(&self) -> &HeatmapApplication {
        &self.last_applied
    }

    pub fn set_last_applied(&mut self, state: HeatmapApplication) {
        self.last_applied = state;
    }
}
