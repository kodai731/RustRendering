use crate::flame::constants::*;
use crate::flame::gpu::components::FlameBranchAgeProfile;

impl Default for FlameBranchAgeProfile {
    fn default() -> Self {
        Self {
            wind_fraction: BRANCH_WIND_FRACTION,
            burnout_start_fraction: BRANCH_BURNOUT_START_FRACTION,
            burnout_release_fraction: BRANCH_BURNOUT_RELEASE_FRACTION,
            burnout_margin: BRANCH_BURNOUT_MARGIN,
            burnout_trunk_inner: BRANCH_BURNOUT_TRUNK_INNER,
            _padding: [0.0; 3],
        }
    }
}
