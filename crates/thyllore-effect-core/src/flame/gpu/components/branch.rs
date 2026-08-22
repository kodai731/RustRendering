use crate::flame::constants::*;
use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct FlameBranchElement {
        pub spawn_time: f32,
        pub side: f32,
        pub azimuth: f32,
        pub spawn_height: f32,
        /// Size multiplier of reach and core (scatter lane).
        pub size: f32,
        /// Tilt of the vortex line out of the horizontal [rad] (scatter lane).
        pub tilt: f32,
        /// Window center shift along the line in reach units (scatter lane).
        pub along_offset: f32,
        pub hash01: f32,
        /// Trunk support radius at the spawn height, in flame-local units; the
        /// reach and core radii are ratios of it.
        pub trunk_radius: f32,
        pub _padding: [f32; 3],
    }
}

declare_gpu_block! {
    /// Age-profile fractions of the vortex transport (winding, burnout), shared with
    /// the shader so both sides evaluate the same envelope.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct FlameBranchAgeProfile {
        pub wind_fraction: f32,
        pub burnout_start_fraction: f32,
        pub burnout_release_fraction: f32,
        pub burnout_margin: f32,
        pub burnout_trunk_inner: f32,
        pub _padding: [f32; 3],
    }
}

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

declare_gpu_block! {
    /// Branch element table (newest first) with the per-effect age-profile
    /// constants; `count` = 0 leaves every consumer bit-identical.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct FlameBranchField {
        pub count: f32,
        pub period: f32,
        pub life: f32,
        pub gain: f32,
        pub rise_rate: f32,
        pub drift_rate: f32,
        pub aspect: f32,
        pub core_radius: f32,
        pub reach_start: f32,
        pub reach_end: f32,
        pub envelope_time: f32,
        pub core_offset: f32,
        pub bounding_pad: f32,
        pub bounding_pad_y: f32,
        pub _padding1: [f32; 2],
        pub age_profile: FlameBranchAgeProfile = nested FlameBranchAgeProfile,
        pub elements: [FlameBranchElement; BRANCH_MAX_ELEMENTS] = nested FlameBranchElement,
    }
}
