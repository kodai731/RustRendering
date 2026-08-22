use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameTwistMode {
        pub kappa: f32,
        pub omega: f32,
        pub phase: f32,
        pub amp: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameTwistField {
        pub modes: [FlameTwistMode; 2] = nested FlameTwistMode,
        pub core_radius_sq: f32,
        pub _padding: [f32; 3],
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameMeanderMode {
        pub direction: [f32; 2],
        pub kappa: f32,
        pub omega: f32,
        pub phase: f32,
        pub _padding: [f32; 3],
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameSupportMotion {
        pub support_margin: f32,
        pub meander_amp: f32,
        pub swirl_speed: f32,
        /// 0 = delegate the twist rate to swirl_speed.
        pub twist_speed: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameBoundaryParams {
        pub amp: f32,
        pub freq: f32,
        pub speed: f32,
        pub radius_ratio: f32,
    }
}
