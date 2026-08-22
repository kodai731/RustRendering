use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameWarpStyle {
        pub warp_amp: f32,
        pub warp_freq: f32,
        pub rise_speed: f32,
        pub taper_power: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameWarpStrainParams {
        pub strain_base: f32,
        pub strain_tip: f32,
        pub inv_reach: f32,
        pub inv_strain_norm: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameWarpFormParams {
        pub displacement_form: f32,
        pub burnout_gain: f32,
        pub _padding: [f32; 2],
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameWindBend {
        pub wind_direction: [f32; 2],
        pub bend_amount: f32,
        pub bend_power: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameUnifiedParams {
        pub enabled: f32,
        pub sigma_floor: f32,
        pub _padding: [f32; 2],
    }
}
