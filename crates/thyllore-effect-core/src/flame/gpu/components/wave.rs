use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameWaveShaping {
        pub tracked_count: f32,
        pub env_coeff: f32,
        pub inverse_scale: f32,
        pub amplitude: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameWaveCfParams {
        pub enabled: f32,
        pub shear_layer_count: f32,
        pub skipped_power_plain: f32,
        pub skipped_power_env: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameMixParams {
        pub lo: f32,
        pub hi: f32,
        pub inv_carrier_std: f32,
        pub height_gain: f32,
        /// Wavenumber scale of the mixing eddies relative to the low erosion octave.
        pub scale: f32,
        pub radial_gain: f32,
        pub _padding: [f32; 2],
    }
}
