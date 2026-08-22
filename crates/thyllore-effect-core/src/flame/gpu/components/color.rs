use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameColorBase {
        pub rgb: [f32; 3],
        pub occlusion_lum_ref: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameColorMid {
        pub rgb: [f32; 3],
        pub _padding: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameColorTip {
        pub rgb: [f32; 3],
        pub _padding: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameThermalParams {
        pub density_exp: f32,
        pub temp_exp: f32,
        pub temp_hot_k: f32,
        pub temp_cold_k: f32,
        pub wien_c_k: f32,
        pub _padding: [f32; 3],
    }
}
