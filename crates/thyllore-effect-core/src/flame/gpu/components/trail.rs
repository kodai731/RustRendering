use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameTrailMeta {
        pub sample_count: f32,
        pub max_age: f32,
        pub _padding: [f32; 2],
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameTemporalParams {
        pub accum_weight: f32,
        pub frame_index: f32,
        pub noise_aniso_y: f32,
        pub warp_y_scale: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameLightParams {
        pub direction: [f32; 3],
        pub self_shadow_strength: f32,
    }
}
