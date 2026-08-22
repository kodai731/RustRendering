use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameEmitterParams {
        pub kind: f32,
        pub ring_major_ratio: f32,
        pub ring_angular_speed: f32,
        /// Gaussian half-depth of the SDF billboard slab (emitter kind 2).
        pub sdf_slab_depth: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameContourParams {
        pub wiggle_amp: f32,
        pub aniso_axis_advect: f32,
        pub rte_bands: f32,
        pub sigma_dispersion: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameProfileParams {
        pub radius_active: f32,
        pub radius_max: f32,
        pub color_active: f32,
        pub _padding: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameSegmentParams {
        pub count: f32,
        pub inv_count: f32,
        pub _padding: [f32; 2],
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameSpreadParams {
        pub gain: f32,
        pub edge_outer_sharpen: f32,
        pub twist_gain: f32,
        pub erosion_noise_gain: f32,
    }
}
