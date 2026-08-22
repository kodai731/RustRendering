use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameEdgeStyle {
        pub radius_tip_ratio: f32,
        pub edge_low: f32,
        pub edge_high: f32,
        pub white_boost: f32,
    }
}

declare_gpu_block! {
    /// Bridge model of smoothstep(edge_low, edge_high, x): two gaussians around a
    /// center (thyllore_math_core::ErfResponseModel).
    #[derive(Clone, Copy, Debug)]
    pub struct FlameErosionResponse {
        pub center: f32,
        pub kappa: f32,
        pub weight1: f32,
        pub weight2: f32,
    }
}

declare_gpu_block! {
    /// Near fade plus the amplitude-scaled effective erosion edge window.
    #[derive(Clone, Copy, Debug)]
    pub struct FlameNearFadeParams {
        pub radius: f32,
        pub carve_residual: f32,
        pub edge_low: f32,
        pub edge_high: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameTipCarveParams {
        pub depth: f32,
        pub inv_reach: f32,
        pub primitive_top: f32,
        pub inv_primitive_range: f32,
    }
}
