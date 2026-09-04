// Generated from SPIR-V by `cargo run -p thyllore-shader-manifest --bin generate_gpu_blocks`; do not edit.
use cgmath::Matrix4;
use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct WaterUBO {
        pub model: Matrix4<f32>,
        pub inverse_model: Matrix4<f32>,
        pub radii: [f32; 4],
        pub absorption: [f32; 4],
        pub flow: [f32; 4],
        pub composite: [f32; 4],
        pub tint: [f32; 4],
        pub lighting: [f32; 4],
        pub scattering: [f32; 4],
        pub temporal: [f32; 4],
        pub wave_modes: [[f32; 4]; 16],
        pub inv_view_proj: Matrix4<f32>,
        pub lb_modes: [[f32; 4]; 20],
    }
}
