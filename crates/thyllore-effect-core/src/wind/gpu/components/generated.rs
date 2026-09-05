// Generated from SPIR-V by `cargo run -p thyllore-shader-manifest --bin generate_gpu_blocks`; do not edit.
use cgmath::Matrix4;
use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct WindUBO {
        pub model: Matrix4<f32>,
        pub inverse_model: Matrix4<f32>,
        pub shape: [f32; 4],
        pub core: [f32; 4],
        pub optics: [f32; 4],
        pub albedo: [f32; 4],
        pub inv_view_proj: Matrix4<f32>,
    }
}
