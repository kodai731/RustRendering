use crate::flame::gpu::components::*;
use crate::flame_wave::{WAVE_MODE_COUNT, WAVE_MODE_SLOTS};
use cgmath::Matrix4;
use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameUBO {
        pub model: Matrix4<f32>,
        pub inverse_model: Matrix4<f32>,
        pub height_primitive_coefficients: [[f32; 4]; 3],
        pub radial_coefficients: [[f32; 4]; 2],
        pub height_coefficients: [[f32; 4]; 2],
        pub time: f32,
        pub sigma_t: f32,
        pub intensity: f32,
        pub height_axis_scale: f32,
        pub noise_amplitude: f32,
        pub noise_frequency: f32,
        pub noise_scroll_speed: f32,
        pub radial_sharpness: f32,
        pub color_base: FlameColorBase = nested FlameColorBase,
        pub color_mid: FlameColorMid = nested FlameColorMid,
        pub color_tip: FlameColorTip = nested FlameColorTip,
        pub temporal_data: FlameTemporalParams = nested FlameTemporalParams,
        pub light_data: FlameLightParams = nested FlameLightParams,
        pub warp_style: FlameWarpStyle = nested FlameWarpStyle,
        pub edge_style: FlameEdgeStyle = nested FlameEdgeStyle,
        pub wind_bend: FlameWindBend = nested FlameWindBend,
        pub trail_unit_inverse: Matrix4<f32>,
        pub trail_meta: FlameTrailMeta = nested FlameTrailMeta,
        pub trail_coefficients: [[f32; 4]; 4],
        pub emitter_params: FlameEmitterParams = nested FlameEmitterParams,
        pub contour_params: FlameContourParams = nested FlameContourParams,
        pub erosion_response: FlameErosionResponse = nested FlameErosionResponse,
        pub wave_cf_params: FlameWaveCfParams = nested FlameWaveCfParams,
        pub boundary_params: FlameBoundaryParams = nested FlameBoundaryParams,
        pub near_fade_params: FlameNearFadeParams = nested FlameNearFadeParams,
        pub radius_coefficients: [[f32; 4]; 2],
        pub color_ramp: [[f32; 4]; 8],
        /// Planckian chromaticity sampled from temperature_tip_k (index 0) to
        /// temperature_base_k (index 7); the shader interpolates by node temperature.
        pub temp_ramp: [[f32; 4]; 8],
        pub profile_params: FlameProfileParams = nested FlameProfileParams,
        pub wave_params: FlameWaveShaping = nested FlameWaveShaping,
        pub tip_carve_params: FlameTipCarveParams = nested FlameTipCarveParams,
        pub warp_strain_params: FlameWarpStrainParams = nested FlameWarpStrainParams,
        pub warp_form_params: FlameWarpFormParams = nested FlameWarpFormParams,
        pub unified_params: FlameUnifiedParams = nested FlameUnifiedParams,
        pub mix_params: FlameMixParams = nested FlameMixParams,
        pub segment_params: FlameSegmentParams = nested FlameSegmentParams,
        pub thermal_params: FlameThermalParams = nested FlameThermalParams,
        pub spread_params: FlameSpreadParams = nested FlameSpreadParams,
        pub support_motion: FlameSupportMotion = nested FlameSupportMotion,
        pub twist_field: FlameTwistField = nested FlameTwistField,
        pub meander_modes: [FlameMeanderMode; 2] = nested FlameMeanderMode,
        pub branch_field: FlameBranchField = nested FlameBranchField,
        pub wave_modes: [[f32; 4]; 2 * WAVE_MODE_SLOTS],
        pub wave_jitter: [[f32; 4]; WAVE_MODE_COUNT],
    }
}
