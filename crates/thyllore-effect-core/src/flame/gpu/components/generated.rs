// Generated from SPIR-V by `cargo run -p thyllore-shader-manifest --bin generate_gpu_blocks`; do not edit.
use cgmath::Matrix4;
use thyllore_spirv_reflect::declare_gpu_block;

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameBoundaryParams {
        pub amp: f32,
        pub freq: f32,
        pub speed: f32,
        pub radius_ratio: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct FlameBranchAgeProfile {
        pub wind_fraction: f32,
        pub burnout_start_fraction: f32,
        pub burnout_release_fraction: f32,
        pub burnout_margin: f32,
        pub burnout_trunk_inner: f32,
        pub _padding: [f32; 3],
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct FlameBranchElement {
        pub spawn_time: f32,
        pub side: f32,
        pub azimuth: f32,
        pub spawn_height: f32,
        pub size: f32,
        pub tilt: f32,
        pub along_offset: f32,
        pub hash01: f32,
        pub trunk_radius: f32,
        pub _padding: [f32; 3],
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct FlameBranchField {
        pub count: f32,
        pub period: f32,
        pub life: f32,
        pub gain: f32,
        pub rise_rate: f32,
        pub drift_rate: f32,
        pub aspect: f32,
        pub core_radius: f32,
        pub reach_start: f32,
        pub reach_end: f32,
        pub envelope_time: f32,
        pub core_offset: f32,
        pub bounding_pad: f32,
        pub bounding_pad_y: f32,
        pub _padding: [f32; 2],
        pub age_profile: FlameBranchAgeProfile = nested FlameBranchAgeProfile,
        pub elements: [FlameBranchElement; 32] = nested FlameBranchElement,
    }
}

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
    pub struct FlameContourParams {
        pub wiggle_amp: f32,
        pub aniso_axis_advect: f32,
        pub rte_bands: f32,
        pub sigma_dispersion: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameEdgeStyle {
        pub radius_tip_ratio: f32,
        pub edge_low: f32,
        pub edge_high: f32,
        pub white_boost: f32,
        pub base_spread: f32,
        pub base_spread_height: f32,
        pub _padding: [f32; 2],
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameEmitterParams {
        pub kind: f32,
        pub ring_major_ratio: f32,
        pub ring_angular_speed: f32,
        pub sdf_slab_depth: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameErosionResponse {
        pub center: f32,
        pub kappa: f32,
        pub weight1: f32,
        pub weight2: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameFlowField {
        pub gain: f32,
        pub count: f32,
        pub _padding: [f32; 2],
        pub markers: [[f32; 4]; 32],
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameLightParams {
        pub direction: [f32; 3],
        pub self_shadow_strength: f32,
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
    pub struct FlameMixParams {
        pub lo: f32,
        pub hi: f32,
        pub inv_carrier_std: f32,
        pub height_gain: f32,
        pub scale: f32,
        pub radial_gain: f32,
        pub core_radius: f32,
        pub _padding: f32,
    }
}

declare_gpu_block! {
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
    pub struct FlameProfileParams {
        pub radius_active: f32,
        pub radius_max: f32,
        pub color_active: f32,
        pub _padding: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct FlamePuffField {
        pub count: f32,
        pub gain: f32,
        pub aspect: f32,
        pub _padding: f32,
        pub puffs: [[f32; 4]; 16],
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

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameSupportMotion {
        pub support_margin: f32,
        pub meander_amp: f32,
        pub swirl_speed: f32,
        pub twist_speed: f32,
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
    pub struct FlameThermalParams {
        pub density_exp: f32,
        pub temp_exp: f32,
        pub temp_hot_k: f32,
        pub temp_cold_k: f32,
        pub wien_ck: f32,
        pub _padding: [f32; 3],
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
    pub struct FlameTwistField {
        pub modes: [FlameTwistMode; 2] = nested FlameTwistMode,
        pub core_radius_sq: f32,
        pub _padding: [f32; 3],
    }
}

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
    pub struct FlameUnifiedParams {
        pub enabled: f32,
        pub sigma_floor: f32,
        pub _padding: [f32; 2],
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
    pub struct FlameWarpStrainParams {
        pub strain_base: f32,
        pub strain_tip: f32,
        pub inv_reach: f32,
        pub inv_strain_norm: f32,
    }
}

declare_gpu_block! {
    #[derive(Clone, Copy, Debug)]
    pub struct FlameWarpStyle {
        pub warp_amp: f32,
        pub warp_freq: f32,
        pub rise_speed: f32,
        pub taper_power: f32,
        pub rise_accel: f32,
        pub _padding: [f32; 3],
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
    pub struct FlameWaveShaping {
        pub tracked_count: f32,
        pub env_coeff: f32,
        pub inverse_scale: f32,
        pub amplitude: f32,
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
        pub puff_field: FlamePuffField = nested FlamePuffField,
        pub flow_field: FlameFlowField = nested FlameFlowField,
        pub wave_modes: [[f32; 4]; 428],
        pub wave_jitter: [[f32; 4]; 96],
    }
}
