use super::warp_strain::build_warp_form_params;
use super::wave::{
    build_segment_params, build_wave_cf_params, build_wave_ubo_fields, WaveUboFields,
};
use crate::flame::*;
use crate::flame_trail::FlameTrailState;
use cgmath::{InnerSpace, Matrix4, Vector3};
use thyllore_math_core::{evaluate_chebyshev, fit_erf_response};

/// Medium spread gain alpha (motion_design L3); the reach shares the tip
/// carve inv_reach in the shader.
fn build_medium_spread_params(effect: &FlameEffect) -> FlameSpreadParams {
    FlameSpreadParams {
        gain: effect.spread_gain.max(0.0),
        edge_outer_sharpen: effect.edge.outer_sharpen,
        twist_gain: effect.twist.gain,
        erosion_noise_gain: effect.noise.erosion_gain,
    }
}

fn build_profile_params(effect: &FlameEffect, baked: &FlameBaked) -> FlameProfileParams {
    let inactive = FlameProfileParams {
        radius_active: 0.0,
        radius_max: 0.0,
        color_active: 0.0,
        _padding: 0.0,
    };
    if baked.radius.is_none() || baked.blend <= 0.0 {
        return inactive;
    }
    let series = thyllore_math_core::ChebyshevSeries::new(
        effect
            .coefficients
            .radius_scale
            .iter()
            .flatten()
            .copied()
            .collect(),
        (0.0, 1.0),
    );
    let mut max_val = 0.0f32;
    for i in 0..=32 {
        let h = i as f32 / 32.0;
        let val = evaluate_chebyshev(&series, h);
        if val > max_val {
            max_val = val;
        }
    }
    let color_flag = if baked.color.is_some() && baked.blend > 0.0 {
        1.0
    } else {
        0.0
    };
    FlameProfileParams {
        radius_active: 1.0,
        radius_max: max_val.max(0.05),
        color_active: color_flag,
        _padding: 0.0,
    }
}

pub struct FlameUboInputs<'a> {
    pub effect: &'a FlameEffect,
    pub baked: &'a FlameBaked,
    pub temporal: &'a FlameTemporalAccum,
    pub trail: Option<&'a FlameTrailState>,
}

struct FlameUboScratch {
    colors: ([f32; 3], [f32; 3], [f32; 3]),
    edge_window: (f32, f32),
    wave: WaveUboFields,
}

impl FlameUboScratch {
    fn compute(effect: &FlameEffect) -> Self {
        Self {
            colors: resolve_flame_colors(&effect.color),
            edge_window: effective_edge_window(&effect.edge, &effect.noise),
            wave: build_wave_ubo_fields(effect),
        }
    }
}

struct FlameUboPlacement {
    model: Matrix4<f32>,
    inverse_model: Matrix4<f32>,
    trail_unit_inverse: Matrix4<f32>,
    trail_meta: FlameTrailMeta,
    trail_coefficients: [[f32; 4]; 4],
}

fn build_placement(effect: &FlameEffect, trail: Option<&FlameTrailState>) -> FlameUboPlacement {
    let Some(trail) = trail else {
        return FlameUboPlacement {
            model: build_flame_model_matrix(effect),
            inverse_model: build_flame_inverse_model_matrix(effect),
            trail_unit_inverse: Matrix4::<f32>::from_scale(1.0),
            trail_meta: FlameTrailMeta {
                sample_count: 0.0,
                max_age: 0.0,
                _padding: [0.0; 2],
            },
            trail_coefficients: [[0.0; 4]; 4],
        };
    };

    let model = build_flame_trail_expanded_matrix(effect, &trail.samples);
    let inverse_model =
        Matrix4::from_nonuniform_scale(1.0 / model[0][0], 1.0 / model[1][1], 1.0 / model[2][2])
            * Matrix4::from_translation(-Vector3::new(model[3][0], model[3][1], model[3][2]));
    let (trail_unit_inverse, trail_meta, trail_coefficients) =
        build_flame_trail_ubo_fields(effect, trail);
    FlameUboPlacement {
        model,
        inverse_model,
        trail_unit_inverse,
        trail_meta,
        trail_coefficients,
    }
}

pub fn build_flame_ubo(
    effect: &FlameEffect,
    baked: &FlameBaked,
    temporal: &FlameTemporalAccum,
) -> FlameUBO {
    build_flame_ubo_from_inputs(&FlameUboInputs {
        effect,
        baked,
        temporal,
        trail: None,
    })
}

pub fn build_flame_ubo_with_trail(
    effect: &FlameEffect,
    baked: &FlameBaked,
    temporal: &FlameTemporalAccum,
    trail: Option<&FlameTrailState>,
    trail_render_active: bool,
) -> FlameUBO {
    let active_trail =
        trail.filter(|trail| trail_render_active && trail.enabled && !trail.samples.is_empty());
    build_flame_ubo_from_inputs(&FlameUboInputs {
        effect,
        baked,
        temporal,
        trail: active_trail,
    })
}

pub fn build_flame_ubo_from_inputs(inputs: &FlameUboInputs) -> FlameUBO {
    let FlameUboInputs {
        effect,
        baked,
        temporal,
        trail,
    } = *inputs;
    let scratch = FlameUboScratch::compute(effect);
    let (color_base, color_mid, color_tip) = scratch.colors;
    let edge_window = scratch.edge_window;
    let wave_fields = &scratch.wave;
    let placement = build_placement(effect, trail);

    FlameUBO {
        model: placement.model,
        inverse_model: placement.inverse_model,
        height_primitive_coefficients: effect.coefficients.height_primitive,
        radial_coefficients: effect.coefficients.radial,
        height_coefficients: effect.coefficients.height,
        time: effect.time,
        sigma_t: effective_sigma_t(effect),
        intensity: effect.intensity,
        height_axis_scale: 1.0,
        noise_amplitude: effect.noise.amplitude,
        noise_frequency: effect.noise.frequency,
        noise_scroll_speed: effect.noise.scroll_speed,
        radial_sharpness: effect.radial_sharpness,
        color_base: FlameColorBase {
            rgb: color_base,
            occlusion_lum_ref: effect.color.occlusion_lum_ref,
        },
        color_mid: FlameColorMid {
            rgb: color_mid,
            _padding: 1.0,
        },
        color_tip: FlameColorTip {
            rgb: color_tip,
            _padding: 0.0,
        },
        temporal_data: FlameTemporalParams {
            accum_weight: temporal.weight,
            frame_index: (temporal.frame_index % 16384) as f32,
            noise_aniso_y: effective_noise_aniso_y(&effect.noise, effect.height, effect.radius),
            warp_y_scale: effect.warp.y_scale,
        },
        light_data: build_light_params(effect),
        warp_style: build_warp_style(&effect.warp),
        edge_style: build_edge_style(&effect.edge, &effect.noise),
        wind_bend: build_wind_bend(&effect.wind),
        trail_unit_inverse: placement.trail_unit_inverse,
        trail_meta: placement.trail_meta,
        trail_coefficients: placement.trail_coefficients,
        emitter_params: build_emitter_params(&effect.emitter, effect.radius),
        contour_params: build_contour_params(&effect.contour),
        erosion_response: build_erosion_response(edge_window),
        wave_cf_params: FlameWaveCfParams {
            skipped_power_plain: wave_fields.skipped_power[0],
            skipped_power_env: wave_fields.skipped_power[1],
            ..build_wave_cf_params()
        },
        boundary_params: build_boundary_params(&effect.boundary),
        near_fade_params: build_near_fade_params(&effect.carve, edge_window),
        radius_coefficients: effect.coefficients.radius_scale,
        color_ramp: build_color_ramp(&effect.color, baked),
        temp_ramp: build_temperature_ramp(&effect.color),
        profile_params: build_profile_params(effect, baked),
        wave_params: wave_fields.shaping,
        tip_carve_params: build_tip_carve_params(&effect.carve, &effect.coefficients),
        warp_strain_params: build_warp_strain_params(effect),
        warp_form_params: build_warp_form_params(effect),
        unified_params: build_unified_field_params(effect),
        mix_params: build_mix_params(&effect.mix, wave_fields.low_carrier_std),
        segment_params: build_segment_params(effect),
        thermal_params: build_thermal_params(&effect.thermal, &effect.color),
        spread_params: build_medium_spread_params(effect),
        support_motion: FlameSupportMotion {
            support_margin: effect.support_margin,
            meander_amp: effect.meander.amp,
            swirl_speed: effect.swirl.speed,
            twist_speed: effect.twist.speed,
        },
        twist_field: build_twist_field(&effect.twist, &effect.swirl),
        meander_modes: build_meander_modes(&effect.meander, &effect.swirl),
        branch_field: build_branch_field(effect, baked),
        puff_field: build_puff_field(effect, baked),
        flow_field: build_flow_field(effect, baked),
        wave_modes: wave_fields.packed,
        wave_jitter: wave_fields.jitter,
    }
}

fn build_light_params(effect: &FlameEffect) -> FlameLightParams {
    let radius = flame_bounding_radius(effect).max(MIN_FLAME_EXTENT);
    let height = effect.height.max(MIN_FLAME_EXTENT);
    let relative = effect.light_position_world - effect.position;
    let direction = Vector3::new(
        relative.x / radius,
        relative.y / height,
        relative.z / radius,
    );
    let unit_direction = if direction.dot(direction) < 1e-6 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        direction.normalize()
    };
    FlameLightParams {
        direction: [unit_direction.x, unit_direction.y, unit_direction.z],
        self_shadow_strength: effect.self_shadow_strength,
    }
}

fn build_erosion_response(edge_window: (f32, f32)) -> FlameErosionResponse {
    let model = fit_erf_response(edge_window.0, edge_window.1);
    FlameErosionResponse {
        center: model.center,
        kappa: model.kappa,
        weight1: model.gaussian_weights[0],
        weight2: model.gaussian_weights[1],
    }
}

impl Default for FlameUBO {
    fn default() -> Self {
        build_flame_ubo(
            &FlameEffect::default(),
            &FlameBaked::default(),
            &FlameTemporalAccum::default(),
        )
    }
}
