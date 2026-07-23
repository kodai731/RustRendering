use std::io::Write;

use serde_json::{json, Value};

use crate::ecs::resource::{FlameDumpSink, FlameTemporalState};
use thyllore_render_core::{build_flame_ubo, FlameEffect, FlameUBO};

pub fn build_effect_json(effect: &FlameEffect) -> serde_json::Value {
    json!({
        "frame_index": effect.frame_index,
        "time": effect.time,
        "position": [effect.position.x, effect.position.y, effect.position.z],
        "height": effect.height,
        "radius": effect.radius,
        "sigma_t": effect.sigma_t,
        "intensity": effect.intensity,
        "color_base": [effect.color_base[0], effect.color_base[1], effect.color_base[2]],
        "color_tip": [effect.color_tip[0], effect.color_tip[1], effect.color_tip[2]],
        "temperature_base_k": effect.temperature_base_k,
        "temperature_tip_k": effect.temperature_tip_k,
        "use_blackbody": effect.use_blackbody,
        "noise_amplitude": effect.noise_amplitude,
        "noise_frequency": effect.noise_frequency,
        "noise_scroll_speed": effect.noise_scroll_speed,
        "coefficients": {
            "height_primitive": effect.coefficients.height_primitive,
            "radial": effect.coefficients.radial,
            "height": effect.coefficients.height
        },
        "temporal_weight": effect.temporal_weight,
        "light_position_world": [effect.light_position_world.x, effect.light_position_world.y, effect.light_position_world.z],
        "self_shadow_strength": effect.self_shadow_strength,
        "warp_amp": effect.warp_amp,
        "warp_freq": effect.warp_freq,
        "rise_speed": effect.rise_speed,
        "taper_power": effect.taper_power,
       "radius_tip_ratio": effect.radius_tip_ratio,
        "edge_low": effect.edge_low,
        "edge_high": effect.edge_high,
        "white_boost": effect.white_boost,
        "wind_direction": [effect.wind_direction.x, effect.wind_direction.y],
        "bend_amount": effect.bend_amount,
        "bend_power": effect.bend_power,
        "envelope_peak": effect.envelope_peak,
        "envelope_base": effect.envelope_base,
        "envelope_tail": effect.envelope_tail,
        "radial_sharpness": effect.radial_sharpness
    })
}

fn matrix4_to_array(m: &cgmath::Matrix4<f32>) -> [f32; 16] {
    [
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ]
}

pub fn build_ubo_json(ubo: &FlameUBO) -> serde_json::Value {
    json!({
        "model": matrix4_to_array(&ubo.model),
        "inverse_model": matrix4_to_array(&ubo.inverse_model),
        "height_primitive_coefficients": ubo.height_primitive_coefficients,
        "radial_coefficients": ubo.radial_coefficients,
        "height_coefficients": ubo.height_coefficients,
        "sigma_t": ubo.sigma_t,
        "intensity": ubo.intensity,
        "height_axis_scale": ubo.height_axis_scale,
        "noise_amplitude": ubo.noise_amplitude,
        "noise_frequency": ubo.noise_frequency,
        "noise_scroll_speed": ubo.noise_scroll_speed,
        "color_base": [ubo.color_base.x, ubo.color_base.y, ubo.color_base.z, ubo.color_base.w],
        "color_mid": [ubo.color_mid.x, ubo.color_mid.y, ubo.color_mid.z, ubo.color_mid.w],
        "color_tip": [ubo.color_tip.x, ubo.color_tip.y, ubo.color_tip.z, ubo.color_tip.w],
        "light_data": [ubo.light_data.x, ubo.light_data.y, ubo.light_data.z, ubo.light_data.w]
    })
}

pub fn build_temporal_json(state: &FlameTemporalState) -> serde_json::Value {
    let has_previous = state.previous.is_some();
    let previous = state.previous.as_ref().map(|snap| {
        json!({
            "view": matrix4_to_array(&snap.view),
            "appearance": build_effect_json(&snap.appearance),
            "settings": {
                "mode": snap.settings.shading_mode.as_shader_value(),
                "reference_step_count": snap.settings.reference_step_count,
                "noise_step_count": snap.settings.noise_step_count
            }
        })
    });
    json!({
        "has_previous": has_previous,
        "previous": previous
    })
}

pub fn build_flame_dump_record(
    effect: &FlameEffect,
    temporal: &FlameTemporalState,
    instance_index: usize,
    trail_enabled: bool,
    trail_len: usize,
    trail_fade_seconds: f32,
    trail_oldest_age: f32,
) -> Value {
    let ubo = build_flame_ubo(effect);
    let mut record = build_effect_json(effect).as_object().unwrap().clone();
    for (k, v) in build_ubo_json(&ubo).as_object().unwrap() {
        record.insert(k.clone(), v.clone());
    }
    record.insert("temporal_data".to_string(), build_temporal_json(temporal));
    record.insert("instance_index".to_string(), Value::Number(instance_index.into()));
    record.insert("trail_enabled".to_string(), Value::Bool(trail_enabled));
    record.insert("trail_len".to_string(), Value::Number(trail_len.into()));
    record.insert("trail_fade_seconds".to_string(), Value::Number(serde_json::Number::from_f64(trail_fade_seconds as f64).unwrap()));
    record.insert("trail_oldest_age".to_string(), Value::Number(serde_json::Number::from_f64(trail_oldest_age as f64).unwrap()));
    Value::Object(record)
}

pub fn flame_dump_system(
    sink: &mut FlameDumpSink,
    temporal: &FlameTemporalState,
    effects: &[FlameEffect],
    trails: &[Option<crate::ecs::component::flame_trail::FlameTrail>],
) {
    for (i, effect) in effects.iter().enumerate() {
        let trail = &trails[i];
        let (trail_enabled, trail_len, trail_fade_seconds, trail_oldest_age) = match trail {
            Some(t) => {
                let oldest_age = t.state.samples.last().map(|s| s.age_seconds).unwrap_or(0.0);
                (t.state.enabled, t.state.samples.len(), t.state.fade_seconds, oldest_age)
            }
            None => (false, 0, 0.8, 0.0),
        };
        let record = build_flame_dump_record(effect, temporal, i, trail_enabled, trail_len, trail_fade_seconds, trail_oldest_age);
        let line = serde_json::to_string(&record).expect("failed to serialize flame dump record");
        writeln!(sink.writer, "{}", line).expect("failed to write flame dump line");
    }
    sink.writer.flush().expect("failed to flush flame dump writer");
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Vector3, Vector4};
    use thyllore_render_core::FlameCoefficients;

    fn sample_effect() -> FlameEffect {
        FlameEffect {
            position: Vector3::new(1.0, 2.0, 3.0),
            height: 1.0,
            radius: 0.5,
            sigma_t: 0.1,
            intensity: 1.0,
            color_base: [1.0, 0.0, 0.0],
            color_tip: [0.0, 1.0, 0.0],
            temperature_base_k: 1000.0,
            temperature_tip_k: 500.0,
            use_blackbody: false,
            noise_amplitude: 0.0,
            noise_frequency: 0.0,
            noise_scroll_speed: 0.0,
            time: 0.0,
            time_scale: 1.0,
            time_offset: 0.0,
            coefficients: thyllore_render_core::fit_flame_coefficients(&thyllore_render_core::FlameProfile::default()),
            temporal_weight: 0.5,
            frame_index: 42,
            light_position_world: Vector3::new(2.0, 3.0, 2.0),
            self_shadow_strength: 0.5,
            warp_amp: 0.25,
            warp_freq: 2.5,
            rise_speed: 0.8,
            taper_power: 1.4,
            radius_tip_ratio: 0.1,
            edge_low: 0.3,
            edge_high: 0.7,
            white_boost: 0.0,
            wind_direction: cgmath::Vector2::new(0.0, 0.0),
            bend_amount: 0.0,
            bend_power: 1.7,
            envelope_peak: 0.35,
            envelope_base: 0.45,
            envelope_tail: 1.6,
            radial_sharpness: 4.0,
            rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0),
             ..FlameEffect::default()
            }
    }

    fn sample_temporal() -> FlameTemporalState {
        FlameTemporalState { previous: None }
    }

    #[test]
    fn build_flame_dump_record_produces_valid_json() {
        let effect = sample_effect();
        let temporal = sample_temporal();
        let record = build_flame_dump_record(&effect, &temporal, 0, false, 0, 0.8, 0.0);
        assert_eq!(record["frame_index"], 42);
        assert_eq!(record["time"], 0.0);
        assert_eq!(record["position"], json!([1.0, 2.0, 3.0]));
        assert_eq!(record["trail_enabled"], false);
        assert_eq!(record["trail_len"], 0);
        assert!((record["trail_fade_seconds"].as_f64().unwrap() - 0.8).abs() < 1e-5);
        assert_eq!(record["trail_oldest_age"], 0.0);
    }
}
