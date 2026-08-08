use std::io::Write;

use serde_json::{json, Value};

use crate::ecs::resource::{Camera, FlameDumpSink, FlameRenderSettings, FlameTemporalState};
use crate::ecs::World;
use thyllore_render_core::{build_flame_ubo, probe_flame_wall, FlameEffect, FlameUBO, WallProbeView};

pub fn build_effect_json(effect: &FlameEffect) -> serde_json::Value {
    let mut value = json!({
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
       "noise_aniso_y": effect.noise_aniso_y,
        "warp_y_scale": effect.warp_y_scale,
      "coefficients": {
            "height_primitive": effect.coefficients.height_primitive,
            "radial": effect.coefficients.radial,
            "height": effect.coefficients.height,
            "radius_scale": effect.coefficients.radius_scale
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
        "radial_sharpness": effect.radial_sharpness,
        "occlusion_lum_ref": effect.occlusion_lum_ref,
        "contour_wiggle_amp": effect.contour_wiggle_amp
    });
    value["rotation"] = json!([
        effect.rotation.s,
        effect.rotation.v.x,
        effect.rotation.v.y,
        effect.rotation.v.z
    ]);
    value["time_scale"] = json!(effect.time_scale);
    value["time_offset"] = json!(effect.time_offset);
    value["emitter_kind"] = json!(effect.emitter_kind);
    value["ring_major_radius"] = json!(effect.ring_major_radius);
    value["ring_angular_speed"] = json!(effect.ring_angular_speed);
    value["aniso_axis_advect"] = json!(effect.aniso_axis_advect);
    value["rte_bands"] = json!(effect.rte_bands);
    value["sigma_dispersion"] = json!(effect.sigma_dispersion);
    value["edge_temperature_blend"] = json!(effect.edge_temperature_blend);
    value["boundary_amp"] = json!(effect.boundary_amp);
    value["boundary_freq"] = json!(effect.boundary_freq);
    value["boundary_speed"] = json!(effect.boundary_speed);
    value["boundary_radius_ratio"] = json!(effect.boundary_radius_ratio);
    value["baked_blend"] = json!(effect.baked_blend);
    value["baked_envelope"] = json!(effect.baked_envelope.map(|a| a.to_vec()));
    value["baked_radius"] = json!(effect.baked_radius.map(|a| a.to_vec()));
    value["baked_color"] = json!(effect
        .baked_color
        .map(|a| a.iter().map(|c| c.to_vec()).collect::<Vec<_>>()));
    value
}

fn matrix4_to_array(m: &cgmath::Matrix4<f32>) -> [f32; 16] {
    [
        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1],
        m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3],
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
    record.insert(
        "instance_index".to_string(),
        Value::Number(instance_index.into()),
    );
    record.insert("trail_enabled".to_string(), Value::Bool(trail_enabled));
    record.insert("trail_len".to_string(), Value::Number(trail_len.into()));
    record.insert(
        "trail_fade_seconds".to_string(),
        Value::Number(serde_json::Number::from_f64(trail_fade_seconds as f64).unwrap()),
    );
    record.insert(
        "trail_oldest_age".to_string(),
        Value::Number(serde_json::Number::from_f64(trail_oldest_age as f64).unwrap()),
    );
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
                (
                    t.state.enabled,
                    t.state.samples.len(),
                    t.state.fade_seconds,
                    oldest_age,
                )
            }
            None => (false, 0, 0.8, 0.0),
        };
        let record = build_flame_dump_record(
            effect,
            temporal,
            i,
            trail_enabled,
            trail_len,
            trail_fade_seconds,
            trail_oldest_age,
        );
        let line = serde_json::to_string(&record).expect("failed to serialize flame dump record");
        writeln!(sink.writer, "{}", line).expect("failed to write flame dump line");
    }
    sink.writer
        .flush()
        .expect("failed to flush flame dump writer");
}

pub fn build_wall_probe_camera_json(camera: &crate::ecs::resource::Camera) -> Value {
    use super::camera_systems::{
        compute_camera_direction, compute_camera_position, compute_camera_right, compute_camera_up,
    };
    let position = compute_camera_position(camera);
    let forward = compute_camera_direction(camera);
    let right = compute_camera_right(camera);
    let up = compute_camera_up(camera);
    let batch_camera = format!(
        "{:.3},{:.3},{:.4},{:.4},{:.4},{:.4}",
        camera.yaw.to_degrees(),
        camera.pitch.to_degrees(),
        camera.distance,
        camera.pivot.x,
        camera.pivot.y,
        camera.pivot.z
    );
    json!({
        "pivot": [camera.pivot.x, camera.pivot.y, camera.pivot.z],
        "yaw_degrees": camera.yaw.to_degrees(),
        "pitch_degrees": camera.pitch.to_degrees(),
        "distance": camera.distance,
        "fov_y_degrees": camera.fov_y.0,
        "near_plane": camera.near_plane,
        "position": [position.x, position.y, position.z],
        "forward": [forward.x, forward.y, forward.z],
        "right": [right.x, right.y, right.z],
        "up": [up.x, up.y, up.z],
        "batch_camera": batch_camera
    })
}

fn build_wall_probe_ray_json(ray: &thyllore_render_core::WallProbeRay) -> Value {
    json!({
        "ndc": ray.ndc,
        "hit": ray.hit,
        "t_enter": ray.t_enter,
        "t_exit": ray.t_exit,
        "chord": ray.chord,
        "chord_world": ray.chord_world,
        "noise_cells": ray.noise_cells,
        "grazing_deg": ray.grazing_deg,
        "density_mean": ray.density_mean,
        "density_max": ray.density_max,
        "saturated_fraction": ray.saturated_fraction,
        "tau": ray.tau,
        "pixels_per_cell": ray.pixels_per_cell,
        "segment_dt": ray.segment_dt,
        "cells_per_segment": ray.cells_per_segment,
        "unresolved_fraction": ray.unresolved_fraction,
        "support_intervals": ray.support_intervals,
        "support_gap": ray.support_gap
    })
}

pub fn build_wall_probe_json(report: &thyllore_render_core::WallProbeReport) -> Value {
    json!({
        "camera_local": report.camera_local,
        "camera_density": report.camera_density,
        "camera_inside_support": report.camera_inside_support,
        "emitter_approximated": report.emitter_approximated,
        "summary": {
            "hit_fraction": report.hit_fraction,
            "tangential_hit_fraction": report.tangential_hit_fraction,
            "saturated_hit_fraction": report.saturated_hit_fraction,
            "median_noise_cells": report.median_noise_cells,
            "median_pixels_per_cell": report.median_pixels_per_cell,
            "mean_tau": report.mean_tau
        },
        "rays": report.rays.iter().map(build_wall_probe_ray_json).collect::<Vec<_>>()
    })
}

pub fn build_wall_probe_dump_record(
    camera: &crate::ecs::resource::Camera,
    settings: &crate::ecs::resource::FlameRenderSettings,
    viewport_size: [f32; 2],
    flames: &[(FlameEffect, thyllore_render_core::WallProbeReport)],
    unix_time: u64,
) -> Value {
    json!({
        "schema": "flame-wall-probe-v1",
        "unix_time": unix_time,
        "viewport_size_px": viewport_size,
        "camera": build_wall_probe_camera_json(camera),
        "render_settings": {
            "shading_mode": settings.shading_mode.label(),
            "reference_step_count": settings.reference_step_count,
            "noise_step_count": settings.noise_step_count
        },
        "flames": flames.iter().map(|(effect, report)| {
            let mut entry = build_effect_json(effect).as_object().unwrap().clone();
            entry.insert("wall_probe".to_string(), build_wall_probe_json(report));
            Value::Object(entry)
        }).collect::<Vec<_>>()
    })
}

pub fn write_flame_wall_probe_dump(
    camera: &crate::ecs::resource::Camera,
    settings: &crate::ecs::resource::FlameRenderSettings,
    viewport_size: [f32; 2],
    flames: &[(FlameEffect, thyllore_render_core::WallProbeReport)],
) -> std::io::Result<std::path::PathBuf> {
    let unix_time = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let record = build_wall_probe_dump_record(camera, settings, viewport_size, flames, unix_time);

    let directory = std::path::Path::new("log/flame");
    std::fs::create_dir_all(directory)?;
    let path = directory.join(format!("wall_probe_{}.json", unix_time));
    std::fs::write(&path, serde_json::to_string_pretty(&record)?)?;
    Ok(path)
}

/// Build and write the wall-probe dump from the world's camera + flame entities.
/// This is the shared implementation used by both the interactive mode (UIEvent)
/// and the batch run path (synchronous call after render).
pub fn perform_flame_wall_probe_dump(world: &World, viewport_size: [f32; 2]) {
    use crate::ecs::systems::camera_systems::{
        compute_camera_direction, compute_camera_position, compute_camera_right, compute_camera_up,
    };

    let camera = (*world.resource::<Camera>()).clone();
    let settings = world
        .get_resource::<FlameRenderSettings>()
        .map(|s| *s)
        .unwrap_or_default();
    let view = WallProbeView {
        position: compute_camera_position(&camera).into(),
        forward: compute_camera_direction(&camera).into(),
        right: compute_camera_right(&camera).into(),
        up: compute_camera_up(&camera).into(),
        fov_y_radians: camera.fov_y.0.to_radians(),
        viewport_size_px: viewport_size,
    };

    let flames: Vec<_> = world
        .query_flames()
        .into_iter()
        .filter_map(|entity| world.get_component::<FlameEffect>(entity))
        .map(|effect| {
            let report = probe_flame_wall(&effect, &view);
            (effect.clone(), report)
        })
        .collect();
    if flames.is_empty() {
        log_warn!("wall probe dump skipped: no flame entity");
        return;
    }

    match write_flame_wall_probe_dump(&camera, &settings, viewport_size, &flames) {
        Ok(path) => log!("wall probe dumped to {}", path.display()),
        Err(error) => log_warn!("wall probe dump failed: {}", error),
    }
}

/// Provenance dump of one texture-fit load (G10): a metadata json plus a

/// Provenance dump of one texture-fit load (G10): a metadata json plus a
/// verbatim copy of the source bytes under `log/flame/texture_fit_<ts>.{json,png}`,
/// so a user-applied fit state can be bit-reproduced later
/// (`--batch-flame-texture <copy>,<blend>,<mode>` + the wall probe camera).
/// Failures are recorded too — a fit that silently no-ops must leave a trace.
/// Best-effort: dump IO errors never disturb the fit itself.
pub fn write_texture_fit_provenance(
    route: &str,
    path: &str,
    source_bytes: Option<&[u8]>,
    request: Value,
    result: Value,
    effect_before: &FlameEffect,
    effect_after: &FlameEffect,
) {
    const MAX_COPY_BYTES: usize = 32 * 1024 * 1024;
    let unix_time = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let directory = std::path::Path::new("log/flame");
    if std::fs::create_dir_all(directory).is_err() {
        return;
    }

    let mut sha256 = Value::Null;
    let mut copy_name = Value::Null;
    if let Some(bytes) = source_bytes {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        sha256 = json!(format!("{:x}", hasher.finalize()));
        if bytes.len() <= MAX_COPY_BYTES {
            let name = format!("texture_fit_{unix_time}.png");
            if std::fs::write(directory.join(&name), bytes).is_ok() {
                copy_name = json!(name);
            }
        }
    }

    let before = build_effect_json(effect_before);
    let after = build_effect_json(effect_after);
    let changed: Vec<&String> = match (&before, &after) {
        (Value::Object(before), Value::Object(after)) => before
            .iter()
            .filter(|(key, value)| after.get(*key) != Some(value))
            .map(|(key, _)| key)
            .collect(),
        _ => Vec::new(),
    };

    let record = json!({
        "schema": "texture_fit_provenance_v1",
        "unix_time": unix_time,
        "route": route,
        "source": {
            "path": std::fs::canonicalize(path)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| path.to_string()),
            "bytes": source_bytes.map(|b| b.len()),
            "sha256": sha256,
            "copy": copy_name,
        },
        "request": request,
        "result": result,
        "changed_fields": changed,
        "effect_before": before,
        "effect_after": after,
    });
    if let Ok(text) = serde_json::to_string_pretty(&record) {
        let _ = std::fs::write(
            directory.join(format!("texture_fit_{unix_time}.json")),
            text,
        );
    }
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
            coefficients: thyllore_render_core::fit_flame_coefficients(
                &thyllore_render_core::FlameProfile::default(),
            ),
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
            occlusion_lum_ref: 1.0,
            rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0),
            ..FlameEffect::default()
        }
    }

    fn sample_temporal() -> FlameTemporalState {
        FlameTemporalState { previous: None }
    }

    #[test]
    fn build_wall_probe_dump_record_reproduces_camera_as_batch_arg() {
        let camera = crate::ecs::resource::Camera {
            yaw: 30.0f32.to_radians(),
            pitch: (-16.0f32).to_radians(),
            distance: 1.15,
            pivot: cgmath::Vector3::new(0.0, 1.15, 0.0),
            ..crate::ecs::resource::Camera::default()
        };
        let settings = crate::ecs::resource::FlameRenderSettings::default();
        let effect = sample_effect();
        let view = thyllore_render_core::WallProbeView {
            position: [0.0, 1.15, 1.15],
            forward: [0.0, 0.0, -1.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y_radians: 45.0f32.to_radians(),
            viewport_size_px: [1680.0, 840.0],
        };
        let report = thyllore_render_core::probe_flame_wall(&effect, &view);
        let record = build_wall_probe_dump_record(
            &camera,
            &settings,
            [1680.0, 840.0],
            &[(effect, report)],
            123,
        );

        assert_eq!(record["schema"], "flame-wall-probe-v1");
        assert_eq!(record["unix_time"], 123);
        assert_eq!(
            record["camera"]["batch_camera"],
            "30.000,-16.000,1.1500,0.0000,1.1500,0.0000"
        );
        let flame = &record["flames"][0];
        assert_eq!(flame["position"], json!([1.0, 2.0, 3.0]));
        let summary = &flame["wall_probe"]["summary"];
        assert!(summary["hit_fraction"].is_number());
        assert_eq!(
            flame["wall_probe"]["rays"].as_array().unwrap().len(),
            thyllore_render_core::WALL_PROBE_GRID_COLS * thyllore_render_core::WALL_PROBE_GRID_ROWS
        );
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
