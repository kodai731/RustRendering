use std::io::Write;

use serde_json::{json, Value};

use crate::ecs::resource::{Camera, FlameDumpSink, FlameRenderSettings, FlameTemporalState};
use crate::ecs::World;
use thyllore_effect_core::{
    build_flame_ubo, probe_flame_wall, FlameBaked, FlameEffect, FlameTemporalAccum, FlameUBO,
    WallProbeView,
};

pub fn build_effect_json(
    effect: &FlameEffect,
    baked: &FlameBaked,
    temporal: &FlameTemporalAccum,
) -> serde_json::Value {
    let mut value = json!({
        "frame_index": temporal.frame_index,
        "time": effect.time,
        "position": [effect.position.x, effect.position.y, effect.position.z],
        "height": effect.height,
        "radius": effect.radius,
        "sigma_t": effect.sigma_t,
        "intensity": effect.intensity,
        "color_base": [effect.color.base[0], effect.color.base[1], effect.color.base[2]],
        "color_tip": [effect.color.tip[0], effect.color.tip[1], effect.color.tip[2]],
        "temperature_base_k": effect.color.temperature_base_k,
        "temperature_tip_k": effect.color.temperature_tip_k,
        "use_blackbody": effect.color.use_blackbody,
        "noise_amplitude": effect.noise.amplitude,
        "noise_contrast": effect.noise.contrast,
        "noise_frequency": effect.noise.frequency,
        "noise_scroll_speed": effect.noise.scroll_speed,
       "noise_aniso_y": effect.noise.aniso_y,
        "noise_lobe_scale": effect.noise.lobe_scale,
        "noise_lobe_aniso": effect.noise.lobe_aniso,
        "warp_y_scale": effect.warp.y_scale,
      "coefficients": {
            "height_primitive": effect.coefficients.height_primitive,
            "radial": effect.coefficients.radial,
            "height": effect.coefficients.height,
            "radius_scale": effect.coefficients.radius_scale
        },
        "temporal_weight": temporal.weight,
        "light_position_world": [effect.light_position_world.x, effect.light_position_world.y, effect.light_position_world.z],
        "self_shadow_strength": effect.self_shadow_strength,
        "warp_amp": effect.warp.amp,
        "warp_freq": effect.warp.freq,
        "rise_speed": effect.warp.rise_speed,
        "taper_power": effect.warp.taper_power,
       "radius_tip_ratio": effect.edge.radius_tip_ratio,
        "edge_low": effect.edge.low,
        "edge_high": effect.edge.high,
        "white_boost": effect.edge.white_boost,
        "wind_direction": [effect.wind.direction.x, effect.wind.direction.y],
        "bend_amount": effect.wind.bend_amount,
        "bend_power": effect.wind.bend_power,
        "envelope_peak": effect.envelope.peak,
        "envelope_base": effect.envelope.base,
        "envelope_tail": effect.envelope.tail,
        "radial_sharpness": effect.radial_sharpness,
        "occlusion_lum_ref": effect.color.occlusion_lum_ref,
        "contour_wiggle_amp": effect.contour.wiggle_amp
    });
    value["rotation"] = json!([
        effect.rotation.s,
        effect.rotation.v.x,
        effect.rotation.v.y,
        effect.rotation.v.z
    ]);
    value["time_scale"] = json!(effect.time_scale);
    value["time_offset"] = json!(effect.time_offset);
    value["emitter_kind"] = json!(effect.emitter.kind);
    value["ring_major_radius"] = json!(effect.emitter.ring_major_radius);
    value["ring_angular_speed"] = json!(effect.emitter.ring_angular_speed);
    value["aniso_axis_advect"] = json!(effect.contour.aniso_axis_advect);
    value["rte_bands"] = json!(effect.contour.rte_bands);
    value["sigma_dispersion"] = json!(effect.contour.sigma_dispersion);
    value["boundary_amp"] = json!(effect.boundary.amp);
    value["boundary_freq"] = json!(effect.boundary.freq);
    value["boundary_speed"] = json!(effect.boundary.speed);
    value["boundary_radius_ratio"] = json!(effect.boundary.radius_ratio);
    value["tip_carve_depth"] = json!(effect.carve.tip.depth);
    value["tip_carve_reach"] = json!(effect.carve.tip.reach);
    value["warp_reach"] = json!(effect.warp.reach);
    value["swirl_gain"] = json!(effect.swirl.gain);
    value["swirl_speed"] = json!(effect.swirl.speed);
    value["spread_gain"] = json!(effect.spread_gain);
    value["support_margin"] = json!(effect.support_margin);
    value["edge_outer_sharpen"] = json!(effect.edge.outer_sharpen);
    value["base_spread"] = json!(effect.edge.base_spread);
    value["base_spread_height"] = json!(effect.edge.base_spread_height);
    value["noise_scale_mode"] = json!(effect.noise.scale_mode);
    value["erosion_noise_gain"] = json!(effect.noise.erosion_gain);
    value["twist_gain"] = json!(effect.twist.gain);
    value["twist_speed"] = json!(effect.twist.speed);
    value["burnout_gain"] = json!(effect.carve.burnout_gain);
    value["noise_shaping_scale"] = json!(effect.noise.shaping_scale);
    value["optical_depth"] = json!(effect.optical_depth);
    value["meander_amp"] = json!(effect.meander.amp);
    value["meander_frequency"] = json!(effect.meander.frequency);
    value["mix_lo"] = json!(effect.mix.lo);
    value["mix_hi"] = json!(effect.mix.hi);
    value["mix_height_gain"] = json!(effect.mix.height_gain);
    value["mix_scale"] = json!(effect.mix.scale);
    value["mix_radial_gain"] = json!(effect.mix.radial_gain);
    value["mix_core_radius"] = json!(effect.mix.core_radius);
    value["density_exp"] = json!(effect.thermal.density_exp);
    value["temp_exp"] = json!(effect.thermal.temp_exp);
    value["wien_c_k"] = json!(effect.thermal.wien_c_k);
    value["wave_segments"] = json!(effect.wave_segments);
    value["branch_period"] = json!(effect.branch.period);
    value["branch_life"] = json!(effect.branch.life);
    value["branch_gain"] = json!(effect.branch.gain);
    value["branch_core_radius"] = json!(effect.branch.core_radius);
    value["branch_core_offset"] = json!(effect.branch.core_offset);
    value["branch_reach"] = json!(effect.branch.reach);
    value["branch_spread"] = json!(effect.branch.spread);
    value["branch_spawn_height"] = json!(effect.branch.spawn_height);
    value["branch_spawn_range"] = json!(effect.branch.spawn_range);
    value["branch_seed"] = json!(effect.branch.seed);
    value["puff_gain"] = json!(effect.puff.gain);
    value["puff_period"] = json!(effect.puff.period);
    value["puff_rise"] = json!(effect.puff.rise);
    value["puff_radius"] = json!(effect.puff.radius);
    value["puff_spread"] = json!(effect.puff.spread);
    value["puff_decay"] = json!(effect.puff.decay);
    value["puff_aspect"] = json!(effect.puff.aspect);
    value["flow_gain"] = json!(effect.flow.gain);
    value["flow_period"] = json!(effect.flow.period);
    value["flow_rise"] = json!(effect.flow.rise);
    value["flow_strength"] = json!(effect.flow.strength);
    value["flow_core"] = json!(effect.flow.core);
    value["flow_gust"] = json!(effect.flow.gust);
    value["flow_gust_frequency"] = json!(effect.flow.gust_frequency);
    value["flow_burst"] = json!(effect.flow.burst);
    value["flow_damping"] = json!(effect.flow.damping);
    value["lobe_gain"] = json!(effect.lobe.gain);
    value["lobe_period"] = json!(effect.lobe.period);
    value["lobe_life"] = json!(effect.lobe.life);
    value["lobe_rise"] = json!(effect.lobe.rise);
    value["lobe_size"] = json!(effect.lobe.size);
    value["lobe_spawn_height"] = json!(effect.lobe.spawn_height);
    value["lobe_spread"] = json!(effect.lobe.spread);
    value["lobe_shift"] = json!(effect.lobe.shift);
    let strain = thyllore_effect_core::build_warp_strain_params(effect);
    value["warp_strain_params"] = json!([
        strain.strain_base,
        strain.strain_tip,
        strain.inv_reach,
        strain.inv_strain_norm,
    ]);
    value["warp_strain_cap"] = json!(thyllore_effect_core::flame_wave::WARP_STRAIN_CAP);
    value["warp_form"] = json!(if thyllore_effect_core::read_env_warp_form_displacement() {
        "disp"
    } else {
        "seq"
    });
    value["warp_strain_norm"] = json!(if strain.inv_strain_norm > 0.0 {
        1.0 / strain.inv_strain_norm
    } else {
        0.0
    });
    value["unified_field"] = json!({
        "active": thyllore_effect_core::read_env_wave_unified(),
        "window_beta": thyllore_effect_core::read_env_unified_beta(),
        "tilt_gain_b": thyllore_effect_core::read_env_unified_tilt_gain_b(),
        "tilt_gain_w": thyllore_effect_core::read_env_unified_tilt_gain_w(),
    });
    value["baked_blend"] = json!(baked.blend);
    value["baked_envelope"] = json!(baked.envelope.map(|a| a.to_vec()));
    value["baked_radius"] = json!(baked.radius.map(|a| a.to_vec()));
    value["baked_color"] = json!(baked
        .color
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
        "color_base": [ubo.color_base.rgb[0], ubo.color_base.rgb[1], ubo.color_base.rgb[2], ubo.color_base.occlusion_lum_ref],
        "color_mid": ubo.color_mid.rgb,
        "color_tip": ubo.color_tip.rgb,
        "light_data": [ubo.light_data.direction[0], ubo.light_data.direction[1], ubo.light_data.direction[2], ubo.light_data.self_shadow_strength],
        "unified_params": [ubo.unified_params.enabled, ubo.unified_params.sigma_floor],
        "mix_params": [ubo.mix_params.lo, ubo.mix_params.hi, ubo.mix_params.inv_carrier_std, ubo.mix_params.height_gain, ubo.mix_params.scale, ubo.mix_params.radial_gain, ubo.mix_params.core_radius],
        "segment_params": [ubo.segment_params.count],
        "thermal_params": [ubo.thermal_params.density_exp, ubo.thermal_params.temp_exp, ubo.thermal_params.temp_hot_k, ubo.thermal_params.temp_cold_k, ubo.thermal_params.wien_ck],
        "spread_params": [ubo.spread_params.gain, ubo.spread_params.edge_outer_sharpen, ubo.spread_params.twist_gain, ubo.spread_params.erosion_noise_gain],
        "support_margin": [
            ubo.support_motion.support_margin,
            ubo.support_motion.meander_amp,
            ubo.support_motion.swirl_speed,
            ubo.support_motion.twist_speed,
        ],
        "branch_field": build_branch_field_json(&ubo.branch_field),
    })
}

fn build_branch_field_json(field: &thyllore_effect_core::FlameBranchField) -> serde_json::Value {
    let count = (field.count as usize).min(thyllore_effect_core::BRANCH_MAX_ELEMENTS);
    json!({
        "count": field.count,
        "period": field.period,
        "life": field.life,
        "gain": field.gain,
        "rise_rate": field.rise_rate,
        "drift_rate": field.drift_rate,
        "aspect": field.aspect,
        "core_radius": field.core_radius,
        "core_offset": field.core_offset,
        "reach": [field.reach_start, field.reach_end],
        "envelope_time": field.envelope_time,
        "bounding_pad": [field.bounding_pad, field.bounding_pad_y],
        "elements": field.elements[..count]
            .iter()
            .map(|element| json!([
                element.spawn_time,
                element.side,
                element.azimuth,
                element.spawn_height,
                element.size,
                element.tilt,
                element.along_offset,
                element.hash01,
                element.trunk_radius,
            ]))
            .collect::<Vec<_>>(),
    })
}

pub fn build_temporal_json(state: &FlameTemporalState) -> serde_json::Value {
    let has_previous = state.previous.is_some();
    let previous = state.previous.as_ref().map(|snap| {
        json!({
            "view": matrix4_to_array(&snap.view),
            "appearance": build_effect_json(
                &snap.appearance,
                &snap.baked,
                &FlameTemporalAccum::default(),
            ),
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
    baked: &FlameBaked,
    temporal_accum: &FlameTemporalAccum,
    temporal: &FlameTemporalState,
    instance_index: usize,
    trail_enabled: bool,
    trail_len: usize,
    trail_fade_seconds: f32,
    trail_oldest_age: f32,
) -> Value {
    let ubo = build_flame_ubo(effect, baked, temporal_accum);
    let mut record = build_effect_json(effect, baked, temporal_accum)
        .as_object()
        .unwrap()
        .clone();
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
    flames: &[(FlameEffect, FlameBaked, FlameTemporalAccum)],
    trails: &[Option<crate::ecs::component::flame_trail::FlameTrail>],
) {
    for (i, (effect, baked, temporal_accum)) in flames.iter().enumerate() {
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
            baked,
            temporal_accum,
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

fn build_wall_probe_ray_json(ray: &thyllore_effect_core::WallProbeRay) -> Value {
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

pub fn build_wall_probe_json(report: &thyllore_effect_core::WallProbeReport) -> Value {
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

/// Declared field composition for the dump: which noise was active, driving what, gated by which parameter.
pub fn build_field_manifest_json(manifest: &thyllore_effect_core::FieldManifest) -> Value {
    json!({
        "summary": manifest.summary(),
        "active_sources": manifest
            .active_sources()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        "unification_pending": manifest
            .active_unification_pending()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        "influences": manifest.influences.iter().map(|i| json!({
            "source": i.source.as_str(),
            "target": i.target.as_str(),
            "parameter": i.parameter,
            "active": i.active,
        })).collect::<Vec<_>>(),
    })
}

pub fn build_wall_probe_dump_record(
    camera: &crate::ecs::resource::Camera,
    settings: &crate::ecs::resource::FlameRenderSettings,
    viewport_size: [f32; 2],
    flames: &[(
        FlameEffect,
        FlameBaked,
        FlameTemporalAccum,
        thyllore_effect_core::WallProbeReport,
    )],
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
        "flames": flames.iter().map(|(effect, baked, temporal, report)| {
            let mut entry = build_effect_json(effect, baked, temporal)
                .as_object()
                .unwrap()
                .clone();
            entry.insert("wall_probe".to_string(), build_wall_probe_json(report));
            entry.insert(
                "field_manifest".to_string(),
                build_field_manifest_json(&thyllore_effect_core::flame_field_manifest(effect)),
            );
            Value::Object(entry)
        }).collect::<Vec<_>>()
    })
}

pub fn write_flame_wall_probe_dump(
    camera: &crate::ecs::resource::Camera,
    settings: &crate::ecs::resource::FlameRenderSettings,
    viewport_size: [f32; 2],
    flames: &[(
        FlameEffect,
        FlameBaked,
        FlameTemporalAccum,
        thyllore_effect_core::WallProbeReport,
    )],
    output_path: Option<&std::path::Path>,
) -> std::io::Result<std::path::PathBuf> {
    let unix_time = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let record = build_wall_probe_dump_record(camera, settings, viewport_size, flames, unix_time);

    let path = if let Some(output_path) = output_path {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        output_path.to_path_buf()
    } else {
        let directory = std::path::Path::new("log/flame");
        std::fs::create_dir_all(directory)?;
        directory.join(format!("wall_probe_{}.json", unix_time))
    };
    std::fs::write(&path, serde_json::to_string_pretty(&record)?)?;
    Ok(path)
}

/// Full numerical replay of the analytic flame path (every node / segment
/// intermediate, from the packed FlameUBO the GPU receives) — one JSON per
/// flame, next to the wall probe dump. Grid density via
/// THYLLORE_FLAME_TRACE_COLS / THYLLORE_FLAME_TRACE_ROWS.
pub fn write_flame_field_traces(
    view: &WallProbeView,
    flames: &[(
        FlameEffect,
        FlameBaked,
        FlameTemporalAccum,
        thyllore_effect_core::WallProbeReport,
    )],
    output_path: Option<&std::path::Path>,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    let unix_time = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut paths = Vec::new();
    for (index, (effect, baked, temporal, _)) in flames.iter().enumerate() {
        let trace = thyllore_render_debug::flame_field_trace::trace_flame_field(
            effect, baked, temporal, view,
        );
        let path = if let Some(output_path) = output_path {
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            output_path.to_path_buf()
        } else {
            let directory = std::path::Path::new("log/flame");
            std::fs::create_dir_all(directory)?;
            let name = if flames.len() > 1 {
                format!("flame_trace_{}_{}.json", unix_time, index)
            } else {
                format!("flame_trace_{}.json", unix_time)
            };
            directory.join(name)
        };
        std::fs::write(&path, serde_json::to_string(&trace)?)?;
        paths.push(path);
    }
    Ok(paths)
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
    effect_before: (&FlameEffect, &FlameBaked),
    effect_after: (&FlameEffect, &FlameBaked),
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

    let temporal = FlameTemporalAccum::default();
    let before = build_effect_json(effect_before.0, effect_before.1, &temporal);
    let after = build_effect_json(effect_after.0, effect_after.1, &temporal);
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

    fn sample_effect() -> FlameEffect {
        use thyllore_effect_core::{
            FlameColor, FlameEdge, FlameEnvelope, FlameNoise, FlameWarp, FlameWind,
        };
        FlameEffect {
            position: Vector3::new(1.0, 2.0, 3.0),
            height: 1.0,
            radius: 0.5,
            sigma_t: 0.1,
            intensity: 1.0,
            color: FlameColor {
                base: [1.0, 0.0, 0.0],
                tip: [0.0, 1.0, 0.0],
                temperature_base_k: 1000.0,
                temperature_tip_k: 500.0,
                use_blackbody: false,
                occlusion_lum_ref: 1.0,
            },
            noise: FlameNoise {
                amplitude: 0.0,
                frequency: 0.0,
                scroll_speed: 0.0,
                ..FlameNoise::default()
            },
            time: 0.0,
            time_scale: 1.0,
            time_offset: 0.0,
            coefficients: thyllore_effect_core::fit_flame_coefficients(
                &thyllore_effect_core::FlameProfile::default(),
            ),
            light_position_world: Vector3::new(2.0, 3.0, 2.0),
            self_shadow_strength: 0.5,
            warp: FlameWarp {
                amp: 0.25,
                freq: 2.5,
                rise_speed: 0.8,
                taper_power: 1.4,
                ..FlameWarp::default()
            },
            edge: FlameEdge {
                radius_tip_ratio: 0.1,
                low: 0.3,
                high: 0.7,
                white_boost: 0.0,
                ..FlameEdge::default()
            },
            wind: FlameWind {
                direction: cgmath::Vector2::new(0.0, 0.0),
                bend_amount: 0.0,
                bend_power: 1.7,
            },
            envelope: FlameEnvelope {
                peak: 0.35,
                base: 0.45,
                tail: 1.6,
            },
            radial_sharpness: 4.0,
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
        let view = thyllore_effect_core::WallProbeView {
            position: [0.0, 1.15, 1.15],
            forward: [0.0, 0.0, -1.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y_radians: 45.0f32.to_radians(),
            viewport_size_px: [1680.0, 840.0],
        };
        let report = thyllore_effect_core::probe_flame_wall(&effect, &Default::default(), &view);
        let record = build_wall_probe_dump_record(
            &camera,
            &settings,
            [1680.0, 840.0],
            &[(effect, Default::default(), Default::default(), report)],
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
            thyllore_effect_core::WALL_PROBE_GRID_COLS * thyllore_effect_core::WALL_PROBE_GRID_ROWS
        );
    }

    #[test]
    fn field_manifest_json_names_sources_targets_and_parameters() {
        let mut effect = sample_effect();
        effect.noise.amplitude = 1.5;
        effect.boundary.amp = 0.2;
        let manifest = thyllore_effect_core::flame_field_manifest(&effect);
        let value = build_field_manifest_json(&manifest);
        assert!(value["summary"]
            .as_str()
            .unwrap()
            .contains("erosion-wave-table"));
        assert!(value["active_sources"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s == "erosion-wave-table"));
        let first = &value["influences"][0];
        assert!(first["source"].is_string());
        assert!(first["target"].is_string());
        assert!(first["parameter"].is_string());
        assert!(first["active"].is_boolean());
    }

    #[test]
    fn build_flame_dump_record_produces_valid_json() {
        let effect = sample_effect();
        let temporal = sample_temporal();
        let record = build_flame_dump_record(
            &effect,
            &Default::default(),
            &FlameTemporalAccum {
                weight: 0.5,
                frame_index: 42,
            },
            &temporal,
            0,
            false,
            0,
            0.8,
            0.0,
        );
        assert_eq!(record["frame_index"], 42);
        assert_eq!(record["time"], 0.0);
        assert_eq!(record["position"], json!([1.0, 2.0, 3.0]));
        assert_eq!(record["trail_enabled"], false);
        assert_eq!(record["trail_len"], 0);
        assert!((record["trail_fade_seconds"].as_f64().unwrap() - 0.8).abs() < 1e-5);
        assert_eq!(record["trail_oldest_age"], 0.0);
    }
}
