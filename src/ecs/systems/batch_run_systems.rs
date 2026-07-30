use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use cgmath::Vector2;

use crate::ecs::component::FlameEffect;
use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::{BatchRun, BatchRunState, FlameShadingMode};
use crate::ecs::world::World;

const BATCH_SCREENSHOT_FLAG: &str = "--batch-screenshot";
const BATCH_FRAMES_FLAG: &str = "--batch-frames";
const BATCH_FLAME_MODE_FLAG: &str = "--batch-flame-mode";
const BATCH_FLAME_STEPS_FLAG: &str = "--batch-flame-steps";
const BATCH_CAMERA_FLAG: &str = "--batch-camera";
const FLAME_DUMP_FLAG: &str = "--flame-dump";
const GPU_TIMINGS_FLAG: &str = "--gpu-timings";
const EXPOSURE_DUMP_FLAG: &str = "--exposure-dump";
const BATCH_FLAME_COUNT_FLAG: &str = "--batch-flame-count";
const BATCH_FLAME_TRAIL_FLAG: &str = "--batch-flame-trail";
const BATCH_FLAME_ORBIT_FLAG: &str = "--batch-flame-orbit";
const BATCH_FLAME_BONE_FLAG: &str = "--batch-flame-bone";
const BATCH_FLAME_PRESET_FLAG: &str = "--batch-flame-preset";
const BATCH_FLAME_MOTION_FLAG: &str = "--batch-flame-motion";
const BATCH_FLAME_SDF_FLAG: &str = "--batch-flame-sdf";
const BATCH_FLAME_SET_FLAG: &str = "--batch-flame-set";
const BATCH_PICK_FLAG: &str = "--batch-pick";
const DEFAULT_SCREENSHOT_FRAME: u64 = 120;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchCameraPose {
    pub yaw_degrees: f32,
    pub pitch_degrees: f32,
    pub distance: f32,
}

pub struct EngineCliOverrides {
    pub batch_run: Option<BatchRun>,
    pub flame_mode: Option<FlameShadingMode>,
    pub flame_steps: Option<u32>,
    pub camera_pose: Option<BatchCameraPose>,
    pub flame_dump_path: Option<String>,
    pub gpu_timings_path: Option<String>,
    pub exposure_dump_path: Option<String>,
    pub flame_count: Option<usize>,
    pub flame_preset: Option<String>,
    pub flame_set: Vec<(String, f32)>,
    pub flame_trail: Option<f32>,
    pub flame_orbit: Option<(f32, f32)>,
    pub flame_motion: Option<(f32, f32)>,
    pub flame_bone: Option<String>,
    pub flame_sdf: Option<String>,
    pub pick_pixel: Option<(u32, u32)>,
    pub batch_play: bool,
}

pub fn resolve_engine_cli_overrides(args: &[String]) -> Result<EngineCliOverrides> {
    Ok(EngineCliOverrides {
        batch_run: batch_run_resolve_from_args(args)?,
        flame_mode: flame_mode_resolve_from_args(args)?,
        flame_steps: flame_steps_resolve_from_args(args)?,
        camera_pose: camera_pose_resolve_from_args(args)?,
        flame_dump_path: flame_dump_path_resolve_from_args(args)?,
        gpu_timings_path: gpu_timings_path_resolve_from_args(args)?,
        exposure_dump_path: exposure_dump_path_resolve_from_args(args)?,
        flame_count: flame_count_resolve_from_args(args)?,
        flame_preset: flame_preset_resolve_from_args(args)?,
        flame_set: flame_set_resolve_from_args(args)?,
        flame_trail: flame_trail_resolve_from_args(args)?,
        flame_orbit: flame_orbit_resolve_from_args(args)?,
        flame_motion: flame_motion_resolve_from_args(args)?,
        flame_bone: flame_bone_resolve_from_args(args)?,
        pick_pixel: pick_pixel_resolve_from_args(args)?,
        flame_sdf: flame_sdf_resolve_from_args(args)?,
        batch_play: args.iter().any(|a| a == "--batch-play"),
    })
}

pub fn camera_pose_resolve_from_args(args: &[String]) -> Result<Option<BatchCameraPose>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_CAMERA_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_CAMERA_FLAG} requires <yaw_deg>,<pitch_deg>,<distance>");
    };

    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 3 {
        bail!("{BATCH_CAMERA_FLAG} expects 3 comma-separated values, got '{value}'");
    }
    let numbers: Vec<f32> = parts
        .iter()
        .map(|part| part.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .map_err(|_| anyhow::anyhow!("invalid {BATCH_CAMERA_FLAG} value '{value}'"))?;
    if !numbers.iter().all(|n| n.is_finite()) || numbers[2] <= 0.0 {
        bail!("{BATCH_CAMERA_FLAG} distance must be > 0 and all values finite: '{value}'");
    }

    Ok(Some(BatchCameraPose {
        yaw_degrees: numbers[0],
        pitch_degrees: numbers[1],
        distance: numbers[2],
    }))
}

pub fn batch_run_resolve_from_args(args: &[String]) -> Result<Option<BatchRun>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_SCREENSHOT_FLAG) else {
        if args.iter().any(|arg| arg == BATCH_FRAMES_FLAG) {
            bail!("{BATCH_FRAMES_FLAG} requires {BATCH_SCREENSHOT_FLAG} <output.png>");
        }
        return Ok(None);
    };

    let Some(output) = args.get(position + 1).filter(|v| !v.starts_with("--")) else {
        bail!("{BATCH_SCREENSHOT_FLAG} requires an output path");
    };
    let output = resolve_absolute_output(Path::new(output))?;

    let screenshot_frame = match args.iter().position(|arg| arg == BATCH_FRAMES_FLAG) {
        Some(frames_position) => {
            let Some(value) = args.get(frames_position + 1) else {
                bail!("{BATCH_FRAMES_FLAG} requires a frame count");
            };
            let frames: u64 = value
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid frame count '{value}': expected integer"))?;
            if frames == 0 {
                bail!("{BATCH_FRAMES_FLAG} must be >= 1");
            }
            frames
        }
        None => DEFAULT_SCREENSHOT_FRAME,
    };

    let flame_set = flame_set_resolve_from_args(args)?;

    Ok(Some(BatchRun::new(output, screenshot_frame, flame_set)))
}

pub fn flame_mode_resolve_from_args(args: &[String]) -> Result<Option<FlameShadingMode>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_FLAME_MODE_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_FLAME_MODE_FLAG} requires a value: analytic|raymarch|thickness|noise|depthclamp");
    };
    let mode = FlameShadingMode::parse(value).ok_or_else(|| {
        anyhow::anyhow!(
            "invalid flame mode '{value}': expected analytic|raymarch|thickness|noise|depthclamp"
        )
    })?;
    Ok(Some(mode))
}

pub fn flame_steps_resolve_from_args(args: &[String]) -> Result<Option<u32>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_FLAME_STEPS_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_FLAME_STEPS_FLAG} requires a step count");
    };
    let steps: u32 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid step count '{value}': expected integer"))?;
    if steps == 0 {
        bail!("{BATCH_FLAME_STEPS_FLAG} must be >= 1");
    }
    Ok(Some(steps))
}

pub fn flame_dump_path_resolve_from_args(args: &[String]) -> Result<Option<String>> {
    let Some(position) = args.iter().position(|arg| arg == FLAME_DUMP_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{FLAME_DUMP_FLAG} requires a path");
    };
    Ok(Some(value.clone()))
}

pub fn gpu_timings_path_resolve_from_args(args: &[String]) -> Result<Option<String>> {
    let Some(position) = args.iter().position(|arg| arg == GPU_TIMINGS_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{GPU_TIMINGS_FLAG} requires a path");
    };
    Ok(Some(value.clone()))
}

pub fn exposure_dump_path_resolve_from_args(args: &[String]) -> Result<Option<String>> {
    let Some(position) = args.iter().position(|arg| arg == EXPOSURE_DUMP_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{EXPOSURE_DUMP_FLAG} requires a path");
    };
    Ok(Some(value.clone()))
}

pub fn flame_count_resolve_from_args(args: &[String]) -> Result<Option<usize>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_FLAME_COUNT_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_FLAME_COUNT_FLAG} requires a count");
    };
    let count: usize = value
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid flame count '{value}': expected integer"))?;
    if !(1..=4).contains(&count) {
        bail!(
            "{BATCH_FLAME_COUNT_FLAG} must be in range 1..=4, got {}",
            count
        );
    }
    Ok(Some(count))
}
pub(crate) const FLAME_SET_KEYS: &[&str] = &[
    "warp_amp",
    "warp_freq",
    "rise_speed",
    "taper_power",
    "radius_tip_ratio",
    "edge_low",
    "edge_high",
    "white_boost",
    "bend_amount",
    "bend_power",
    "wind_x",
    "wind_z",
    "noise_amplitude",
    "noise_frequency",
    "noise_scroll_speed",
    "sigma_t",
    "intensity",
    "height",
    "radius",
    "time",
    "time_scale",
    "time_offset",
    "rot_z_deg",
    "temperature_base_k",
    "temperature_tip_k",
    "envelope_peak",
    "envelope_base",
    "envelope_tail",
    "radial_sharpness",
    "emitter_kind",
    "ring_major_radius",
    "ring_angular_speed",
    "noise_aniso_y",
    "warp_y_scale",
    "occlusion_lum_ref",
    "contour_wiggle_amp",
];

fn flame_set_resolve_from_args(args: &[String]) -> Result<Vec<(String, f32)>> {
    let valid_keys = FLAME_SET_KEYS;

    let mut pairs: Vec<(String, f32)> = Vec::new();
    for i in 0..args.len() {
        let payload = if args[i] == BATCH_FLAME_SET_FLAG {
            if i + 1 >= args.len() {
                anyhow::bail!("{} requires a value after it", BATCH_FLAME_SET_FLAG);
            }
            args[i + 1].clone()
        } else if let Some(rest) = args[i].strip_prefix(BATCH_FLAME_SET_FLAG) {
            rest.trim_start_matches('=').trim().to_string()
        } else {
            continue;
        };

        let parts: Vec<&str> = payload.splitn(2, '=').collect();
        if parts.len() != 2 {
            anyhow::bail!(
                "batch-flame-set value must be KEY=VALUE format, got '{}'",
                payload
            );
        }
        let key = parts[0].trim().to_string();
        let value_str = parts[1].trim();
        let value: f32 = value_str.parse().context(format!(
            "batch-flame-set value must be a number, got '{}'",
            value_str
        ))?;

        if !valid_keys.contains(&key.as_str()) {
            anyhow::bail!(
                "unknown batch-flame-set key '{}'. Valid keys: {}",
                key,
                valid_keys.join(", ")
            );
        }

        pairs.push((key, value));
    }
    Ok(pairs)
}

fn flame_preset_resolve_from_args(args: &[String]) -> Result<Option<String>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_FLAME_PRESET_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_FLAME_PRESET_FLAG} requires <name>");
    };
    if !thyllore_render_core::FLAME_PRESET_NAMES.contains(&value.as_str()) {
        bail!(
            "unknown flame preset '{}'. Valid presets: {}",
            value,
            thyllore_render_core::FLAME_PRESET_NAMES.join(", ")
        );
    }
    Ok(Some(value.clone()))
}

fn flame_trail_resolve_from_args(args: &[String]) -> Result<Option<f32>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_FLAME_TRAIL_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_FLAME_TRAIL_FLAG} requires <fade_seconds>");
    };
    let fade: f32 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid {BATCH_FLAME_TRAIL_FLAG} value '{value}'"))?;
    if !fade.is_finite() || fade <= 0.0 {
        bail!("{BATCH_FLAME_TRAIL_FLAG} fade_seconds must be > 0 and finite: '{value}'");
    }
    Ok(Some(fade))
}

fn flame_orbit_resolve_from_args(args: &[String]) -> Result<Option<(f32, f32)>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_FLAME_ORBIT_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_FLAME_ORBIT_FLAG} requires <radius>,<period_seconds>");
    };
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 2 {
        bail!("{BATCH_FLAME_ORBIT_FLAG} expects 2 comma-separated values, got '{value}'");
    }
    let radius: f32 = parts[0]
        .trim()
        .parse::<f32>()
        .map_err(|_| anyhow::anyhow!("invalid {BATCH_FLAME_ORBIT_FLAG} radius in '{value}'"))?;
    let period: f32 = parts[1]
        .trim()
        .parse::<f32>()
        .map_err(|_| anyhow::anyhow!("invalid {BATCH_FLAME_ORBIT_FLAG} period in '{value}'"))?;
    if !radius.is_finite() || radius < 0.0 || !period.is_finite() || period <= 0.0 {
        bail!("{BATCH_FLAME_ORBIT_FLAG} radius must be >= 0 and period > 0, all finite: '{value}'");
    }
    Ok(Some((radius, period)))
}

/// Drives one viewport click from the command line so the picking readback path can be exercised
/// headlessly, where there is no mouse to press.
fn pick_pixel_resolve_from_args(args: &[String]) -> Result<Option<(u32, u32)>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_PICK_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_PICK_FLAG} requires <x>,<y>");
    };
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 2 {
        bail!("{BATCH_PICK_FLAG} expects 2 comma-separated values, got '{value}'");
    }
    let x: u32 = parts[0]
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid {BATCH_PICK_FLAG} x in '{value}'"))?;
    let y: u32 = parts[1]
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid {BATCH_PICK_FLAG} y in '{value}'"))?;
    Ok(Some((x, y)))
}

fn flame_motion_resolve_from_args(args: &[String]) -> Result<Option<(f32, f32)>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_FLAME_MOTION_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_FLAME_MOTION_FLAG} requires <radius>,<angular_speed>");
    };
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 2 {
        bail!("{BATCH_FLAME_MOTION_FLAG} expects 2 comma-separated values, got '{value}'");
    }
    let radius: f32 = parts[0]
        .trim()
        .parse::<f32>()
        .map_err(|_| anyhow::anyhow!("invalid {BATCH_FLAME_MOTION_FLAG} radius in '{value}'"))?;
    let angular_speed: f32 = parts[1].trim().parse::<f32>().map_err(|_| {
        anyhow::anyhow!("invalid {BATCH_FLAME_MOTION_FLAG} angular_speed in '{value}'")
    })?;
    if !radius.is_finite() || radius < 0.0 || !angular_speed.is_finite() {
        bail!("{BATCH_FLAME_MOTION_FLAG} radius must be >= 0 and angular_speed finite: '{value}'");
    }
    Ok(Some((radius, angular_speed)))
}

fn flame_bone_resolve_from_args(args: &[String]) -> Result<Option<String>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_FLAME_BONE_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_FLAME_BONE_FLAG} requires <name-or-index>");
    };
    if value.starts_with("--") {
        bail!("{BATCH_FLAME_BONE_FLAG} requires <name-or-index>");
    }
    Ok(Some(value.clone()))
}

fn flame_sdf_resolve_from_args(args: &[String]) -> Result<Option<String>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_FLAME_SDF_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_FLAME_SDF_FLAG} requires <path>");
    };
    if value.starts_with("--") {
        bail!("{BATCH_FLAME_SDF_FLAG} requires <path>");
    }
    Ok(Some(value.clone()))
}
/// Compute the XZ circular orbit offset at time t for a given radius and period.
/// position = (R * cos(2*pi*t/T), 0, R * sin(2*pi*t/T))
/// If period <= 0, returns [0, 0, 0].
pub fn compute_orbit_offset(radius: f32, period_seconds: f32, t_seconds: f32) -> [f32; 3] {
    if period_seconds <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let angle = 2.0 * std::f32::consts::PI * t_seconds / period_seconds;
    [radius * angle.cos(), 0.0, radius * angle.sin()]
}

/// Update flame entity Transform positions based on BatchFlameOrbit resource.
/// On first run (initial is None), determines the initial position from the first flame's
/// Transform.translation (or (0,0,0) if missing) and inserts MotionPath components for all
/// flame entities. Subsequent calls are no-ops — position updates are handled by sync_motion_paths.
pub fn batch_run_update_orbit(world: &mut World) {
    // Extract orbit params and initial from resource, then drop the borrow before mutating components
    let (radius, period_seconds, initial) = {
        let mut orbit = match world.get_resource_mut::<crate::ecs::resource::BatchFlameOrbit>() {
            Some(o) => o,
            None => return,
        };
        let radius = orbit.radius;
        let period_seconds = orbit.period_seconds;
        // Only act when initial is None (first run)
        if orbit.initial.is_some() {
            return;
        }
        let flame_entities: Vec<_> = world.query_flames();
        let pos = if flame_entities.is_empty() {
            cgmath::Vector3::new(0.0, 0.0, 0.0)
        } else {
            let first = flame_entities[0];
            world
                .get_component::<crate::ecs::world::Transform>(first)
                .map(|t| t.translation)
                .unwrap_or(cgmath::Vector3::new(0.0, 0.0, 0.0))
        };
        orbit.initial = Some(pos);
        (radius, period_seconds, pos)
    };

    // Guard: period_seconds <= 0 means no motion (degenerate orbit), so insert nothing
    if period_seconds <= 0.0 {
        return;
    }

    let flame_entities: Vec<_> = world.query_flames();
    for &e in &flame_entities {
        let path = crate::ecs::component::MotionPath {
            center: initial,
            radius,
            angular_speed: 2.0 * std::f32::consts::PI / period_seconds,
            phase_offset: 0.0,
            enabled: true,
        };
        world.insert_component(e, path);
    }
}

pub fn apply_flame_overrides(effect: &mut FlameEffect, overrides: &[(String, f32)]) {
    for (key, value) in overrides {
        match key.as_str() {
            "warp_amp" => effect.warp_amp = *value,
            "warp_freq" => effect.warp_freq = *value,
            "rise_speed" => effect.rise_speed = *value,
            "taper_power" => effect.taper_power = *value,
            "radius_tip_ratio" => effect.radius_tip_ratio = *value,
            "edge_low" => effect.edge_low = *value,
            "edge_high" => effect.edge_high = *value,
            "white_boost" => effect.white_boost = *value,
            "bend_amount" => effect.bend_amount = *value,
            "bend_power" => effect.bend_power = *value,
            "wind_x" => effect.wind_direction.x = *value,
            "wind_z" => effect.wind_direction.y = *value,
            "noise_amplitude" => effect.noise_amplitude = *value,
            "noise_frequency" => effect.noise_frequency = *value,
            "noise_scroll_speed" => effect.noise_scroll_speed = *value,
            "noise_aniso_y" => effect.noise_aniso_y = *value,
            "warp_y_scale" => effect.warp_y_scale = *value,
            "sigma_t" => effect.sigma_t = *value,
            "intensity" => effect.intensity = *value,
            "height" => effect.height = *value,
            "radius" => effect.radius = *value,
            "time" => effect.time = *value,
            "time_scale" => effect.time_scale = *value,
            "time_offset" => effect.time_offset = *value,
            "temperature_base_k" => effect.temperature_base_k = *value,
            "temperature_tip_k" => effect.temperature_tip_k = *value,
            "envelope_peak" => effect.envelope_peak = *value,
            "envelope_base" => effect.envelope_base = *value,
            "envelope_tail" => effect.envelope_tail = *value,
            "radial_sharpness" => effect.radial_sharpness = *value,
            "rot_z_deg" => {
                effect.rotation = cgmath::Quaternion::from(cgmath::Euler::new(
                    cgmath::Deg(0.0),
                    cgmath::Deg(0.0),
                    cgmath::Deg(*value),
                ))
            }
            "emitter_kind" => effect.emitter_kind = *value as u32,
            "ring_major_radius" => effect.ring_major_radius = *value,
            "ring_angular_speed" => effect.ring_angular_speed = *value,
            "occlusion_lum_ref" => effect.occlusion_lum_ref = *value,
            "contour_wiggle_amp" => effect.contour_wiggle_amp = *value,
            _ => unreachable!("unknown key (parser should have rejected)"),
        }
    }
}

fn resolve_absolute_output(output: &Path) -> Result<PathBuf> {
    if output.extension().and_then(|e| e.to_str()) != Some("png") {
        bail!(
            "batch screenshot output must end with .png: {}",
            output.display()
        );
    }
    if output.is_absolute() {
        return Ok(output.to_path_buf());
    }
    Ok(std::env::current_dir()?.join(output))
}

pub fn batch_run_tick(world: &World) {
    if !world.contains_resource::<BatchRun>() {
        return;
    }

    let should_request = {
        let mut batch = world.resource_mut::<BatchRun>();
        batch.frames_rendered += 1;
        matches!(batch.state, BatchRunState::WaitingForFrame)
            && batch.frames_rendered >= batch.screenshot_frame
    };

    if should_request {
        world
            .resource_mut::<UIEventQueue>()
            .send(UIEvent::TakeScreenshot);
        world.resource_mut::<BatchRun>().state = BatchRunState::ScreenshotRequested;
    }
}

pub fn batch_run_record_screenshot(world: &World, save_result: Result<String, String>) {
    let Some(mut batch) = world.get_resource_mut::<BatchRun>() else {
        return;
    };
    if !matches!(batch.state, BatchRunState::ScreenshotRequested) {
        return;
    }

    let result = save_result.and_then(|saved| move_screenshot_to_output(&saved, &batch.output));
    batch.state = BatchRunState::Completed { result };
}

fn move_screenshot_to_output(saved: &str, output: &Path) -> Result<String, String> {
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    std::fs::copy(saved, output)
        .map_err(|e| format!("failed to copy {saved} to {}: {e}", output.display()))?;
    if let Err(e) = std::fs::remove_file(saved) {
        log_warn!("failed to remove intermediate screenshot {saved}: {e}");
    }
    Ok(output.to_string_lossy().to_string())
}

pub fn batch_run_is_completed(world: &World) -> bool {
    world
        .get_resource::<BatchRun>()
        .map(|batch| batch.is_completed())
        .unwrap_or(false)
}

pub fn batch_run_report(batch: &BatchRun) -> (bool, String) {
    match &batch.state {
        BatchRunState::Completed { result: Ok(path) } => {
            (true, serde_json::json!({"ok": true, "path": path}).to_string())
        }
        BatchRunState::Completed { result: Err(error) } => (
            false,
            serde_json::json!({"ok": false, "error": error}).to_string(),
        ),
        BatchRunState::WaitingForFrame | BatchRunState::ScreenshotRequested => (
            false,
            serde_json::json!({"ok": false, "error": "batch run ended before screenshot completed"})
                .to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn pick_pixel_is_absent_without_the_flag() {
        assert_eq!(pick_pixel_resolve_from_args(&args(&["bin"])).unwrap(), None);
    }

    #[test]
    fn pick_pixel_parses_a_pixel_pair() {
        assert_eq!(
            pick_pixel_resolve_from_args(&args(&["bin", "--batch-pick", "947,150"])).unwrap(),
            Some((947, 150))
        );
    }

    #[test]
    fn pick_pixel_rejects_a_malformed_pair() {
        assert!(pick_pixel_resolve_from_args(&args(&["bin", "--batch-pick", "947"])).is_err());
        assert!(pick_pixel_resolve_from_args(&args(&["bin", "--batch-pick", "a,b"])).is_err());
    }

    #[test]
    fn resolve_returns_none_without_flag() {
        let resolved = batch_run_resolve_from_args(&args(&["thyllore-animation"])).unwrap();
        assert!(resolved.is_none());
    }

    #[test]
    fn resolve_parses_output_and_default_frames() {
        let resolved =
            batch_run_resolve_from_args(&args(&["bin", "--batch-screenshot", "/tmp/out.png"]))
                .unwrap()
                .unwrap();
        assert_eq!(resolved.output, PathBuf::from("/tmp/out.png"));
        assert_eq!(resolved.screenshot_frame, DEFAULT_SCREENSHOT_FRAME);
        assert!(matches!(resolved.state, BatchRunState::WaitingForFrame));
    }

    #[test]
    fn resolve_parses_explicit_frames() {
        let resolved = batch_run_resolve_from_args(&args(&[
            "bin",
            "--batch-screenshot",
            "/tmp/out.png",
            "--batch-frames",
            "30",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(resolved.screenshot_frame, 30);
    }

    #[test]
    fn resolve_rejects_missing_output() {
        assert!(batch_run_resolve_from_args(&args(&["bin", "--batch-screenshot"])).is_err());
        assert!(batch_run_resolve_from_args(&args(&[
            "bin",
            "--batch-screenshot",
            "--batch-frames"
        ]))
        .is_err());
    }

    #[test]
    fn resolve_rejects_non_png_output() {
        assert!(
            batch_run_resolve_from_args(&args(&["bin", "--batch-screenshot", "/tmp/out.jpg"]))
                .is_err()
        );
    }

    #[test]
    fn resolve_rejects_invalid_frames() {
        assert!(batch_run_resolve_from_args(&args(&[
            "bin",
            "--batch-screenshot",
            "/tmp/out.png",
            "--batch-frames",
            "0"
        ]))
        .is_err());
        assert!(batch_run_resolve_from_args(&args(&[
            "bin",
            "--batch-screenshot",
            "/tmp/out.png",
            "--batch-frames",
            "abc"
        ]))
        .is_err());
    }

    #[test]
    fn resolve_rejects_frames_without_screenshot() {
        assert!(batch_run_resolve_from_args(&args(&["bin", "--batch-frames", "30"])).is_err());
    }

    #[test]
    fn resolve_flame_mode_and_steps() {
        let overrides = resolve_engine_cli_overrides(&args(&[
            "bin",
            "--batch-flame-mode",
            "raymarch",
            "--batch-flame-steps",
            "512",
        ]))
        .unwrap();
        assert!(overrides.batch_run.is_none());
        assert_eq!(
            overrides.flame_mode,
            Some(FlameShadingMode::ReferenceRaymarch)
        );
        assert_eq!(overrides.flame_steps, Some(512));
    }

    #[test]
    fn resolve_rejects_invalid_flame_overrides() {
        assert!(flame_mode_resolve_from_args(&args(&["bin", "--batch-flame-mode", "x"])).is_err());
        assert!(
            flame_steps_resolve_from_args(&args(&["bin", "--batch-flame-steps", "0"])).is_err()
        );
        assert!(
            flame_steps_resolve_from_args(&args(&["bin", "--batch-flame-steps", "abc"])).is_err()
        );
    }

    #[test]
    fn resolve_camera_pose() {
        let pose = camera_pose_resolve_from_args(&args(&["bin", "--batch-camera", "30,5,4"]))
            .unwrap()
            .unwrap();
        assert_eq!(
            pose,
            BatchCameraPose {
                yaw_degrees: 30.0,
                pitch_degrees: 5.0,
                distance: 4.0
            }
        );
        assert!(camera_pose_resolve_from_args(&args(&["bin"]))
            .unwrap()
            .is_none());
    }

    #[test]
    fn resolve_rejects_invalid_camera_pose() {
        for value in ["30,5", "a,b,c", "30,5,0", "30,5,-1"] {
            assert!(
                camera_pose_resolve_from_args(&args(&["bin", "--batch-camera", value])).is_err(),
                "expected error for '{value}'"
            );
        }
    }

    #[test]
    fn tick_requests_screenshot_at_target_frame() {
        let mut world = World::new();
        world.insert_resource(UIEventQueue::default());
        world.insert_resource(BatchRun::new(PathBuf::from("/tmp/out.png"), 2, Vec::new()));

        batch_run_tick(&world);
        assert!(matches!(
            world.resource::<BatchRun>().state,
            BatchRunState::WaitingForFrame
        ));

        batch_run_tick(&world);
        assert!(matches!(
            world.resource::<BatchRun>().state,
            BatchRunState::ScreenshotRequested
        ));
    }

    #[test]
    fn record_ignores_keyboard_screenshot_while_waiting() {
        let world = {
            let mut world = World::new();
            world.insert_resource(BatchRun::new(
                PathBuf::from("/tmp/out.png"),
                100,
                Vec::new(),
            ));
            world
        };

        batch_run_record_screenshot(&world, Ok("log/screenshot_1.png".to_string()));
        assert!(matches!(
            world.resource::<BatchRun>().state,
            BatchRunState::WaitingForFrame
        ));
    }

    #[test]
    fn record_stores_error_result() {
        let mut world = World::new();
        world.insert_resource(BatchRun::new(PathBuf::from("/tmp/out.png"), 1, Vec::new()));
        world.resource_mut::<BatchRun>().state = BatchRunState::ScreenshotRequested;

        batch_run_record_screenshot(&world, Err("save failed".to_string()));

        let batch = world.resource::<BatchRun>();
        assert!(batch.is_completed());
        let (ok, line) = batch_run_report(&batch);
        assert!(!ok);
        assert!(line.contains("save failed"));
    }

    #[test]
    fn report_incomplete_state_is_error() {
        let batch = BatchRun::new(PathBuf::from("/tmp/out.png"), 1, Vec::new());
        let (ok, line) = batch_run_report(&batch);
        assert!(!ok);
        assert!(line.contains("before screenshot completed"));
    }

    #[test]
    fn flame_set_combined_form() {
        let args: Vec<String> = vec!["--batch-flame-set=noise_amplitude=0.35".into()];
        let pairs = flame_set_resolve_from_args(&args).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "noise_amplitude");
        assert!((pairs[0].1 - 0.35).abs() < 1e-6);
    }

    #[test]
    fn flame_set_separate_form() {
        let args: Vec<String> = vec!["--batch-flame-set".into(), "noise_amplitude=0.35".into()];
        let pairs = flame_set_resolve_from_args(&args).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "noise_amplitude");
        assert!((pairs[0].1 - 0.35).abs() < 1e-6);
    }

    #[test]
    fn flame_set_unknown_key_error() {
        let args: Vec<String> = vec!["--batch-flame-set".into(), "invalid_key=1.0".into()];
        let err = flame_set_resolve_from_args(&args).unwrap_err();
        assert!(err.to_string().contains("invalid_key"),);
    }

    #[test]
    fn apply_flame_overrides_no_panic_for_all_keys() {
        for &key in FLAME_SET_KEYS {
            let mut effect = FlameEffect::default();
            let overrides: Vec<(String, f32)> = vec![(key.to_string(), 1.0)];
            apply_flame_overrides(&mut effect, &overrides);
        }
    }

    #[test]
    fn batch_run_update_orbit_inserts_missing_transform() {
        let mut world = World::new();

        // Spawn an entity with only FlameEffect (no Transform)
        let e = world.spawn();
        world.insert_component(e, FlameEffect::default());

        // Insert BatchRun and BatchFlameOrbit resources
        world.insert_resource(BatchRun::new(PathBuf::from("/tmp/out.png"), 1, Vec::new()));
        world.resource_mut::<BatchRun>().frames_rendered = 1;
        world.insert_resource(crate::ecs::resource::BatchFlameOrbit {
            radius: 2.0,
            period_seconds: 4.0,
            initial: None,
        });

        // Call once: initializes `initial` from the (missing) Transform to (0,0,0)
        // and inserts MotionPath component for the flame entity
        batch_run_update_orbit(&mut world);

        // Assert that the entity now has a MotionPath component
        let motion_path = world.get_component::<crate::ecs::component::MotionPath>(e);
        assert!(
            motion_path.is_some(),
            "MotionPath should have been inserted"
        );

        let motion_path = motion_path.unwrap();
        assert_eq!(motion_path.center, cgmath::Vector3::new(0.0, 0.0, 0.0));
        assert!((motion_path.radius - 2.0).abs() < 1e-5);
        assert!(
            (motion_path.angular_speed - 2.0 * std::f32::consts::PI / 4.0).abs() < 1e-5,
            "angular_speed: got {}, expected {}",
            motion_path.angular_speed,
            2.0 * std::f32::consts::PI / 4.0
        );

        // Call sync_motion_paths to update Transform from MotionPath
        crate::ecs::systems::sync_motion_paths(&mut world);

        // Assert that the entity now has a Transform component (inserted by sync_motion_paths)
        let transform = world.get_component::<crate::ecs::world::Transform>(e);
        assert!(
            transform.is_some(),
            "Transform should have been inserted by sync_motion_paths"
        );

        let transform = transform.unwrap();
        let offset = compute_orbit_offset(2.0, 4.0, 1.0 / 60.0);

        assert!(
            (transform.translation.x - offset[0]).abs() < 1e-5,
            "translation.x: got {}, expected {}",
            transform.translation.x,
            offset[0]
        );
        assert!(
            (transform.translation.z - offset[2]).abs() < 1e-5,
            "translation.z: got {}, expected {}",
            transform.translation.z,
            offset[2]
        );
    }

    #[test]
    fn test_flame_preset_resolve_valid() {
        let args = vec![String::from("--batch-flame-preset"), String::from("candle")];
        let result = flame_preset_resolve_from_args(&args).unwrap();
        assert_eq!(result, Some(String::from("candle")));
    }

    #[test]
    fn test_flame_preset_then_override_order() {
        // "candle" preset sets height=0.28, radius=0.07, intensity=2.0, etc.
        let mut effect = FlameEffect::default();
        thyllore_render_core::apply_flame_preset(&mut effect, "candle");

        // Now apply an individual override for height via flame_set
        let overrides: Vec<(String, f32)> = vec![(String::from("height"), 1.5)];
        apply_flame_overrides(&mut effect, &overrides);

        // The override should be final (1.5), not the preset value (0.28)
        assert!(
            (effect.height - 1.5).abs() < 1e-5,
            "height should be overridden to 1.5, got {}",
            effect.height
        );
        // Other candle preset values should remain
        assert!(
            (effect.radius - 0.07).abs() < 1e-5,
            "radius should still be candle's 0.07, got {}",
            effect.radius
        );
    }

    #[test]
    fn test_orbit_motion_path_equivalence() {
        use crate::ecs::component::{motion_path_position, MotionPath};
        use std::f32::consts::PI;

        let center = cgmath::Vector3::new(1.0, 2.0, 3.0);
        let radius = 1.5;
        let period = 2.0;
        let path = MotionPath {
            center,
            radius,
            angular_speed: 2.0 * PI / period,
            phase_offset: 0.0,
            enabled: true,
        };

        for &t in &[0.0, 0.7, 1.9, 3.3] {
            let mp_pos = motion_path_position(&path, t);
            let offset = compute_orbit_offset(radius, period, t);
            let orbit_pos = cgmath::Vector3::new(
                center.x + offset[0],
                center.y + offset[1],
                center.z + offset[2],
            );

            assert!(
                (mp_pos.x - orbit_pos.x).abs() < 1e-5,
                "t={}: x diff {} (mp={}, orbit={})",
                t,
                (mp_pos.x - orbit_pos.x).abs(),
                mp_pos.x,
                orbit_pos.x
            );
            assert!(
                (mp_pos.y - orbit_pos.y).abs() < 1e-5,
                "t={}: y diff {} (mp={}, orbit={})",
                t,
                (mp_pos.y - orbit_pos.y).abs(),
                mp_pos.y,
                orbit_pos.y
            );
            assert!(
                (mp_pos.z - orbit_pos.z).abs() < 1e-5,
                "t={}: z diff {} (mp={}, orbit={})",
                t,
                (mp_pos.z - orbit_pos.z).abs(),
                mp_pos.z,
                orbit_pos.z
            );
        }
    }
}
