use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use serde_json::json;

use crate::ecs::resource::{BatchRun, BatchRunState, FlameShadingMode};
use crate::ecs::world::World;

#[cfg(test)]
use crate::asset::AssetStorage;
#[cfg(test)]
use crate::ecs::component::{ClipSchedule, FlameEffect};
#[cfg(test)]
use crate::ecs::events::{UIEvent, UIEventQueue};
#[cfg(test)]
use crate::ecs::resource::{ClipLibrary, DebugViewState, TimelineState};

mod anim_edits;
mod debug_actions;
mod flame_args;
mod sequence_analyze;
mod water_args;

pub use anim_edits::*;
pub use debug_actions::*;
#[cfg(test)]
use flame_args::flame_set_valid_keys;
pub use flame_args::{
    apply_flame_overrides, apply_flame_style_from_path, apply_texture_fit_from_path,
    batch_run_flame_dump, dump_flame_style_to_path, flame_count_resolve_from_args,
    flame_debug_view_resolve_from_args, flame_dump_npy_path, flame_dump_path_resolve_from_args,
    flame_mode_resolve_from_args, flame_steps_resolve_from_args, load_flame_style_from_path,
};
use flame_args::{
    flame_bone_resolve_from_args, flame_motion_resolve_from_args, flame_orbit_resolve_from_args,
    flame_preset_resolve_from_args, flame_sdf_resolve_from_args, flame_set_resolve_from_args,
    flame_style_resolve_from_args, flame_texture_fit_resolve_from_args,
    flame_trail_resolve_from_args, heat_plume_resolve_from_args, parse_texture_fit_args,
};
pub use sequence_analyze::*;
pub use water_args::*;

const BATCH_SCREENSHOT_FLAG: &str = "--batch-screenshot";
const BATCH_SCREENSHOT_SEQUENCE_FLAG: &str = "--batch-screenshot-sequence";
const BATCH_FRAMES_FLAG: &str = "--batch-frames";
const BATCH_FLAME_MODE_FLAG: &str = "--batch-flame-mode";
const BATCH_FLAME_DEBUG_VIEW_FLAG: &str = "--batch-flame-debug-view";
const BATCH_WATER_DEBUG_VIEW_FLAG: &str = "--batch-water-debug-view";
const BATCH_WATER_SECONDARY_FLAG: &str = "--batch-water-secondary";
const BATCH_WATER_CAUSTIC_DEBUG_FLAG: &str = "--batch-water-caustic-debug";
const BATCH_WATER_HISTORY_FLAG: &str = "--batch-water-history";
const BATCH_WATER_TIME_FLAG: &str = "--batch-water-time";
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
const BATCH_FLAME_STYLE_FLAG: &str = "--batch-flame-style";
const BATCH_FLAME_STYLE_DUMP_FLAG: &str = "--batch-flame-style-dump";
const BATCH_FLAME_TEXTURE_FLAG: &str = "--batch-flame-texture";
const BATCH_HEAT_PLUME_FLAG: &str = "--batch-heat-plume";
const BATCH_PICK_FLAG: &str = "--batch-pick";
const BATCH_ANIM_EDIT_FLAG: &str = "--batch-anim-edit";
const BATCH_ANIM_DUMP_FLAG: &str = "--batch-anim-dump";
const BATCH_FLAME_TRACE_FLAG: &str = "--batch-flame-trace";
const BATCH_WALL_PROBE_FLAG: &str = "--batch-wall-probe";
const BATCH_WATER_PROBE_FLAG: &str = "--batch-water-probe";
const BATCH_DEBUG_ACTION_FLAG: &str = "--batch-debug-action";
pub const BATCH_LIST_DEBUG_ACTIONS_FLAG: &str = "--batch-list-debug-actions";
const DEFAULT_SCREENSHOT_FRAME: u64 = 120;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchCameraPose {
    pub yaw_degrees: f32,
    pub pitch_degrees: f32,
    pub distance: f32,
    pub pivot: Option<[f32; 3]>,
}

pub struct EngineCliOverrides {
    pub batch_run: Option<BatchRun>,
    pub flame_mode: Option<FlameShadingMode>,
    pub flame_debug_view: Option<thyllore_effect_core::FlameDebugView>,
    pub water_debug_view: Option<i32>,
    pub water_secondary: Option<thyllore_effect_core::WaterSecondaryRays>,
    pub water_caustic_debug: Option<i32>,
    pub water_history_weight: Option<f32>,
    pub water_fixed_time: Option<f32>,
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
    pub flame_sdf: Option<String>,
    pub pick_pixel: Option<(u32, u32)>,
    pub flame_bone: Option<String>,
    pub flame_texture_fit: Option<(String, f32, bool)>,
    pub flame_style: Option<(String, thyllore_effect_core::StyleGroups)>,
    pub flame_style_dump: Option<String>,
    pub heat_plume: Option<(f32, f32)>,
    pub batch_play: bool,
    pub scene_path: Option<String>,
    pub anim_edits: Vec<BatchAnimEdit>,
    pub anim_dump_path: Option<String>,
    pub debug_actions: Vec<BatchDebugAction>,
    pub wall_probe_path: Option<String>,
    pub water_probe_path: Option<String>,
}
pub fn resolve_engine_cli_overrides(args: &[String]) -> Result<EngineCliOverrides> {
    Ok(EngineCliOverrides {
        batch_run: batch_run_resolve_from_args(args)?,
        flame_mode: flame_mode_resolve_from_args(args)?,
        flame_debug_view: flame_debug_view_resolve_from_args(args)?,
        water_debug_view: water_debug_view_resolve_from_args(args)?,
        water_secondary: water_secondary_resolve_from_args(args)?,
        water_caustic_debug: water_caustic_debug_resolve_from_args(args)?,
        water_history_weight: water_history_weight_resolve_from_args(args)?,
        water_fixed_time: water_fixed_time_resolve_from_args(args)?,
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
        flame_texture_fit: flame_texture_fit_resolve_from_args(args)?,
        flame_style: flame_style_resolve_from_args(args)?,
        flame_style_dump: flag_value_resolve_from_args(args, BATCH_FLAME_STYLE_DUMP_FLAG)?,
        heat_plume: heat_plume_resolve_from_args(args)?,
        batch_play: args.iter().any(|a| a == "--batch-play"),
        scene_path: scene_path_resolve_from_args(args)?,
        anim_edits: anim_edits_resolve_from_args(args)?,
        anim_dump_path: flag_value_resolve_from_args(args, BATCH_ANIM_DUMP_FLAG)?,
        debug_actions: debug_actions_resolve_from_args(args)?,
        wall_probe_path: flag_value_resolve_from_args(args, BATCH_WALL_PROBE_FLAG)?,
        water_probe_path: flag_value_resolve_from_args(args, BATCH_WATER_PROBE_FLAG)?,
    })
}

pub(super) fn flag_value_resolve_from_args(args: &[String], flag: &str) -> Result<Option<String>> {
    let Some(position) = args.iter().position(|arg| arg == flag) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1).filter(|v| !v.starts_with("--")) else {
        bail!("{flag} requires a value");
    };
    Ok(Some(value.clone()))
}

pub fn camera_pose_resolve_from_args(args: &[String]) -> Result<Option<BatchCameraPose>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_CAMERA_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_CAMERA_FLAG} requires <yaw_deg>,<pitch_deg>,<distance>");
    };

    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 3 && parts.len() != 6 {
        bail!(
            "{BATCH_CAMERA_FLAG} expects <yaw>,<pitch>,<distance>[,<pivot_x>,<pivot_y>,<pivot_z>], got '{value}'"
        );
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
        pivot: (numbers.len() == 6).then(|| [numbers[3], numbers[4], numbers[5]]),
    }))
}

pub fn batch_run_resolve_from_args(args: &[String]) -> Result<Option<BatchRun>> {
    // Check for sequence mode first
    if let Some(sequence_position) = args
        .iter()
        .position(|arg| arg == BATCH_SCREENSHOT_SEQUENCE_FLAG)
    {
        let Some(value) = args
            .get(sequence_position + 1)
            .filter(|v| !v.starts_with("--"))
        else {
            bail!("{BATCH_SCREENSHOT_SEQUENCE_FLAG} requires <dir>,<count>,<stride>");
        };
        let parts: Vec<&str> = value.split(',').collect();
        if parts.len() != 3 {
            bail!("{BATCH_SCREENSHOT_SEQUENCE_FLAG} expects <dir>,<count>,<stride>, got '{value}'");
        }
        let dir = parts[0].trim();
        let count: u32 = parts[1].trim().parse().map_err(|_| {
            anyhow::anyhow!("invalid count '{}': expected positive integer", parts[1])
        })?;
        let stride: u32 = parts[2].trim().parse().map_err(|_| {
            anyhow::anyhow!("invalid stride '{}': expected positive integer", parts[2])
        })?;
        if count == 0 {
            bail!("{BATCH_SCREENSHOT_SEQUENCE_FLAG} count must be >= 1");
        }
        if stride == 0 {
            bail!("{BATCH_SCREENSHOT_SEQUENCE_FLAG} stride must be >= 1");
        }

        let screenshot_frame = match args.iter().position(|arg| arg == BATCH_FRAMES_FLAG) {
            Some(frames_position) => {
                let Some(value) = args.get(frames_position + 1) else {
                    bail!("{BATCH_FRAMES_FLAG} requires a frame count");
                };
                let frames: u64 = value.parse().map_err(|_| {
                    anyhow::anyhow!("invalid frame count '{value}': expected integer")
                })?;
                if frames == 0 {
                    bail!("{BATCH_FRAMES_FLAG} must be >= 1");
                }
                frames
            }
            None => DEFAULT_SCREENSHOT_FRAME,
        };

        let flame_set = flame_set_resolve_from_args(args)?;
        let dump_wall_probe = debug_actions_has_wall_probe_dump(args);
        let dump_water_debug = debug_actions_has_water_debug_dump(args);

        let mut batch = BatchRun::new(PathBuf::from(dir), screenshot_frame, flame_set);
        batch.dump_wall_probe = dump_wall_probe;
        batch.dump_water_debug = dump_water_debug;
        batch.captures_remaining = count;
        batch.stride = stride;
        batch.sequence_dir = Some(PathBuf::from(dir));
        batch.total_count = count;
        batch.flame_trace_path =
            flag_value_resolve_from_args(args, BATCH_FLAME_TRACE_FLAG)?.map(PathBuf::from);
        batch.wall_probe_path =
            flag_value_resolve_from_args(args, BATCH_WALL_PROBE_FLAG)?.map(PathBuf::from);
        batch.water_probe_path =
            flag_value_resolve_from_args(args, BATCH_WATER_PROBE_FLAG)?.map(PathBuf::from);
        Ok(Some(batch))
    } else {
        // Single-shot mode (existing behavior)
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
                let frames: u64 = value.parse().map_err(|_| {
                    anyhow::anyhow!("invalid frame count '{value}': expected integer")
                })?;
                if frames == 0 {
                    bail!("{BATCH_FRAMES_FLAG} must be >= 1");
                }
                frames
            }
            None => DEFAULT_SCREENSHOT_FRAME,
        };

        let flame_set = flame_set_resolve_from_args(args)?;

        let dump_wall_probe = debug_actions_has_wall_probe_dump(args);
        let dump_water_debug = debug_actions_has_water_debug_dump(args);

        let mut batch = BatchRun::new(output, screenshot_frame, flame_set);
        batch.dump_wall_probe = dump_wall_probe;
        batch.dump_water_debug = dump_water_debug;
        batch.flame_trace_path =
            flag_value_resolve_from_args(args, BATCH_FLAME_TRACE_FLAG)?.map(PathBuf::from);
        batch.wall_probe_path =
            flag_value_resolve_from_args(args, BATCH_WALL_PROBE_FLAG)?.map(PathBuf::from);
        batch.water_probe_path =
            flag_value_resolve_from_args(args, BATCH_WATER_PROBE_FLAG)?.map(PathBuf::from);
        Ok(Some(batch))
    }
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

const ROT_Z_DEG_KEY: &str = "rot_z_deg";

fn scene_path_resolve_from_args(args: &[String]) -> Result<Option<String>> {
    let Some(position) = args.iter().position(|arg| arg == "--batch-scene") else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("--batch-scene requires <path>");
    };
    Ok(Some(value.clone()))
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

    let mut batch = world.resource_mut::<BatchRun>();
    batch.frames_rendered += 1;
    if matches!(batch.state, BatchRunState::WaitingForFrame)
        && batch.frames_rendered >= batch.screenshot_frame
    {
        batch.state = BatchRunState::ScreenshotRequested;
    }
}

fn format_camera_string(world: &World) -> String {
    if let Some(camera) = world.get_resource::<crate::ecs::resource::Camera>() {
        format!(
            "{:.1},{:.1},{:.2}",
            camera.yaw.to_degrees(),
            camera.pitch.to_degrees(),
            camera.distance
        )
    } else {
        "default".to_string()
    }
}

pub fn batch_run_record_screenshot(world: &World, save_result: Result<String, String>) {
    let Some(mut batch) = world.get_resource_mut::<BatchRun>() else {
        return;
    };
    if !matches!(batch.state, BatchRunState::ScreenshotRequested) {
        return;
    }

    // Sequence mode: write frame file and continue or finish
    let sequence_dir = batch.sequence_dir.clone();
    if let Some(sequence_dir) = sequence_dir {
        let saved = match save_result {
            Ok(path) => path,
            Err(e) => {
                batch.state = BatchRunState::Completed { result: Err(e) };
                return;
            }
        };

        let frame_index = batch.total_count - batch.captures_remaining; // 0-based index of this capture
        let frame_path = sequence_dir.join(format!("frame_{:02}.png", frame_index));

        if let Some(parent) = frame_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                batch.state = BatchRunState::Completed {
                    result: Err(format!("failed to create {}: {e}", parent.display())),
                };
                return;
            }
        }

        let copy_result = std::fs::copy(&saved, &frame_path)
            .map_err(|e| format!("failed to copy {} to {}: {e}", saved, frame_path.display()));

        // Remove intermediate file
        if let Err(e) = std::fs::remove_file(&saved) {
            log_warn!("failed to remove intermediate screenshot {saved}: {e}");
        }

        batch.captures_remaining -= 1;

        if batch.captures_remaining > 0 {
            // More captures to do: advance screenshot_frame by stride and wait
            batch.screenshot_frame += batch.stride as u64;
            batch.state = BatchRunState::WaitingForFrame;
        } else {
            // Last capture: write meta.json and complete
            let camera_str = format_camera_string(world);
            let meta = json!({
                "fps": 60.0 / batch.stride as f64,
                "count": batch.stride,
                "stride": batch.stride,
                "camera": camera_str,
                "flame_set": batch.flame_set.iter().map(|(k, v)| json!({"key": k, "value": v})).collect::<Vec<_>>(),
                "engine_args": std::env::args().collect::<Vec<_>>(),
            });

            let meta_path = sequence_dir.join("meta.json");
            if let Err(e) = std::fs::write(
                &meta_path,
                serde_json::to_string_pretty(&meta).unwrap_or_default(),
            ) {
                batch.state = BatchRunState::Completed {
                    result: Err(format!("failed to write {}: {e}", meta_path.display())),
                };
                return;
            }

            match copy_result {
                Ok(_) => {
                    batch.state = BatchRunState::Completed {
                        result: Ok(frame_path.to_string_lossy().to_string()),
                    };
                }
                Err(e) => {
                    batch.state = BatchRunState::Completed { result: Err(e) };
                }
            }
        }
        return;
    }

    // Single-shot mode (existing behavior)
    let result = save_result.and_then(|saved| move_screenshot_to_output(&saved, &batch.output));

    // Flame trace / wall probe dumps: if paths are set, dump synchronously
    // at the same frame/time as the screenshot.
    let flame_trace_path = batch.flame_trace_path.clone();
    let wall_probe_path = batch.wall_probe_path.clone();
    if flame_trace_path.is_some() || wall_probe_path.is_some() {
        crate::ecs::systems::batch_run_flame_dump(
            world,
            flame_trace_path.as_deref(),
            wall_probe_path.as_deref(),
        );
    }

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

pub const DEBUG_ACTION_NAMES: &[&str] = &[
    "reset_camera",
    "reset_camera_up",
    "camera_to_model",
    "add_flame",
    "open_flame_curves",
    "view_mode=<final|position|normal|shadow_mask|ndotl|light_direction|view_depth|object_id|selection_view|selection_ubo>",
    "black_background (clear the HDR viewport to black and hide the grid and light gizmo, for reference-footage comparison)",
    "flame_clip_preview=<end_seconds> (draw the first flame's clip block as a mid-drag TrimEnd preview, without committing)",
    "timeline_select_flame_clip (enqueue TimelineSelectClip for the flame clip — the double-click path — to check it leaves the flame schedule's trim intact)",
    "dump_wall_probe (write camera pose + wall-regime ray diagnostics to log/flame/)",
    "dump_water_debug (write water parameters, UBO, camera and a screenshot to log/water/)",
    "apply_texture_fit:<path>,<blend>,<profile|statistics> (clone FlameEffect, apply texture fit from path, send UpdateFlameEffect)",
    "apply_texture_fit_roundtrip:<path>,<blend>,<profile|statistics> (same as apply_texture_fit, then restore original FlameEffect)",
    "spawn_cube (spawn the debug cube primitive, same as the debug window Spawn Cube button)",
    "spawn_sphere (spawn the debug sphere primitive, same as the debug window Spawn Sphere button)",
    "spawn_floor (spawn the debug floor primitive, same as the debug window Spawn Floor button)",
];

#[cfg(test)]
mod tests;
