use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use cgmath::Vector2;

use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::{BatchRun, BatchRunState, FlameEffect, FlameShadingMode};
use crate::ecs::world::World;

const BATCH_SCREENSHOT_FLAG: &str = "--batch-screenshot";
const BATCH_FRAMES_FLAG: &str = "--batch-frames";
const BATCH_FLAME_MODE_FLAG: &str = "--batch-flame-mode";
const BATCH_FLAME_STEPS_FLAG: &str = "--batch-flame-steps";
const BATCH_CAMERA_FLAG: &str = "--batch-camera";
const FLAME_DUMP_FLAG: &str = "--flame-dump";
const GPU_TIMINGS_FLAG: &str = "--gpu-timings";
const BATCH_FLAME_COUNT_FLAG: &str = "--batch-flame-count";
const BATCH_FLAME_SET_FLAG: &str = "--batch-flame-set";
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
    pub flame_count: Option<usize>,
    pub flame_set: Vec<(String, f32)>,
}

pub fn resolve_engine_cli_overrides(args: &[String]) -> Result<EngineCliOverrides> {
    Ok(EngineCliOverrides {
        batch_run: batch_run_resolve_from_args(args)?,
        flame_mode: flame_mode_resolve_from_args(args)?,
        flame_steps: flame_steps_resolve_from_args(args)?,
        camera_pose: camera_pose_resolve_from_args(args)?,
        flame_dump_path: flame_dump_path_resolve_from_args(args)?,
        gpu_timings_path: gpu_timings_path_resolve_from_args(args)?,
        flame_count: flame_count_resolve_from_args(args)?,
        flame_set: flame_set_resolve_from_args(args)?,
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
        bail!("{BATCH_FLAME_MODE_FLAG} requires a value: analytic|raymarch|thickness|noise");
    };
    let mode = FlameShadingMode::parse(value).ok_or_else(|| {
        anyhow::anyhow!("invalid flame mode '{value}': expected analytic|raymarch|thickness|noise")
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
        bail!("{BATCH_FLAME_COUNT_FLAG} must be in range 1..=4, got {}", count);
    }
    Ok(Some(count))
}
pub(crate) const FLAME_SET_KEYS: &[&str] = &[
    "warp_amp", "warp_freq", "rise_speed", "taper_power", "radius_tip_ratio",
    "edge_low", "edge_high", "white_boost", "bend_amount", "bend_power",
    "wind_x", "wind_z", "noise_amplitude", "noise_frequency", "noise_scroll_speed",
    "sigma_t", "intensity", "height", "radius", "time", "time_scale", "time_offset",
    "rot_z_deg", "temperature_base_k", "temperature_tip_k",
    "envelope_peak", "envelope_base", "envelope_tail", "radial_sharpness",
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
            "rot_z_deg" => effect.rotation = cgmath::Quaternion::from(cgmath::Euler::new(cgmath::Deg(0.0), cgmath::Deg(0.0), cgmath::Deg(*value))),
            _ => unreachable!("unknown key (parser should have rejected)"),
        }
    }
}

fn resolve_absolute_output(output: &Path) -> Result<PathBuf> {
    if output.extension().and_then(|e| e.to_str()) != Some("png") {
        bail!("batch screenshot output must end with .png: {}", output.display());
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
            world.insert_resource(BatchRun::new(PathBuf::from("/tmp/out.png"), 100, Vec::new()));
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
        let args: Vec<String> = vec![
            "--batch-flame-set=noise_amplitude=0.35".into(),
        ];
        let pairs = flame_set_resolve_from_args(&args).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "noise_amplitude");
        assert!((pairs[0].1 - 0.35).abs() < 1e-6);
    }

    #[test]
    fn flame_set_separate_form() {
        let args: Vec<String> = vec![
            "--batch-flame-set".into(),
            "noise_amplitude=0.35".into(),
        ];
        let pairs = flame_set_resolve_from_args(&args).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "noise_amplitude");
        assert!((pairs[0].1 - 0.35).abs() < 1e-6);
    }

    #[test]
    fn flame_set_unknown_key_error() {
        let args: Vec<String> = vec![
            "--batch-flame-set".into(),
            "invalid_key=1.0".into(),
        ];
        let err = flame_set_resolve_from_args(&args).unwrap_err();
        assert!(
            err.to_string().contains("invalid_key"),
        );
    }

    #[test]
    fn apply_flame_overrides_no_panic_for_all_keys() {
        for &key in FLAME_SET_KEYS {
            let mut effect = FlameEffect::default();
            let overrides: Vec<(String, f32)> = vec![(key.to_string(), 1.0)];
            apply_flame_overrides(&mut effect, &overrides);
        }
    }
}
