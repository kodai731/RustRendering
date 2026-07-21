use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::{BatchRun, BatchRunState, FlameShadingMode};
use crate::ecs::world::World;

const BATCH_SCREENSHOT_FLAG: &str = "--batch-screenshot";
const BATCH_FRAMES_FLAG: &str = "--batch-frames";
const BATCH_FLAME_MODE_FLAG: &str = "--batch-flame-mode";
const BATCH_FLAME_STEPS_FLAG: &str = "--batch-flame-steps";
const BATCH_CAMERA_FLAG: &str = "--batch-camera";
const FLAME_DUMP_FLAG: &str = "--flame-dump";
const BATCH_FLAME_COUNT_FLAG: &str = "--batch-flame-count";
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
    pub flame_count: Option<usize>,
}

pub fn resolve_engine_cli_overrides(args: &[String]) -> Result<EngineCliOverrides> {
    Ok(EngineCliOverrides {
        batch_run: batch_run_resolve_from_args(args)?,
        flame_mode: flame_mode_resolve_from_args(args)?,
        flame_steps: flame_steps_resolve_from_args(args)?,
        camera_pose: camera_pose_resolve_from_args(args)?,
        flame_dump_path: flame_dump_path_resolve_from_args(args)?,
        flame_count: flame_count_resolve_from_args(args)?,
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

    Ok(Some(BatchRun::new(output, screenshot_frame)))
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
        world.insert_resource(BatchRun::new(PathBuf::from("/tmp/out.png"), 2));

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
            world.insert_resource(BatchRun::new(PathBuf::from("/tmp/out.png"), 100));
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
        world.insert_resource(BatchRun::new(PathBuf::from("/tmp/out.png"), 1));
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
        let batch = BatchRun::new(PathBuf::from("/tmp/out.png"), 1);
        let (ok, line) = batch_run_report(&batch);
        assert!(!ok);
        assert!(line.contains("before screenshot completed"));
    }
}
