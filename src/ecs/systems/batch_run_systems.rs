use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::json;

use thyllore_anim_core::editable::PropertyType;

use crate::asset::AssetStorage;
use crate::ecs::component::{
    scalar_channel_domains, scalar_channel_for_cli_name, scalar_channel_for_property,
    scalar_cli_names_joined, ClipSchedule, FlameEffect,
};
use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::{
    BatchRun, BatchRunState, ClipLibrary, DebugViewMode, DebugViewState, FlameShadingMode,
    TimelineState,
};
use crate::ecs::world::World;

const BATCH_SCREENSHOT_FLAG: &str = "--batch-screenshot";
const BATCH_SCREENSHOT_SEQUENCE_FLAG: &str = "--batch-screenshot-sequence";
const BATCH_FRAMES_FLAG: &str = "--batch-frames";
const BATCH_FLAME_MODE_FLAG: &str = "--batch-flame-mode";
const BATCH_FLAME_DEBUG_VIEW_FLAG: &str = "--batch-flame-debug-view";
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
const BATCH_FLAME_TEXTURE_FLAG: &str = "--batch-flame-texture";
const BATCH_HEAT_PLUME_FLAG: &str = "--batch-heat-plume";
const BATCH_PICK_FLAG: &str = "--batch-pick";
const BATCH_ANIM_EDIT_FLAG: &str = "--batch-anim-edit";
const BATCH_ANIM_DUMP_FLAG: &str = "--batch-anim-dump";
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
    pub heat_plume: Option<(f32, f32)>,
    pub batch_play: bool,
    pub scene_path: Option<String>,
    pub anim_edits: Vec<BatchAnimEdit>,
    pub anim_dump_path: Option<String>,
    pub debug_actions: Vec<BatchDebugAction>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BatchAnimEdit {
    DebugKeys {
        seed: u64,
    },
    Key {
        property_type: PropertyType,
        time: f32,
        value: f32,
    },
    KeyAtPlayhead {
        property_type: PropertyType,
    },
    TrimEnd {
        seconds: f32,
    },
    Clear,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BatchDebugAction {
    ResetCamera,
    ResetCameraUp,
    CameraToModel,
    AddFlame,
    OpenFlameCurves,
    ViewMode(DebugViewMode),
    FlameClipPreview {
        end_seconds: f32,
    },
    TimelineSelectFlameClip,
    WallProbeDump,
    ApplyTextureFit {
        path: String,
        blend: f32,
        profile: bool,
    },
    ApplyTextureFitRoundtrip {
        path: String,
        blend: f32,
        profile: bool,
    },
}

pub fn resolve_engine_cli_overrides(args: &[String]) -> Result<EngineCliOverrides> {
    Ok(EngineCliOverrides {
        batch_run: batch_run_resolve_from_args(args)?,
        flame_mode: flame_mode_resolve_from_args(args)?,
        flame_debug_view: flame_debug_view_resolve_from_args(args)?,
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
        heat_plume: heat_plume_resolve_from_args(args)?,
        batch_play: args.iter().any(|a| a == "--batch-play"),
        scene_path: scene_path_resolve_from_args(args)?,
        anim_edits: anim_edits_resolve_from_args(args)?,
        anim_dump_path: flag_value_resolve_from_args(args, BATCH_ANIM_DUMP_FLAG)?,
        debug_actions: debug_actions_resolve_from_args(args)?,
    })
}

fn flag_value_resolve_from_args(args: &[String], flag: &str) -> Result<Option<String>> {
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

        let mut batch = BatchRun::new(PathBuf::from(dir), screenshot_frame, flame_set);
        batch.dump_wall_probe = dump_wall_probe;
        batch.captures_remaining = count;
        batch.stride = stride;
        batch.sequence_dir = Some(PathBuf::from(dir));
        batch.total_count = count;
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

        let mut batch = BatchRun::new(output, screenshot_frame, flame_set);
        batch.dump_wall_probe = dump_wall_probe;
        Ok(Some(batch))
    }
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

pub fn flame_debug_view_resolve_from_args(
    args: &[String],
) -> Result<Option<thyllore_effect_core::FlameDebugView>> {
    let Some(position) = args
        .iter()
        .position(|arg| arg == BATCH_FLAME_DEBUG_VIEW_FLAG)
    else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_FLAME_DEBUG_VIEW_FLAG} requires a value: off|shaped|erosion|argument|density|sigma|emission|jitter|wcoord");
    };
    let view = thyllore_effect_core::FlameDebugView::parse(value).ok_or_else(|| {
        anyhow::anyhow!(
            "invalid flame debug view '{value}': expected off|shaped|erosion|argument|density|sigma|emission|jitter|wcoord|grid|strain|stretch"
        )
    })?;
    Ok(Some(view))
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
    "noise_contrast",
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
    "aniso_axis_advect",
    "rte_bands",
    "sigma_dispersion",
    "edge_temperature_blend",
    "boundary_amp",
    "near_fade_radius",
    "carve_residual",
    "tip_carve_depth",
    "tip_carve_reach",
    "warp_reach",
    "swirl_gain",
    "swirl_speed",
    "spread_gain",
    "support_margin",
    "meander_amp",
    "boundary_freq",
    "boundary_speed",
    "boundary_radius_ratio",
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
    if !thyllore_effect_core::FLAME_PRESET_NAMES.contains(&value.as_str()) {
        bail!(
            "unknown flame preset '{}'. Valid presets: {}",
            value,
            thyllore_effect_core::FLAME_PRESET_NAMES.join(", ")
        );
    }
    Ok(Some(value.clone()))
}

fn flame_texture_fit_resolve_from_args(args: &[String]) -> Result<Option<(String, f32, bool)>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_FLAME_TEXTURE_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_FLAME_TEXTURE_FLAG} requires <path>[,<blend>[,<profile>]]");
    };
    let parts: Vec<&str> = value.split(',').collect();
    let path = parts[0].trim().to_string();
    let blend = if parts.len() > 1 {
        parts[1].trim().parse::<f32>().map_err(|_| {
            anyhow::anyhow!(
                "invalid {BATCH_FLAME_TEXTURE_FLAG} blend value '{}'",
                parts[1]
            )
        })?
    } else {
        1.0
    };
    if !blend.is_finite() || blend < 0.0 || blend > 1.0 {
        bail!("{BATCH_FLAME_TEXTURE_FLAG} blend must be in [0, 1] and finite: '{value}'");
    }
    let profile = match parts.get(2).map(|s| s.trim()) {
        None | Some("statistics") => false,
        Some("profile") => true,
        Some(other) => {
            bail!(
                "invalid {BATCH_FLAME_TEXTURE_FLAG} profile value '{}'; must be 'profile' or 'statistics'",
                other
            );
        }
    };
    Ok(Some((path, blend, profile)))
}

fn heat_plume_resolve_from_args(args: &[String]) -> Result<Option<(f32, f32)>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_HEAT_PLUME_FLAG) else {
        return Ok(None);
    };
    let next = args.get(position + 1);
    let value = match next {
        None => "10.0,0.5",
        Some(value) if value.starts_with("--") => "10.0,0.5",
        Some(value) => value.as_str(),
    };
    let parts: Vec<&str> = value.split(',').collect();
    let (gain, amp) = match parts.len() {
        1 => {
            let gain: f32 = parts[0]
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid {BATCH_HEAT_PLUME_FLAG} value '{value}'"))?;
            if !gain.is_finite() || gain < 0.0 {
                bail!("{BATCH_HEAT_PLUME_FLAG} gain must be >= 0 and finite: '{value}'");
            }
            (gain, 0.5)
        }
        2 => {
            let gain: f32 = parts[0]
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid {BATCH_HEAT_PLUME_FLAG} value '{value}'"))?;
            let amp: f32 = parts[1]
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid {BATCH_HEAT_PLUME_FLAG} value '{value}'"))?;
            if !gain.is_finite() || gain < 0.0 {
                bail!("{BATCH_HEAT_PLUME_FLAG} gain must be >= 0 and finite: '{value}'");
            }
            if !amp.is_finite() || amp < 0.0 {
                bail!("{BATCH_HEAT_PLUME_FLAG} amp must be >= 0 and finite: '{value}'");
            }
            (gain, amp)
        }
        _ => bail!("{BATCH_HEAT_PLUME_FLAG} expects <gain>[,<amp>] but got '{value}'"),
    };
    Ok(Some((gain, amp)))
}

fn scene_path_resolve_from_args(args: &[String]) -> Result<Option<String>> {
    let Some(position) = args.iter().position(|arg| arg == "--batch-scene") else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("--batch-scene requires <path>");
    };
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
            "noise_contrast" => effect.noise_contrast = *value,
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
            "aniso_axis_advect" => effect.aniso_axis_advect = *value,
            "rte_bands" => effect.rte_bands = *value,
            "sigma_dispersion" => effect.sigma_dispersion = *value,
            "edge_temperature_blend" => effect.edge_temperature_blend = *value,
            "boundary_amp" => effect.boundary_amp = *value,
            "near_fade_radius" => effect.near_fade_radius = *value,
            "carve_residual" => effect.carve_residual = *value,
            "tip_carve_depth" => effect.tip_carve_depth = *value,
            "tip_carve_reach" => effect.tip_carve_reach = *value,
            "warp_reach" => effect.warp_reach = *value,
            "swirl_gain" => effect.swirl_gain = *value,
            "swirl_speed" => effect.swirl_speed = *value,
            "support_margin" => effect.support_margin = *value,
            "spread_gain" => effect.spread_gain = *value,
            "meander_amp" => effect.meander_amp = *value,
            "boundary_freq" => effect.boundary_freq = *value,
            "boundary_speed" => effect.boundary_speed = *value,
            "boundary_radius_ratio" => effect.boundary_radius_ratio = *value,
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

pub fn apply_texture_fit_from_path(
    effect: &mut FlameEffect,
    baked: &mut thyllore_effect_core::FlameBaked,
    path: &str,
    blend: f32,
    groups: thyllore_effect_core::TextureFitGroups,
    profile: bool,
    route: &str,
) {
    let effect_before = effect.clone();
    let baked_before = *baked;
    let request = json!({
        "blend": blend,
        "profile": profile,
        "groups": {
            "silhouette": groups.silhouette,
            "color": groups.color,
            "turbulence": groups.turbulence,
            "tilt": groups.tilt,
        },
    });
    let dump = |source_bytes: Option<&[u8]>,
                result: serde_json::Value,
                effect_after: &FlameEffect,
                baked_after: &thyllore_effect_core::FlameBaked| {
        crate::ecs::systems::write_texture_fit_provenance(
            route,
            path,
            source_bytes,
            request.clone(),
            result,
            (&effect_before, &baked_before),
            (effect_after, baked_after),
        );
    };

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "warning: failed to read texture fit image '{}': {}",
                path, e
            );
            dump(
                None,
                json!({"ok": false, "error": "not_found", "detail": e.to_string()}),
                effect,
                baked,
            );
            return;
        }
    };

    let decoder = png::Decoder::new(Cursor::new(&bytes));
    let mut reader = match decoder.read_info() {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "warning: failed to decode texture fit image '{}': {}",
                path, e
            );
            dump(
                Some(&bytes),
                json!({"ok": false, "error": "decode_failed", "detail": e.to_string()}),
                effect,
                baked,
            );
            return;
        }
    };

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = match reader.next_frame(&mut buf) {
        Ok(i) => i,
        Err(e) => {
            eprintln!(
                "warning: failed to read texture fit image frame '{}': {}",
                path, e
            );
            dump(
                Some(&bytes),
                json!({"ok": false, "error": "decode_failed", "detail": e.to_string()}),
                effect,
                baked,
            );
            return;
        }
    };

    let width = info.width as usize;
    let height = info.height as usize;
    let png_json = json!({
        "width": width,
        "height": height,
        "color_type": format!("{:?}", info.color_type),
        "bit_depth": format!("{:?}", info.bit_depth),
    });
    let bytes_per_pixel = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => {
            eprintln!(
                "warning: unsupported PNG color type in texture fit image '{}'",
                path
            );
            dump(
                Some(&bytes),
                json!({"ok": false, "error": "unsupported_color_type", "png": png_json}),
                effect,
                baked,
            );
            return;
        }
    };

    let buf = &buf[..info.buffer_size()];
    let total_pixels = width * height;
    let mut pixels: Vec<[f32; 3]> = Vec::with_capacity(total_pixels);
    for i in (0..buf.len()).step_by(bytes_per_pixel) {
        let r = buf[i] as f32 / 255.0;
        let g = buf[i + 1] as f32 / 255.0;
        let b = buf[i + 2] as f32 / 255.0;
        pixels.push([
            thyllore_effect_core::flame_fit::srgb_to_linear(r),
            thyllore_effect_core::flame_fit::srgb_to_linear(g),
            thyllore_effect_core::flame_fit::srgb_to_linear(b),
        ]);
    }
    let mut max_luminance = 0.0f32;
    let mut luminance_sum = 0.0f64;
    for pixel in &pixels {
        let luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
        max_luminance = max_luminance.max(luminance);
        luminance_sum += luminance as f64;
    }
    let decode_json = json!({
        "max_luminance": max_luminance,
        "mean_luminance": luminance_sum / total_pixels.max(1) as f64,
    });

    let fit = match thyllore_effect_core::fit_flame_texture(&pixels, width, height, effect, baked) {
        Some(f) => f,
        None => {
            eprintln!("warning: texture fit failed for image '{}'", path);
            dump(
                Some(&bytes),
                json!({
                    "ok": false,
                    "error": "mask_empty",
                    "png": png_json,
                    "decode": decode_json,
                }),
                effect,
                baked,
            );
            return;
        }
    };

    thyllore_effect_core::apply_texture_fit(effect, baked, &fit, groups, blend, profile);
    dump(
        Some(&bytes),
        json!({
            "ok": true,
            "png": png_json,
            "decode": decode_json,
            "fit": {
                "envelope_peak": fit.envelope_peak,
                "envelope_base": fit.envelope_base,
                "envelope_tail": fit.envelope_tail,
                "radius": fit.radius,
                "radius_tip_ratio": fit.radius_tip_ratio,
                "taper_power": fit.taper_power,
                "use_blackbody": fit.use_blackbody,
                "temperature_base_k": fit.temperature_base_k,
                "temperature_tip_k": fit.temperature_tip_k,
                "noise_amplitude": fit.noise_amplitude,
                "suggested_instances": fit.suggested_instances,
            },
        }),
        effect,
        baked,
    );
}

/// Parse repeated `--batch-anim-edit <spec>` flags. Specs:
/// `debug_keys=<seed>` | `key=<param>@<time>=<value>` | `clear`.
fn anim_edits_resolve_from_args(args: &[String]) -> Result<Vec<BatchAnimEdit>> {
    let mut edits = Vec::new();
    for i in 0..args.len() {
        if args[i] != BATCH_ANIM_EDIT_FLAG {
            continue;
        }
        let Some(spec) = args.get(i + 1).filter(|v| !v.starts_with("--")) else {
            bail!("{BATCH_ANIM_EDIT_FLAG} requires a spec: debug_keys=<seed> | key=<param>@<time>=<value> | key_at_playhead=<param> | trim_end=<seconds> | clear");
        };
        edits.push(anim_edit_parse_spec(spec)?);
    }
    Ok(edits)
}

fn anim_edit_parse_spec(spec: &str) -> Result<BatchAnimEdit> {
    let spec = spec.trim();
    if spec == "clear" {
        return Ok(BatchAnimEdit::Clear);
    }
    if let Some(seed_str) = spec.strip_prefix("debug_keys=") {
        let seed: u64 = seed_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid debug_keys seed '{seed_str}': expected u64"))?;
        return Ok(BatchAnimEdit::DebugKeys { seed });
    }
    if let Some(param_str) = spec.strip_prefix("key_at_playhead=") {
        let (_, channel) = scalar_channel_for_cli_name(param_str.trim()).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown scalar channel '{}'. Valid channels: {}",
                param_str,
                scalar_cli_names_joined()
            )
        })?;
        return Ok(BatchAnimEdit::KeyAtPlayhead {
            property_type: channel.property_type(),
        });
    }
    if let Some(seconds_str) = spec.strip_prefix("trim_end=") {
        let seconds: f32 = seconds_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid trim_end seconds '{seconds_str}'"))?;
        if !seconds.is_finite() || seconds < 0.0 {
            bail!("trim_end seconds must be >= 0 and finite: '{spec}'");
        }
        return Ok(BatchAnimEdit::TrimEnd { seconds });
    }
    if let Some(rest) = spec.strip_prefix("key=") {
        let (param_str, rest) = rest.split_once('@').ok_or_else(|| {
            anyhow::anyhow!("key spec must be key=<param>@<time>=<value>, got '{spec}'")
        })?;
        let (time_str, value_str) = rest.split_once('=').ok_or_else(|| {
            anyhow::anyhow!("key spec must be key=<param>@<time>=<value>, got '{spec}'")
        })?;
        let (_, channel) = scalar_channel_for_cli_name(param_str.trim()).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown scalar channel '{}'. Valid channels: {}",
                param_str,
                scalar_cli_names_joined()
            )
        })?;
        let time: f32 = time_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid key time '{time_str}'"))?;
        let value: f32 = value_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid key value '{value_str}'"))?;
        if !time.is_finite() || time < 0.0 || !value.is_finite() {
            bail!("key time must be >= 0 and value finite: '{spec}'");
        }
        return Ok(BatchAnimEdit::Key {
            property_type: channel.property_type(),
            time,
            value,
        });
    }
    bail!("unknown anim edit spec '{spec}'. Expected debug_keys=<seed> | key=<param>@<time>=<value> | key_at_playhead=<param> | trim_end=<seconds> | clear")
}

/// Apply anim edits through the production scalar-clip event dispatcher, so batch
/// runs exercise the same path as the UI (clip creation, undo history, schedule
/// extension). Key edits temporarily move the timeline to the key's time because
/// `InsertScalarKey` always keys at `TimelineState::current_time`.
pub fn batch_apply_anim_edits(
    world: &mut World,
    assets: &mut AssetStorage,
    edits: &[BatchAnimEdit],
) {
    use super::phases::dispatch_scalar_curve::dispatch_scalar_clip_events;
    use super::scalar_clip_systems::{ensure_entity_clip, resolve_selected_scalar_entity};

    for edit in edits {
        match edit {
            BatchAnimEdit::DebugKeys { seed } => {
                dispatch_scalar_clip_events(
                    &[UIEvent::InsertScalarDebugKeys { seed: *seed }],
                    world,
                    assets,
                );
            }
            BatchAnimEdit::Key {
                property_type,
                time,
                value,
            } => {
                let previous_time = {
                    let mut timeline = world.resource_mut::<TimelineState>();
                    let previous = timeline.current_time;
                    timeline.current_time = *time;
                    previous
                };
                dispatch_scalar_clip_events(
                    &[UIEvent::InsertScalarKey {
                        property_type: *property_type,
                        value: *value,
                    }],
                    world,
                    assets,
                );
                world.resource_mut::<TimelineState>().current_time = previous_time;
            }
            BatchAnimEdit::KeyAtPlayhead { property_type } => {
                dispatch_scalar_clip_events(
                    &[UIEvent::InsertScalarKeyAtPlayhead {
                        property_type: *property_type,
                    }],
                    world,
                    assets,
                );
            }
            BatchAnimEdit::TrimEnd { seconds } => {
                let Some((entity, domain)) = resolve_selected_scalar_entity(world) else {
                    continue;
                };
                let clip_id = ensure_entity_clip(world, assets, entity, domain);
                let Some(instance_id) = world.get_component::<ClipSchedule>(entity).and_then(|s| {
                    s.instances
                        .iter()
                        .find(|i| i.source_id == clip_id)
                        .map(|i| i.instance_id)
                }) else {
                    continue;
                };
                super::timeline_systems::process_clip_instance_events(
                    &[UIEvent::ClipInstanceTrimEnd {
                        entity,
                        instance_id,
                        new_clip_out: *seconds,
                    }],
                    world,
                );
            }
            BatchAnimEdit::Clear => {
                dispatch_scalar_clip_events(&[UIEvent::ClearScalarKeys], world, assets);
            }
        }
    }
}

pub const DEBUG_ACTION_NAMES: &[&str] = &[
    "reset_camera",
    "reset_camera_up",
    "camera_to_model",
    "add_flame",
    "open_flame_curves",
    "view_mode=<final|position|normal|shadow_mask|ndotl|light_direction|view_depth|object_id|selection_view|selection_ubo>",
    "flame_clip_preview=<end_seconds> (draw the first flame's clip block as a mid-drag TrimEnd preview, without committing)",
    "timeline_select_flame_clip (enqueue TimelineSelectClip for the flame clip — the double-click path — to check it leaves the flame schedule's trim intact)",
    "dump_wall_probe (write camera pose + wall-regime ray diagnostics to log/flame/)",
    "apply_texture_fit:<path>,<blend>,<profile|statistics> (clone FlameEffect, apply texture fit from path, send UpdateFlameEffect)",
    "apply_texture_fit_roundtrip:<path>,<blend>,<profile|statistics> (same as apply_texture_fit, then restore original FlameEffect)",
];

fn debug_view_mode_parse(name: &str) -> Option<DebugViewMode> {
    match name {
        "final" => Some(DebugViewMode::Final),
        "position" => Some(DebugViewMode::Position),
        "normal" => Some(DebugViewMode::Normal),
        "shadow_mask" => Some(DebugViewMode::ShadowMask),
        "ndotl" => Some(DebugViewMode::NdotL),
        "light_direction" => Some(DebugViewMode::LightDirection),
        "view_depth" => Some(DebugViewMode::ViewDepth),
        "object_id" => Some(DebugViewMode::ObjectID),
        "selection_view" => Some(DebugViewMode::SelectionView),
        "selection_ubo" => Some(DebugViewMode::SelectionUBO),
        _ => None,
    }
}

fn debug_actions_resolve_from_args(args: &[String]) -> Result<Vec<BatchDebugAction>> {
    let mut actions = Vec::new();
    for i in 0..args.len() {
        if args[i] != BATCH_DEBUG_ACTION_FLAG {
            continue;
        }
        let Some(name) = args.get(i + 1).filter(|v| !v.starts_with("--")) else {
            bail!(
                "{BATCH_DEBUG_ACTION_FLAG} requires an action. Valid actions: {}",
                DEBUG_ACTION_NAMES.join(", ")
            );
        };
        actions.push(debug_action_parse(name)?);
    }
    Ok(actions)
}

fn debug_action_parse(name: &str) -> Result<BatchDebugAction> {
    let name = name.trim();
    if let Some(mode_str) = name.strip_prefix("view_mode=") {
        return debug_view_mode_parse(mode_str.trim())
            .map(BatchDebugAction::ViewMode)
            .ok_or_else(|| anyhow::anyhow!("unknown view_mode '{mode_str}'"));
    }
    if let Some(seconds_str) = name.strip_prefix("flame_clip_preview=") {
        let end_seconds: f32 = seconds_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid flame_clip_preview seconds '{seconds_str}'"))?;
        if !end_seconds.is_finite() || end_seconds < 0.0 {
            bail!("flame_clip_preview seconds must be >= 0 and finite: '{seconds_str}'");
        }
        return Ok(BatchDebugAction::FlameClipPreview { end_seconds });
    }
    if let Some(rest) = name.strip_prefix("apply_texture_fit:") {
        let (path, blend, profile) = parse_texture_fit_args(rest)?;
        return Ok(BatchDebugAction::ApplyTextureFit {
            path,
            blend,
            profile,
        });
    }
    if let Some(rest) = name.strip_prefix("apply_texture_fit_roundtrip:") {
        let (path, blend, profile) = parse_texture_fit_args(rest)?;
        return Ok(BatchDebugAction::ApplyTextureFitRoundtrip {
            path,
            blend,
            profile,
        });
    }
    match name {
        "timeline_select_flame_clip" => Ok(BatchDebugAction::TimelineSelectFlameClip),
        "reset_camera" => Ok(BatchDebugAction::ResetCamera),
        "reset_camera_up" => Ok(BatchDebugAction::ResetCameraUp),
        "camera_to_model" => Ok(BatchDebugAction::CameraToModel),
        "add_flame" => Ok(BatchDebugAction::AddFlame),
        "open_flame_curves" => Ok(BatchDebugAction::OpenFlameCurves),
        "dump_wall_probe" => Ok(BatchDebugAction::WallProbeDump),
        _ => bail!(
            "unknown debug action '{name}'. Valid actions: {}",
            DEBUG_ACTION_NAMES.join(", ")
        ),
    }
}

/// Check if `--batch-debug-action dump_wall_probe` is present in the args.
fn debug_actions_has_wall_probe_dump(args: &[String]) -> bool {
    for i in 0..args.len() {
        if args[i] == BATCH_DEBUG_ACTION_FLAG {
            if let Some(name) = args.get(i + 1).filter(|v| !v.starts_with("--")) {
                if name == "dump_wall_probe" {
                    return true;
                }
            }
        }
    }
    false
}

fn parse_texture_fit_args(rest: &str) -> Result<(String, f32, bool)> {
    let parts: Vec<&str> = rest.rsplit(',').collect();
    if parts.len() != 3 {
        bail!(
            "apply_texture_fit expects <path>,<blend>,<profile|statistics>, got '{}'",
            rest
        );
    }
    let profile_str = parts[0];
    let blend_str = parts[1];
    let path = parts[2].to_string();

    let blend: f32 = blend_str
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid blend value '{}'", blend_str))?;
    if !blend.is_finite() || !(0.0..=1.0).contains(&blend) {
        bail!("blend must be in [0.0, 1.0], got {}", blend);
    }

    let profile = match profile_str {
        "profile" => true,
        "statistics" => false,
        _ => bail!(
            "profile mode must be 'profile' or 'statistics', got '{}'",
            profile_str
        ),
    };

    Ok((path, blend, profile))
}

/// Execute debug-window actions headlessly: view-mode radios write the same
/// `DebugViewState` resource the imgui panel edits, buttons enqueue the same
/// `UIEvent`s so they run through the normal dispatch on the first frame.
pub fn batch_apply_debug_actions(world: &World, actions: &[BatchDebugAction]) {
    for action in actions {
        match action {
            BatchDebugAction::ViewMode(mode) => {
                world.resource_mut::<DebugViewState>().debug_view_mode = *mode;
            }
            BatchDebugAction::ResetCamera => {
                world
                    .resource_mut::<UIEventQueue>()
                    .send(UIEvent::ResetCamera);
            }
            BatchDebugAction::ResetCameraUp => {
                world
                    .resource_mut::<UIEventQueue>()
                    .send(UIEvent::ResetCameraUp);
            }
            BatchDebugAction::CameraToModel => {
                world
                    .resource_mut::<UIEventQueue>()
                    .send(UIEvent::MoveCameraToModel);
            }
            BatchDebugAction::AddFlame => {
                world.resource_mut::<UIEventQueue>().send(UIEvent::AddFlame);
            }
            BatchDebugAction::OpenFlameCurves => {
                world
                    .resource_mut::<UIEventQueue>()
                    .send(UIEvent::OpenScalarCurveEditor);
            }
            BatchDebugAction::FlameClipPreview { end_seconds } => {
                apply_flame_clip_preview(world, *end_seconds);
            }
            BatchDebugAction::WallProbeDump => {
                // Wall probe dump is now handled synchronously in the render path
                // via batch.dump_wall_probe, so this is a no-op.
            }
            BatchDebugAction::TimelineSelectFlameClip => {
                let clip_id = world.query_flames().first().and_then(|&flame| {
                    super::scalar_clip_systems::find_entity_clip_id(world, flame)
                });
                if let Some(clip_id) = clip_id {
                    world
                        .resource_mut::<UIEventQueue>()
                        .send(UIEvent::TimelineSelectClip(clip_id));
                }
            }
            BatchDebugAction::ApplyTextureFit {
                path,
                blend,
                profile,
            } => {
                let original = world.query_flames().first().and_then(|&flame| {
                    let effect = world.get_component::<FlameEffect>(flame)?.clone();
                    let baked = world
                        .get_component::<crate::ecs::component::FlameBaked>(flame)
                        .cloned()
                        .unwrap_or_default();
                    Some((effect, baked))
                });
                if let Some((mut copy, mut baked)) = original {
                    apply_texture_fit_from_path(
                        &mut copy,
                        &mut baked,
                        path,
                        *blend,
                        thyllore_effect_core::TextureFitGroups::default(),
                        *profile,
                        "debug_action",
                    );
                    world
                        .resource_mut::<UIEventQueue>()
                        .send(UIEvent::UpdateFlameEffect(Box::new(copy)));
                    world
                        .resource_mut::<UIEventQueue>()
                        .send(UIEvent::UpdateFlameBaked(Box::new(baked)));
                }
            }
            BatchDebugAction::ApplyTextureFitRoundtrip {
                path,
                blend,
                profile,
            } => {
                let original = world.query_flames().first().and_then(|&flame| {
                    let effect = world.get_component::<FlameEffect>(flame)?.clone();
                    let baked = world
                        .get_component::<crate::ecs::component::FlameBaked>(flame)
                        .cloned()
                        .unwrap_or_default();
                    Some((effect, baked))
                });
                if let Some((original_effect, original_baked)) = original {
                    let mut copy = original_effect.clone();
                    let mut baked = original_baked;
                    apply_texture_fit_from_path(
                        &mut copy,
                        &mut baked,
                        path,
                        *blend,
                        thyllore_effect_core::TextureFitGroups::default(),
                        *profile,
                        "debug_action",
                    );
                    world
                        .resource_mut::<UIEventQueue>()
                        .send(UIEvent::UpdateFlameEffect(Box::new(copy)));
                    world
                        .resource_mut::<UIEventQueue>()
                        .send(UIEvent::UpdateFlameBaked(Box::new(baked)));
                    world
                        .resource_mut::<UIEventQueue>()
                        .send(UIEvent::UpdateFlameEffect(Box::new(original_effect)));
                    world
                        .resource_mut::<UIEventQueue>()
                        .send(UIEvent::UpdateFlameBaked(Box::new(original_baked)));
                }
            }
        }
    }
}

/// Make the timeline draw the first flame's clip block as if a TrimEnd drag to
/// `end_seconds` were in progress: same preview math as the live drag, but no
/// commit event, so the underlying instance stays untouched.
fn apply_flame_clip_preview(world: &World, end_seconds: f32) {
    let Some(&flame) = world.query_flames().first() else {
        return;
    };
    let Some(instance) = world
        .get_component::<ClipSchedule>(flame)
        .and_then(|schedule| schedule.first_instance().cloned())
    else {
        return;
    };

    let (start_time, end_time) = super::timeline_systems::clip_drag_preview_times(
        &crate::ecs::resource::ClipDragType::TrimEnd,
        instance.clip_out,
        end_seconds - instance.clip_out,
        instance.start_time,
        instance.end_time(),
        instance.clip_in,
        instance.clip_out,
    );
    world
        .resource_mut::<crate::ecs::resource::TimelineInteractionState>()
        .drag_preview = Some(crate::ecs::resource::ClipDragPreview {
        entity: flame,
        instance_id: instance.instance_id,
        start_time,
        end_time,
    });
}

pub fn debug_actions_json() -> String {
    serde_json::json!({"ok": true, "actions": DEBUG_ACTION_NAMES}).to_string()
}

/// Serialize the animation-facing world state (flames, their scheduled clips,
/// every clip's scalar curves, timeline) so agents can inspect edits without a
/// window. Written once at engine exit; the file is the access surface.
pub fn batch_anim_dump_json(world: &World) -> serde_json::Value {
    use super::scalar_clip_systems::find_entity_clip_id;

    let entities: Vec<serde_json::Value> = scalar_channel_domains()
        .iter()
        .flat_map(|domain| {
            (domain.entities)(world).into_iter().map(move |entity| {
                let params: serde_json::Map<String, serde_json::Value> = domain
                    .channels
                    .iter()
                    .filter_map(|channel| {
                        (domain.read)(world, entity, channel.property_type())
                            .map(|value| (channel.cli_name.to_string(), value.into()))
                    })
                    .collect();
                let schedule: Vec<serde_json::Value> = world
                    .get_component::<ClipSchedule>(entity)
                    .map(|s| {
                        s.instances
                            .iter()
                            .map(|i| {
                                serde_json::json!({
                                    "instance_id": i.instance_id,
                                    "source_id": i.source_id,
                                    "start_time": i.start_time,
                                    "clip_in": i.clip_in,
                                    "clip_out": i.clip_out,
                                    "speed": i.speed,
                                    "muted": i.muted,
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                serde_json::json!({
                    "entity": entity,
                    "domain": domain.name,
                    "time": (domain.local_time)(world, entity),
                    "clip_id": find_entity_clip_id(world, entity),
                    "params": params,
                    "schedule": schedule,
                })
            })
        })
        .collect();

    let clips: Vec<serde_json::Value> = world
        .get_resource::<ClipLibrary>()
        .map(|library| {
            let mut ids: Vec<_> = library.all_clip_ids().copied().collect();
            ids.sort_unstable();
            ids.iter()
                .filter_map(|&id| library.get(id))
                .map(|clip| {
                    let curves: Vec<serde_json::Value> = clip
                        .scalar_curves
                        .iter()
                        .map(|curve| {
                            let property = scalar_channel_for_property(curve.property_type)
                                .map(|(_, c)| c.cli_name.to_string())
                                .unwrap_or_else(|| format!("{:?}", curve.property_type));
                            let keyframes: Vec<serde_json::Value> = curve
                                .keyframes
                                .iter()
                                .map(|k| serde_json::json!({"time": k.time, "value": k.value}))
                                .collect();
                            serde_json::json!({"property": property, "keyframes": keyframes})
                        })
                        .collect();
                    serde_json::json!({
                        "id": clip.id,
                        "name": clip.name,
                        "duration": clip.duration,
                        "bone_track_count": clip.tracks.len(),
                        "scalar_curves": curves,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let drag_preview = world
        .get_resource::<crate::ecs::resource::TimelineInteractionState>()
        .and_then(|s| s.drag_preview)
        .map(|p| {
            serde_json::json!({
                "entity": p.entity,
                "instance_id": p.instance_id,
                "start_time": p.start_time,
                "end_time": p.end_time,
            })
        })
        .unwrap_or(serde_json::Value::Null);

    let timeline = world
        .get_resource::<TimelineState>()
        .map(|t| {
            serde_json::json!({
                "current_time": t.current_time,
                "playing": t.playing,
                "looping": t.looping,
                "current_clip_id": t.current_clip_id,
                "drag_preview": drag_preview,
            })
        })
        .unwrap_or(serde_json::Value::Null);

    serde_json::json!({"entities": entities, "clips": clips, "timeline": timeline})
}

pub fn batch_anim_dump_write(world: &World, path: &str) -> Result<()> {
    let json = batch_anim_dump_json(world);
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&json)?)
        .with_context(|| format!("failed to write anim dump to {path}"))?;
    Ok(())
}

const BATCH_SEQUENCE_ANALYZE_FLAG: &str = "--batch-sequence-analyze";
const BATCH_SEQUENCE_DUMP_FLAG: &str = "--batch-sequence-dump";

#[derive(Clone, Debug)]
pub struct SequenceAnalyzeArgs {
    pub directories: Vec<(String, Option<u64>, Option<u64>)>,
    pub dump_path: String,
}

fn batch_sequence_analyze_resolve_from_args(
    args: &[String],
) -> Result<Option<SequenceAnalyzeArgs>> {
    let directories: Vec<String> = args
        .windows(2)
        .enumerate()
        .filter(|(_, window)| window[0] == BATCH_SEQUENCE_ANALYZE_FLAG)
        .map(|(_, window)| window[1].clone())
        .collect();

    if directories.is_empty() {
        return Ok(None);
    }

    let dump_path = flag_value_resolve_from_args(args, BATCH_SEQUENCE_DUMP_FLAG)?
        .ok_or_else(|| anyhow::anyhow!("{BATCH_SEQUENCE_DUMP_FLAG} is required when {BATCH_SEQUENCE_ANALYZE_FLAG} is specified"))?;

    let parsed: Vec<(String, Option<u64>, Option<u64>)> = directories
        .iter()
        .map(|spec| {
            let parts: Vec<&str> = spec.split(',').collect();
            match parts.len() {
                1 => {
                    let dir = parts[0].trim().to_string();
                    Ok((dir, None, None))
                }
                2 | 3 => {
                    let dir = parts[0].trim().to_string();
                    let from: Option<u64> = if parts.len() >= 2 && !parts[1].is_empty() {
                        Some(parts[1].trim().parse::<u64>().map_err(|_| {
                            anyhow::anyhow!("invalid range in {BATCH_SEQUENCE_ANALYZE_FLAG} value '{spec}': from must be a number")
                        })?)
                    } else {
                        None
                    };
                    let to: Option<u64> = if parts.len() == 3 && !parts[2].is_empty() {
                        Some(parts[2].trim().parse::<u64>().map_err(|_| {
                            anyhow::anyhow!("invalid range in {BATCH_SEQUENCE_ANALYZE_FLAG} value '{spec}': to must be a number")
                        })?)
                    } else {
                        None
                    };
                    Ok((dir, from, to))
                }
                _ => bail!("invalid {BATCH_SEQUENCE_ANALYZE_FLAG} value '{spec}': expected <dir>[,<from>,<to>]"),
            }
        })
        .collect::<Result<_>>()?;

    Ok(Some(SequenceAnalyzeArgs {
        directories: parsed,
        dump_path,
    }))
}

/// Headless sequence analysis: find frame_*.png files in each directory, compute luminance,
/// extract descriptors, and write JSON to the dump path.
pub fn run_sequence_analyze_from_args(args: Vec<String>) -> Option<Result<()>> {
    let args_slice: Vec<String> = args;
    if !args_slice.iter().any(|a| a == BATCH_SEQUENCE_ANALYZE_FLAG) {
        return None;
    }

    Some(run_sequence_analyze(&args_slice))
}

fn run_sequence_analyze(args: &[String]) -> Result<()> {
    let analyze_args = batch_sequence_analyze_resolve_from_args(args)?.ok_or_else(|| {
        anyhow::anyhow!("{BATCH_SEQUENCE_ANALYZE_FLAG} requires at least one directory argument")
    })?;

    let mut sequences = Vec::new();

    for (dir, from, to) in &analyze_args.directories {
        let entries =
            std::fs::read_dir(dir).with_context(|| format!("failed to read directory {dir}"))?;

        let mut frame_files: Vec<(u64, PathBuf)> = Vec::new();
        for entry in entries {
            let entry = entry.with_context(|| format!("failed to read entry in {dir}"))?;
            let path = entry.path();
            let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if filename.starts_with("frame_") && filename.ends_with(".png") {
                // Extract number from frame_NNNN.png
                let stem = &filename[6..filename.len() - 4];
                if let Ok(num) = stem.parse::<u64>() {
                    frame_files.push((num, path));
                }
            } else if filename.starts_with("frame_")
                && (filename.ends_with(".jpg") || filename.ends_with(".jpeg"))
            {
                bail!(
                    "found JPG file in directory {dir}: {filename} — only PNG files are supported"
                );
            }
        }

        frame_files.sort_by_key(|(num, _)| *num);

        // Apply range filter
        let filtered: Vec<(u64, PathBuf)> = if let (Some(f), Some(t)) = (from, to) {
            frame_files
                .into_iter()
                .filter(|(num, _)| *num >= *f && *num <= *t)
                .collect()
        } else if let Some(f) = from {
            frame_files
                .into_iter()
                .filter(|(num, _)| *num >= *f)
                .collect()
        } else if let Some(t) = to {
            frame_files
                .into_iter()
                .filter(|(num, _)| *num <= *t)
                .collect()
        } else {
            frame_files
        };

        if filtered.is_empty() {
            eprintln!("warning: no frame_*.png files found in {dir}");
            continue;
        }

        // Read FPS from meta.json
        let fps = read_fps_from_meta(dir);

        // Read each PNG and compute average luminance
        let mut frames: Vec<Vec<f32>> = Vec::new();
        let mut width: usize = 0;
        let mut height: usize = 0;

        for (num, path) in &filtered {
            let (w, h, luminance) = read_png_luminance(path).with_context(|| {
                format!("failed to read PNG at {} (frame #{num})", path.display())
            })?;
            if frames.is_empty() {
                width = w;
                height = h;
            } else if w != width || h != height {
                bail!(
                    "inconsistent frame size: expected {}x{}, got {}x{} at {}",
                    width,
                    height,
                    w,
                    h,
                    path.display()
                );
            }
            frames.push(luminance);
        }

        // Extract descriptors
        let descriptors =
            thyllore_texture_fit_core::sequence_descriptors::extract_sequence_descriptors(
                &frames, width, height, fps,
            )
            .with_context(|| format!("failed to extract descriptors for {dir}"))?;

        let descriptors_json = serde_json::to_value(&descriptors)?;
        sequences.push(json!({
            "dir": dir,
            "descriptors": descriptors_json,
        }));
    }

    let output = json!({"sequences": sequences});
    let dump_path = PathBuf::from(&analyze_args.dump_path);
    if let Some(parent) = dump_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(&dump_path, serde_json::to_string_pretty(&output)?).with_context(|| {
        format!(
            "failed to write sequence analysis dump to {}",
            dump_path.display()
        )
    })?;

    Ok(())
}

fn read_fps_from_meta(dir: &str) -> f32 {
    let meta_path = PathBuf::from(dir).join("meta.json");
    if let Ok(content) = std::fs::read_to_string(&meta_path) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(fps) = value.get("fps").and_then(|v| v.as_f64()) {
                return fps as f32;
            }
        }
    }
    10.0
}

fn read_png_luminance(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let decoder = png::Decoder::new(reader);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;

    let (width, height) = (info.width as usize, info.height as usize);

    let bytes_per_pixel = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        _ => bail!("unsupported PNG color type: {:?}", info.color_type),
    };

    let buf = &buf[..info.buffer_size()];
    let mut luminance = Vec::with_capacity(width * height);
    for chunk in buf.chunks(bytes_per_pixel) {
        let lum = match bytes_per_pixel {
            3 => (chunk[0] as f32 + chunk[1] as f32 + chunk[2] as f32) / 3.0,
            4 => (chunk[0] as f32 + chunk[1] as f32 + chunk[2] as f32) / 3.0,
            1 => chunk[0] as f32,
            2 => chunk[0] as f32,
            _ => unreachable!(),
        };
        luminance.push(lum);
    }

    Ok((width, height, luminance))
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
                distance: 4.0,
                pivot: None
            }
        );
        let pose =
            camera_pose_resolve_from_args(&args(&["bin", "--batch-camera", "30,5,4,0,1.2,0"]))
                .unwrap()
                .unwrap();
        assert_eq!(pose.pivot, Some([0.0, 1.2, 0.0]));
        assert!(camera_pose_resolve_from_args(&args(&["bin"]))
            .unwrap()
            .is_none());
    }

    #[test]
    fn resolve_rejects_invalid_camera_pose() {
        for value in ["30,5", "a,b,c", "30,5,0", "30,5,-1", "30,5,4,0,1"] {
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
        thyllore_effect_core::apply_flame_preset(&mut effect, "candle");

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

    #[test]
    fn flame_texture_fit_path_only_defaults_blend_to_one() {
        let resolved = flame_texture_fit_resolve_from_args(&args(&[
            "bin",
            "--batch-flame-texture",
            "image.png",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(resolved.0, "image.png");
        assert!((resolved.1 - 1.0).abs() < 1e-6);
        assert!(!resolved.2);
    }

    #[test]
    fn flame_texture_fit_path_with_blend() {
        let resolved = flame_texture_fit_resolve_from_args(&args(&[
            "bin",
            "--batch-flame-texture",
            "image.png,0.4",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(resolved.0, "image.png");
        assert!((resolved.1 - 0.4).abs() < 1e-6);
        assert!(!resolved.2);
    }

    #[test]
    fn flame_texture_fit_invalid_blend_is_err() {
        assert!(flame_texture_fit_resolve_from_args(&args(&[
            "bin",
            "--batch-flame-texture",
            "image.png,abc"
        ]))
        .is_err());
    }

    #[test]
    fn flame_texture_fit_profile() {
        let resolved = flame_texture_fit_resolve_from_args(&args(&[
            "bin",
            "--batch-flame-texture",
            "image.png,0.5,profile",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(resolved.0, "image.png");
        assert!((resolved.1 - 0.5).abs() < 1e-6);
        assert!(resolved.2);
    }

    #[test]
    fn anim_edit_specs_parse_all_forms() {
        let edits = anim_edits_resolve_from_args(&args(&[
            "bin",
            "--batch-anim-edit",
            "debug_keys=42",
            "--batch-anim-edit",
            "key=height@1.5=2.25",
            "--batch-anim-edit",
            "clear",
        ]))
        .unwrap();
        assert_eq!(edits[0], BatchAnimEdit::DebugKeys { seed: 42 });
        assert_eq!(
            edits[1],
            BatchAnimEdit::Key {
                property_type: crate::ecs::component::FlameParam::Height.property_type(),
                time: 1.5,
                value: 2.25
            }
        );
        assert_eq!(edits[2], BatchAnimEdit::Clear);
    }

    #[test]
    fn anim_edit_invalid_specs_are_err() {
        for spec in [
            "debug_keys=abc",
            "key=height@1.5",
            "key=no_such_param@1.0=2.0",
            "key=height@-1.0=2.0",
            "bogus",
        ] {
            assert!(
                anim_edits_resolve_from_args(&args(&["bin", "--batch-anim-edit", spec])).is_err(),
                "{spec} should be rejected"
            );
        }
    }

    #[test]
    fn debug_actions_parse_names_and_view_mode() {
        let actions = debug_actions_resolve_from_args(&args(&[
            "bin",
            "--batch-debug-action",
            "reset_camera",
            "--batch-debug-action",
            "view_mode=normal",
        ]))
        .unwrap();
        assert_eq!(actions[0], BatchDebugAction::ResetCamera);
        assert_eq!(
            actions[1],
            BatchDebugAction::ViewMode(crate::ecs::resource::DebugViewMode::Normal)
        );
        assert!(
            debug_actions_resolve_from_args(&args(&["bin", "--batch-debug-action", "bogus"]))
                .is_err()
        );
    }

    #[test]
    fn anim_edits_apply_and_dump_reflect_clip_state() {
        let mut world = World::new();
        crate::ecs::systems::spawn_flame(
            &mut world,
            crate::ecs::systems::DEFAULT_FLAME_NAME,
            FlameEffect::default(),
        );
        world.insert_resource(ClipLibrary::new());
        world.insert_resource(TimelineState::new());
        world.insert_resource(crate::ecs::resource::EditHistory::new(10));
        let mut assets = AssetStorage::new();

        batch_apply_anim_edits(
            &mut world,
            &mut assets,
            &[
                BatchAnimEdit::DebugKeys { seed: 7 },
                BatchAnimEdit::Key {
                    property_type: crate::ecs::component::FlameParam::Height.property_type(),
                    time: 9.0,
                    value: 3.5,
                },
            ],
        );
        assert!(
            (world.resource::<TimelineState>().current_time).abs() < 1e-6,
            "key edit must restore timeline time"
        );

        let dump = batch_anim_dump_json(&world);
        let entities = dump["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0]["domain"], "Flame");
        let clip_id = entities[0]["clip_id"]
            .as_u64()
            .expect("flame clip scheduled");
        let clips = dump["clips"].as_array().unwrap();
        let clip = clips
            .iter()
            .find(|c| c["id"].as_u64() == Some(clip_id))
            .expect("clip in dump");
        let curves = clip["scalar_curves"].as_array().unwrap();
        assert_eq!(
            curves.len(),
            crate::ecs::component::FLAME_DOMAIN.channels.len()
        );
        let height = curves
            .iter()
            .find(|c| c["property"] == "height")
            .expect("height curve");
        let keyframes = height["keyframes"].as_array().unwrap();
        assert_eq!(
            keyframes.len(),
            crate::ecs::systems::scalar_clip_systems::DEBUG_KEYS_PER_CURVE + 1
        );
        assert!(keyframes
            .iter()
            .any(|k| (k["time"].as_f64().unwrap() - 9.0).abs() < 1e-6
                && (k["value"].as_f64().unwrap() - 3.5).abs() < 1e-6));
        assert!((clip["duration"].as_f64().unwrap() - 9.0).abs() < 1e-6);
    }

    #[test]
    fn flame_clip_preview_parses_and_rejects_invalid() {
        let actions = debug_actions_resolve_from_args(&args(&[
            "bin",
            "--batch-debug-action",
            "flame_clip_preview=3.5",
        ]))
        .unwrap();
        assert_eq!(
            actions[0],
            BatchDebugAction::FlameClipPreview { end_seconds: 3.5 }
        );
        for bad in ["flame_clip_preview=abc", "flame_clip_preview=-1"] {
            assert!(
                debug_actions_resolve_from_args(&args(&["bin", "--batch-debug-action", bad]))
                    .is_err(),
                "{bad} should be rejected"
            );
        }
    }

    #[test]
    fn flame_clip_preview_sets_drag_preview_without_touching_instance() {
        let mut world = World::new();
        world.insert_resource(ClipLibrary::new());
        world.insert_resource(TimelineState::new());
        world.insert_resource(crate::ecs::resource::TimelineInteractionState::default());
        let mut assets = AssetStorage::new();
        let flame = crate::ecs::systems::spawn_flame_with_clip(
            &mut world,
            &mut assets,
            "Flame",
            FlameEffect::default(),
        );

        batch_apply_debug_actions(
            &world,
            &[BatchDebugAction::FlameClipPreview { end_seconds: 3.0 }],
        );

        let preview = world
            .resource::<crate::ecs::resource::TimelineInteractionState>()
            .drag_preview
            .expect("preview set");
        assert_eq!(preview.entity, flame);
        assert!((preview.start_time - 0.0).abs() < 1e-6);
        assert!((preview.end_time - 3.0).abs() < 1e-6);

        let instance = world
            .get_component::<ClipSchedule>(flame)
            .unwrap()
            .first_instance()
            .cloned()
            .unwrap();
        assert!(
            (instance.clip_out - 0.0).abs() < 1e-6,
            "preview must not commit the trim"
        );

        let dump = batch_anim_dump_json(&world);
        assert!(
            (dump["timeline"]["drag_preview"]["end_time"]
                .as_f64()
                .unwrap()
                - 3.0)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn debug_actions_apply_sets_view_mode_and_queues_events() {
        let mut world = World::new();
        world.insert_resource(DebugViewState::default());
        world.insert_resource(UIEventQueue::new());
        batch_apply_debug_actions(
            &world,
            &[
                BatchDebugAction::ViewMode(crate::ecs::resource::DebugViewMode::Normal),
                BatchDebugAction::ResetCamera,
            ],
        );
        assert_eq!(
            world.resource::<DebugViewState>().debug_view_mode,
            crate::ecs::resource::DebugViewMode::Normal
        );
        let events: Vec<UIEvent> = world.resource_mut::<UIEventQueue>().drain().collect();
        assert!(matches!(events[0], UIEvent::ResetCamera));
    }

    #[test]
    fn sequence_analyze_resolve_dir_only() {
        let args = args(&[
            "bin",
            "--batch-sequence-analyze",
            "data/flames",
            "--batch-sequence-dump",
            "out.json",
        ]);
        let result = batch_sequence_analyze_resolve_from_args(&args)
            .unwrap()
            .unwrap();
        assert_eq!(result.directories.len(), 1);
        assert_eq!(result.directories[0].0, "data/flames");
        assert_eq!(result.directories[0].1, None);
        assert_eq!(result.directories[0].2, None);
        assert_eq!(result.dump_path, "out.json");
    }

    #[test]
    fn sequence_analyze_resolve_dir_with_range() {
        let args = args(&[
            "bin",
            "--batch-sequence-analyze",
            "data/flames,5,10",
            "--batch-sequence-dump",
            "out.json",
        ]);
        let result = batch_sequence_analyze_resolve_from_args(&args)
            .unwrap()
            .unwrap();
        assert_eq!(result.directories.len(), 1);
        assert_eq!(result.directories[0].0, "data/flames");
        assert_eq!(result.directories[0].1, Some(5));
        assert_eq!(result.directories[0].2, Some(10));
    }

    #[test]
    fn sequence_analyze_resolve_multiple_dirs() {
        let args = args(&[
            "bin",
            "--batch-sequence-analyze",
            "data/a",
            "--batch-sequence-analyze",
            "data/b,1,5",
            "--batch-sequence-dump",
            "out.json",
        ]);
        let result = batch_sequence_analyze_resolve_from_args(&args)
            .unwrap()
            .unwrap();
        assert_eq!(result.directories.len(), 2);
        assert_eq!(result.directories[0].0, "data/a");
        assert_eq!(result.directories[1].0, "data/b");
        assert_eq!(result.directories[1].1, Some(1));
        assert_eq!(result.directories[1].2, Some(5));
    }

    #[test]
    fn sequence_analyze_resolve_missing_dump() {
        let args = args(&["bin", "--batch-sequence-analyze", "data/flames"]);
        let result = batch_sequence_analyze_resolve_from_args(&args);
        assert!(result.is_err());
    }

    #[test]
    fn sequence_analyze_resolve_invalid_range() {
        let args = args(&[
            "bin",
            "--batch-sequence-analyze",
            "data/flames,abc,10",
            "--batch-sequence-dump",
            "out.json",
        ]);
        let result = batch_sequence_analyze_resolve_from_args(&args);
        assert!(result.is_err());
    }

    #[test]
    fn sequence_analyze_resolve_none_without_flag() {
        let args = args(&["bin", "--batch-screenshot", "data/flames"]);
        let result = batch_sequence_analyze_resolve_from_args(&args).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn sequence_analyze_run_returns_none_without_flag() {
        let args: Vec<String> = vec![
            "bin".to_string(),
            "--batch-screenshot".to_string(),
            "data/flames".to_string(),
        ];
        let result = run_sequence_analyze_from_args(args);
        assert!(result.is_none());
    }

    #[test]
    fn sequence_analyze_end_to_end() {
        let temp_dir = tempfile::tempdir().unwrap();
        let dir_path = temp_dir.path();

        // Write meta.json with custom fps
        let meta_path = dir_path.join("meta.json");
        std::fs::write(&meta_path, r#"{"fps": 30.0}"#).unwrap();

        // Write 3 dummy 2x2 RGB PNGs with distinct colors
        for i in 0..3 {
            let value = (i + 1) as u8 * 50; // 50, 100, 150
            let png_path = dir_path.join(format!("frame_{:04}.png", i));
            write_test_png(&png_path, 2, 2, value);
        }

        let dump_path = temp_dir.path().join("output.json");
        let args = vec![
            "bin".to_string(),
            "--batch-sequence-analyze".to_string(),
            dir_path.to_string_lossy().to_string(),
            "--batch-sequence-dump".to_string(),
            dump_path.to_string_lossy().to_string(),
        ];

        let result = run_sequence_analyze_from_args(args);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.is_ok(), "sequence analysis failed: {:?}", result);

        // Verify output JSON
        let content = std::fs::read_to_string(&dump_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(json.get("sequences").is_some());
        let sequences = json["sequences"].as_array().unwrap();
        assert_eq!(sequences.len(), 1);

        let entry = &sequences[0];
        assert!(entry.get("dir").is_some());
        assert!(entry.get("descriptors").is_some());
        let descriptors = &entry["descriptors"];
        assert!(descriptors.get("f1_width").is_some());
        assert!(descriptors.get("f2_rough").is_some());
        assert!(descriptors.get("meta").is_some());

        // Verify fps from meta.json is used
        let meta = &descriptors["meta"];
        assert!((meta["fps"].as_f64().unwrap() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn sequence_analyze_range_filter() {
        let temp_dir = tempfile::tempdir().unwrap();
        let dir_path = temp_dir.path();

        // Write meta.json
        std::fs::write(dir_path.join("meta.json"), r#"{"fps": 10.0}"#).unwrap();

        // Write 5 dummy 2x2 RGB PNGs
        for i in 0..5 {
            let value = (i + 1) as u8 * 30;
            let png_path = dir_path.join(format!("frame_{:04}.png", i));
            write_test_png(&png_path, 2, 2, value);
        }

        let dump_path = temp_dir.path().join("output.json");
        let args = vec![
            "bin".to_string(),
            "--batch-sequence-analyze".to_string(),
            format!("{},1,3", dir_path.to_string_lossy()),
            "--batch-sequence-dump".to_string(),
            dump_path.to_string_lossy().to_string(),
        ];

        let result = run_sequence_analyze_from_args(args);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.is_ok(), "sequence analysis failed: {:?}", result);

        let content = std::fs::read_to_string(&dump_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&content).unwrap();
        let sequences = json["sequences"].as_array().unwrap();
        assert_eq!(sequences.len(), 1);

        // Verify frame count in meta (should be 3 frames: 1, 2, 3)
        let meta = &sequences[0]["descriptors"]["meta"];
        assert_eq!(meta["frame_count"].as_u64().unwrap(), 3);
    }

    #[test]
    fn sequence_analyze_jpg_error() {
        let temp_dir = tempfile::tempdir().unwrap();
        let dir_path = temp_dir.path();

        // Write a fake JPG file
        std::fs::write(dir_path.join("frame_0001.jpg"), b"fake jpg").unwrap();

        let dump_path = temp_dir.path().join("output.json");
        let args = vec![
            "bin".to_string(),
            "--batch-sequence-analyze".to_string(),
            dir_path.to_string_lossy().to_string(),
            "--batch-sequence-dump".to_string(),
            dump_path.to_string_lossy().to_string(),
        ];

        let result = run_sequence_analyze_from_args(args);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("JPG") || err_msg.contains("jpg"));
    }

    /// Write a simple 2x2 RGB PNG with all pixels having the same color value.
    fn write_test_png(path: &Path, width: u32, height: u32, value: u8) {
        let file = std::fs::File::create(path).unwrap();
        let writer = std::io::BufWriter::new(file);
        let mut encoder = png::Encoder::new(writer, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let mut pixels = vec![value; (width * height * 3) as usize];
        writer.write_image_data(&pixels).unwrap();
        writer.finish().unwrap();
    }
}
