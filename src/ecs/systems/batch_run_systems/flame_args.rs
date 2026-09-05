use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::json;

use crate::ecs::component::{ClipSchedule, FlameBaked, FlameEffect};
use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::FlameShadingMode;
use crate::ecs::world::World;

use super::batch_action::BatchAction;

use super::{
    BATCH_FLAME_BONE_FLAG, BATCH_FLAME_COUNT_FLAG, BATCH_FLAME_DEBUG_VIEW_FLAG,
    BATCH_FLAME_MODE_FLAG, BATCH_FLAME_MOTION_FLAG, BATCH_FLAME_ORBIT_FLAG,
    BATCH_FLAME_PRESET_FLAG, BATCH_FLAME_SDF_FLAG, BATCH_FLAME_SET_FLAG, BATCH_FLAME_STEPS_FLAG,
    BATCH_FLAME_STYLE_FLAG, BATCH_FLAME_TEXTURE_FLAG, BATCH_FLAME_TRAIL_FLAG,
    BATCH_HEAT_PLUME_FLAG, FLAME_DUMP_FLAG, ROT_Z_DEG_KEY,
};

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

pub fn flame_dump_npy_path(json_path: &Path) -> PathBuf {
    let mut npy = json_path.to_path_buf();
    if npy.extension().is_some() {
        npy.set_extension("npy");
    }
    npy
}

#[test]
fn test_flame_dump_npy_path() {
    let json = Path::new("/tmp/flame_dump_frame_0001.json");
    let npy = flame_dump_npy_path(json);
    assert_eq!(npy, PathBuf::from("/tmp/flame_dump_frame_0001.npy"));

    let jsonl = Path::new("/tmp/flame_dump_frame_0001.jsonl");
    let npy = flame_dump_npy_path(jsonl);
    assert_eq!(npy, PathBuf::from("/tmp/flame_dump_frame_0001.npy"));
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

pub(super) fn flame_set_valid_keys() -> Vec<&'static str> {
    thyllore_effect_core::FLAME_SCALAR_PARAMS
        .iter()
        .map(|param| param.name)
        .chain([ROT_Z_DEG_KEY])
        .collect()
}

pub(super) fn flame_set_resolve_from_args(args: &[String]) -> Result<Vec<(String, f32)>> {
    let valid_keys = flame_set_valid_keys();

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

pub(super) fn flame_preset_resolve_from_args(args: &[String]) -> Result<Option<String>> {
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

pub(super) fn flame_style_resolve_from_args(
    args: &[String],
) -> Result<Option<(String, thyllore_effect_core::StyleGroups)>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_FLAME_STYLE_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1).filter(|v| !v.starts_with("--")) else {
        bail!("{BATCH_FLAME_STYLE_FLAG} requires <path>[,motion][,texture][,optics]");
    };

    let mut parts = value.split(',');
    let path = parts.next().unwrap_or_default().trim().to_string();
    if path.is_empty() {
        bail!("{BATCH_FLAME_STYLE_FLAG} requires a non-empty path");
    }

    let group_names: Vec<&str> = parts.map(str::trim).collect();
    if group_names.is_empty() {
        return Ok(Some((path, thyllore_effect_core::StyleGroups::default())));
    }
    let mut groups = thyllore_effect_core::StyleGroups {
        motion: false,
        texture: false,
        optics: false,
    };
    for name in group_names {
        match name {
            "motion" => groups.motion = true,
            "texture" => groups.texture = true,
            "optics" => groups.optics = true,
            other => bail!(
                "unknown {BATCH_FLAME_STYLE_FLAG} group '{}'. Valid groups: motion, texture, optics",
                other
            ),
        }
    }
    Ok(Some((path, groups)))
}

pub fn load_flame_style_from_path(path: &str) -> Option<thyllore_effect_core::FlameStyle> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("warning: failed to read flame style '{}': {}", path, e);
            return None;
        }
    };
    match ron::from_str(&content) {
        Ok(style) => Some(style),
        Err(e) => {
            eprintln!("warning: failed to parse flame style '{}': {}", path, e);
            None
        }
    }
}

pub fn apply_flame_style_from_path(
    effect: &mut FlameEffect,
    path: &str,
    groups: thyllore_effect_core::StyleGroups,
) -> Option<thyllore_effect_core::FlameStyle> {
    let style = load_flame_style_from_path(path)?;
    thyllore_effect_core::apply_flame_style(effect, &style, groups);
    Some(style)
}

pub fn dump_flame_style_to_path(effect: &FlameEffect, path: &str) {
    let name = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("style")
        .trim_end_matches(".style.ron")
        .trim_end_matches(".ron")
        .to_string();
    let style = thyllore_effect_core::flame_style_from_effect(effect, &name);
    let content = match ron::ser::to_string_pretty(&style, ron::ser::PrettyConfig::default()) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("warning: failed to serialize flame style: {}", e);
            return;
        }
    };
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(path, content) {
        eprintln!("warning: failed to write flame style '{}': {}", path, e);
    }
}

pub(super) fn flame_texture_fit_resolve_from_args(
    args: &[String],
) -> Result<Option<(String, f32, bool)>> {
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

pub(super) fn heat_plume_resolve_from_args(args: &[String]) -> Result<Option<(f32, f32)>> {
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

pub(super) fn flame_trail_resolve_from_args(args: &[String]) -> Result<Option<f32>> {
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

pub(super) fn flame_orbit_resolve_from_args(args: &[String]) -> Result<Option<(f32, f32)>> {
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

pub(super) fn flame_motion_resolve_from_args(args: &[String]) -> Result<Option<(f32, f32)>> {
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

pub(super) fn flame_bone_resolve_from_args(args: &[String]) -> Result<Option<String>> {
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

pub(super) fn flame_sdf_resolve_from_args(args: &[String]) -> Result<Option<String>> {
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

pub fn apply_flame_overrides(effect: &mut FlameEffect, overrides: &[(String, f32)]) {
    for (key, value) in overrides {
        if key == ROT_Z_DEG_KEY {
            effect.rotation = cgmath::Quaternion::from(cgmath::Euler::new(
                cgmath::Deg(0.0),
                cgmath::Deg(0.0),
                cgmath::Deg(*value),
            ));
            continue;
        }

        let param =
            thyllore_effect_core::find_scalar_param(thyllore_effect_core::FLAME_SCALAR_PARAMS, key)
                .unwrap_or_else(|| unreachable!("unknown key (parser should have rejected)"));
        (param.set)(effect, *value);
    }
}

pub fn batch_run_flame_dump(
    world: &World,
    flame_trace_path: Option<&std::path::Path>,
    wall_probe_path: Option<&std::path::Path>,
) {
    use crate::ecs::systems::camera_systems::{
        compute_camera_direction, compute_camera_position, compute_camera_right, compute_camera_up,
    };

    let camera = (*world.resource::<crate::ecs::resource::Camera>()).clone();
    let settings = world
        .get_resource::<crate::ecs::resource::FlameRenderSettings>()
        .map(|s| *s)
        .unwrap_or_default();
    let view = thyllore_effect_core::WallProbeView {
        position: compute_camera_position(&camera).into(),
        forward: compute_camera_direction(&camera).into(),
        right: compute_camera_right(&camera).into(),
        up: compute_camera_up(&camera).into(),
        fov_y_radians: camera.fov_y.0.to_radians(),
        viewport_size_px: [1680.0, 840.0],
    };

    let flames: Vec<_> = world
        .query_flames()
        .into_iter()
        .filter_map(|entity| {
            let effect = world.get_component::<crate::ecs::component::FlameEffect>(entity)?;
            let baked = world
                .get_component::<crate::ecs::component::FlameBaked>(entity)
                .cloned()
                .unwrap_or_default();
            let temporal = world
                .get_component::<crate::ecs::component::FlameTemporalAccum>(entity)
                .cloned()
                .unwrap_or_default();
            let report = thyllore_effect_core::probe_flame_wall(&effect, &baked, &view);
            Some((effect.clone(), baked, temporal, report))
        })
        .collect();
    if flames.is_empty() {
        log_warn!("batch flame dump skipped: no flame entity");
        return;
    }

    if let Some(path) = wall_probe_path {
        match crate::ecs::systems::flame_dump_systems::write_flame_wall_probe_dump(
            &camera,
            &settings,
            [1680.0, 840.0],
            &flames,
            Some(path),
        ) {
            Ok(p) => log!("wall probe dumped to {}", p.display()),
            Err(e) => log_warn!("wall probe dump failed: {}", e),
        }
    }

    if let Some(path) = flame_trace_path {
        match crate::ecs::systems::flame_dump_systems::write_flame_field_traces(
            &view,
            &flames,
            Some(path),
        ) {
            Ok(paths) => {
                for p in paths {
                    log!("flame field trace dumped to {}", p.display());
                }
            }
            Err(e) => log_warn!("flame field trace dump failed: {}", e),
        }
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

pub(super) fn parse_texture_fit_args(rest: &str) -> Result<(String, f32, bool)> {
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

#[derive(Debug)]
pub struct AddFlame;

#[derive(Debug)]
pub struct OpenFlameCurves;

#[derive(Debug)]
pub struct FlameClipPreview {
    pub end_seconds: f32,
}

#[derive(Debug)]
pub struct TimelineSelectFlameClip;

#[derive(Debug)]
pub struct ApplyTextureFit {
    pub path: String,
    pub blend: f32,
    pub profile: bool,
}

#[derive(Debug)]
pub struct ApplyTextureFitRoundtrip {
    pub path: String,
    pub blend: f32,
    pub profile: bool,
}

impl BatchAction for AddFlame {
    fn name(&self) -> &'static str {
        "add_flame"
    }
    fn apply(&self, world: &World) {
        world.resource_mut::<UIEventQueue>().send(UIEvent::AddFlame);
    }
}

impl BatchAction for OpenFlameCurves {
    fn name(&self) -> &'static str {
        "open_flame_curves"
    }
    fn apply(&self, world: &World) {
        world
            .resource_mut::<UIEventQueue>()
            .send(UIEvent::OpenScalarCurveEditor);
    }
}

impl BatchAction for FlameClipPreview {
    fn name(&self) -> &'static str {
        "flame_clip_preview"
    }
    fn apply(&self, world: &World) {
        super::debug_actions::apply_flame_clip_preview(world, self.end_seconds);
    }
}

impl BatchAction for TimelineSelectFlameClip {
    fn name(&self) -> &'static str {
        "timeline_select_flame_clip"
    }
    fn apply(&self, world: &World) {
        let clip_id = world.query_flames().first().and_then(|&flame| {
            crate::ecs::systems::scalar_clip_systems::find_entity_clip_id(world, flame)
        });
        if let Some(clip_id) = clip_id {
            world
                .resource_mut::<UIEventQueue>()
                .send(UIEvent::TimelineSelectClip(clip_id));
        }
    }
}

fn apply_texture_fit_effect(world: &World, path: &str, blend: f32, profile: bool) {
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
            blend,
            thyllore_effect_core::TextureFitGroups::default(),
            profile,
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

impl BatchAction for ApplyTextureFit {
    fn name(&self) -> &'static str {
        "apply_texture_fit"
    }
    fn apply(&self, world: &World) {
        apply_texture_fit_effect(world, &self.path, self.blend, self.profile);
    }
}

impl BatchAction for ApplyTextureFitRoundtrip {
    fn name(&self) -> &'static str {
        "apply_texture_fit_roundtrip"
    }
    fn apply(&self, world: &World) {
        let original = world.query_flames().first().and_then(|&flame| {
            let effect = world.get_component::<FlameEffect>(flame)?.clone();
            let baked = world
                .get_component::<crate::ecs::component::FlameBaked>(flame)
                .cloned()
                .unwrap_or_default();
            Some((effect, baked))
        });
        if let Some((original_effect, original_baked)) = original {
            apply_texture_fit_effect(world, &self.path, self.blend, self.profile);
            world
                .resource_mut::<UIEventQueue>()
                .send(UIEvent::UpdateFlameEffect(Box::new(original_effect)));
            world
                .resource_mut::<UIEventQueue>()
                .send(UIEvent::UpdateFlameBaked(Box::new(original_baked)));
        }
    }
}

fn parse_clip_preview_seconds(text: &str) -> Result<f32> {
    let end_seconds: f32 = text
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid flame_clip_preview seconds '{text}'"))?;
    if !end_seconds.is_finite() || end_seconds < 0.0 {
        bail!("flame_clip_preview seconds must be >= 0 and finite: '{text}'");
    }
    Ok(end_seconds)
}

fn parse_flame_clip_preview(s: &str) -> Option<Result<Box<dyn BatchAction>>> {
    let seconds_str = s.strip_prefix("flame_clip_preview=")?.trim();
    Some(
        parse_clip_preview_seconds(seconds_str)
            .map(|end_seconds| Box::new(FlameClipPreview { end_seconds }) as Box<dyn BatchAction>),
    )
}

fn parse_apply_texture_fit(s: &str) -> Option<Result<Box<dyn BatchAction>>> {
    let rest = s.strip_prefix("apply_texture_fit:")?;
    Some(parse_texture_fit_args(rest).map(|(path, blend, profile)| {
        Box::new(ApplyTextureFit {
            path,
            blend,
            profile,
        }) as Box<dyn BatchAction>
    }))
}

fn parse_apply_texture_fit_roundtrip(s: &str) -> Option<Result<Box<dyn BatchAction>>> {
    let rest = s.strip_prefix("apply_texture_fit_roundtrip:")?;
    Some(parse_texture_fit_args(rest).map(|(path, blend, profile)| {
        Box::new(ApplyTextureFitRoundtrip {
            path,
            blend,
            profile,
        }) as Box<dyn BatchAction>
    }))
}

pub fn flame_action_descriptors() -> Vec<super::batch_action::BatchActionDescriptor> {
    use super::batch_action::BatchActionDescriptor;
    vec![
        BatchActionDescriptor {
            name: "add_flame",
            parse: |s| (s == "add_flame").then(|| Ok(Box::new(AddFlame) as Box<dyn BatchAction>)),
        },
        BatchActionDescriptor {
            name: "open_flame_curves",
            parse: |s| {
                (s == "open_flame_curves")
                    .then(|| Ok(Box::new(OpenFlameCurves) as Box<dyn BatchAction>))
            },
        },
        BatchActionDescriptor {
            name: "flame_clip_preview",
            parse: parse_flame_clip_preview,
        },
        BatchActionDescriptor {
            name: "timeline_select_flame_clip",
            parse: |s| {
                (s == "timeline_select_flame_clip")
                    .then(|| Ok(Box::new(TimelineSelectFlameClip) as Box<dyn BatchAction>))
            },
        },
        BatchActionDescriptor {
            name: "apply_texture_fit",
            parse: parse_apply_texture_fit,
        },
        BatchActionDescriptor {
            name: "apply_texture_fit_roundtrip",
            parse: parse_apply_texture_fit_roundtrip,
        },
    ]
}
