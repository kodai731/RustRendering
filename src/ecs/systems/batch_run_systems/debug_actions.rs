use anyhow::{bail, Result};

use crate::ecs::component::{ClipSchedule, FlameEffect};
use crate::ecs::events::{DebugPrimitiveKind, UIEvent, UIEventQueue};
use crate::ecs::resource::{DebugViewMode, DebugViewState};
use crate::ecs::world::World;

use super::{
    apply_texture_fit_from_path, parse_texture_fit_args, BATCH_DEBUG_ACTION_FLAG,
    DEBUG_ACTION_NAMES,
};

#[derive(Clone, Debug, PartialEq)]
pub enum BatchDebugAction {
    ResetCamera,
    ResetCameraUp,
    CameraToModel,
    AddFlame,
    OpenFlameCurves,
    ViewMode(DebugViewMode),
    BlackBackground,
    FlameClipPreview {
        end_seconds: f32,
    },
    TimelineSelectFlameClip,
    WallProbeDump,
    WaterDebugDump,
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
    SpawnDebugPrimitive {
        kind: DebugPrimitiveKind,
    },
}

pub(super) fn debug_view_mode_parse(name: &str) -> Option<DebugViewMode> {
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

pub(super) fn debug_actions_resolve_from_args(args: &[String]) -> Result<Vec<BatchDebugAction>> {
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

pub(super) fn debug_action_parse(name: &str) -> Result<BatchDebugAction> {
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
        "black_background" => Ok(BatchDebugAction::BlackBackground),
        "reset_camera" => Ok(BatchDebugAction::ResetCamera),
        "reset_camera_up" => Ok(BatchDebugAction::ResetCameraUp),
        "camera_to_model" => Ok(BatchDebugAction::CameraToModel),
        "add_flame" => Ok(BatchDebugAction::AddFlame),
        "open_flame_curves" => Ok(BatchDebugAction::OpenFlameCurves),
        "dump_wall_probe" => Ok(BatchDebugAction::WallProbeDump),
        "dump_water_debug" => Ok(BatchDebugAction::WaterDebugDump),
        "spawn_cube" => Ok(BatchDebugAction::SpawnDebugPrimitive {
            kind: DebugPrimitiveKind::Cube,
        }),
        "spawn_sphere" => Ok(BatchDebugAction::SpawnDebugPrimitive {
            kind: DebugPrimitiveKind::Sphere,
        }),
        "spawn_floor" => Ok(BatchDebugAction::SpawnDebugPrimitive {
            kind: DebugPrimitiveKind::Floor,
        }),
        _ => bail!(
            "unknown debug action '{name}'. Valid actions: {}",
            DEBUG_ACTION_NAMES.join(", ")
        ),
    }
}

/// Check if `--batch-debug-action dump_wall_probe` is present in the args.
pub(super) fn debug_actions_has_wall_probe_dump(args: &[String]) -> bool {
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

/// Check if `--batch-debug-action dump_water_debug` is present in the args.
pub(super) fn debug_actions_has_water_debug_dump(args: &[String]) -> bool {
    for i in 0..args.len() {
        if args[i] == BATCH_DEBUG_ACTION_FLAG {
            if let Some(name) = args.get(i + 1).filter(|v| !v.starts_with("--")) {
                if name == "dump_water_debug" {
                    return true;
                }
            }
        }
    }
    false
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
            BatchDebugAction::BlackBackground => {
                world.resource_mut::<DebugViewState>().black_background = true;
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
            BatchDebugAction::WaterDebugDump => {
                world
                    .resource_mut::<UIEventQueue>()
                    .send(UIEvent::DumpWaterDebug);
            }
            BatchDebugAction::TimelineSelectFlameClip => {
                let clip_id = world.query_flames().first().and_then(|&flame| {
                    crate::ecs::systems::scalar_clip_systems::find_entity_clip_id(world, flame)
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
            BatchDebugAction::SpawnDebugPrimitive { kind } => {
                world
                    .resource_mut::<UIEventQueue>()
                    .send(UIEvent::SpawnDebugPrimitive { kind: *kind });
            }
        }
    }
}

/// Make the timeline draw the first flame's clip block as if a TrimEnd drag to
/// `end_seconds` were in progress: same preview math as the live drag, but no
/// commit event, so the underlying instance stays untouched.
pub(super) fn apply_flame_clip_preview(world: &World, end_seconds: f32) {
    let Some(&flame) = world.query_flames().first() else {
        return;
    };
    let Some(instance) = world
        .get_component::<ClipSchedule>(flame)
        .and_then(|schedule| schedule.first_instance().cloned())
    else {
        return;
    };

    let (start_time, end_time) = crate::ecs::systems::timeline_systems::clip_drag_preview_times(
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
