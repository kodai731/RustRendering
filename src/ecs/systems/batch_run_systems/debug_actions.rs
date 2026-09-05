use anyhow::{bail, Result};

use crate::ecs::component::ClipSchedule;
use crate::ecs::world::World;

use super::batch_action::{batch_action_registry, BatchAction};
use super::BATCH_DEBUG_ACTION_FLAG;

pub(super) fn debug_actions_resolve_from_args(
    args: &[String],
) -> Result<Vec<Box<dyn BatchAction>>> {
    let mut actions = Vec::new();
    for i in 0..args.len() {
        if args[i] != BATCH_DEBUG_ACTION_FLAG {
            continue;
        }
        let Some(name) = args.get(i + 1).filter(|v| !v.starts_with("--")) else {
            bail!(
                "{BATCH_DEBUG_ACTION_FLAG} requires an action. Valid actions: {}",
                batch_action_registry()
                    .iter()
                    .map(|d| d.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        };
        actions.push(debug_action_parse(name)?);
    }
    Ok(actions)
}

pub(super) fn debug_action_parse(name: &str) -> Result<Box<dyn BatchAction>> {
    let name = name.trim();
    for descriptor in batch_action_registry() {
        if let Some(result) = (descriptor.parse)(name) {
            return result;
        }
    }
    bail!(
        "unknown debug action '{name}'. Valid actions: {}",
        batch_action_registry()
            .iter()
            .map(|d| d.name)
            .collect::<Vec<_>>()
            .join(", ")
    )
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
pub fn batch_apply_debug_actions(world: &World, actions: &[&dyn BatchAction]) {
    for a in actions {
        a.apply(world);
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
    let names: Vec<&str> = batch_action_registry().iter().map(|d| d.name).collect();
    serde_json::json!({"ok": true, "actions": names}).to_string()
}
