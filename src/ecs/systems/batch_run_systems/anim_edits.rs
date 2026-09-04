use std::path::Path;

use anyhow::{bail, Context, Result};

use thyllore_anim_core::editable::PropertyType;

use crate::asset::AssetStorage;
use crate::ecs::component::{
    scalar_channel_domains, scalar_channel_for_cli_name, scalar_channel_for_property,
    scalar_cli_names_joined, ClipSchedule,
};
use crate::ecs::events::UIEvent;
use crate::ecs::resource::{ClipLibrary, TimelineState};
use crate::ecs::world::World;

use super::BATCH_ANIM_EDIT_FLAG;

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

/// Parse repeated `--batch-anim-edit <spec>` flags. Specs:
/// `debug_keys=<seed>` | `key=<param>@<time>=<value>` | `clear`.
pub(super) fn anim_edits_resolve_from_args(args: &[String]) -> Result<Vec<BatchAnimEdit>> {
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

pub(super) fn anim_edit_parse_spec(spec: &str) -> Result<BatchAnimEdit> {
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
    use crate::ecs::systems::phases::dispatch_scalar_curve::dispatch_scalar_clip_events;
    use crate::ecs::systems::scalar_clip_systems::{
        ensure_entity_clip, resolve_selected_scalar_entity,
    };

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
                crate::ecs::systems::timeline_systems::process_clip_instance_events(
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

/// Serialize the animation-facing world state (flames, their scheduled clips,
/// every clip's scalar curves, timeline) so agents can inspect edits without a
/// window. Written once at engine exit; the file is the access surface.
pub fn batch_anim_dump_json(world: &World) -> serde_json::Value {
    use crate::ecs::systems::scalar_clip_systems::find_entity_clip_id;

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
