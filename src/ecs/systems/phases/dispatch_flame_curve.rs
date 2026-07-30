use crate::ecs::component::{channel_insert_key, FlameChannel, FlameParam, FlameTrack};
use crate::ecs::events::UIEvent;
use crate::ecs::resource::{
    EditCommand, EditCommandAfter, EditEntry, EditHistory, SelectedFlameInstance,
};
use crate::ecs::world::World;
use thyllore_anim_core::editable::{BezierHandle, InterpolationType, KeyframeId};

pub fn dispatch_flame_curve_events(events: &[UIEvent], world: &mut World) {
    for event in events {
        let target_entity = match resolve_target_entity(world) {
            Some(e) => e,
            None => continue,
        };

        let before = world
            .get_component::<FlameTrack>(target_entity)
            .cloned()
            .unwrap_or_default();

        match event {
            UIEvent::FlameCurveAddKey { param, time, value } => {
                apply_add_key(world, target_entity, *param, *time, *value);
            }
            UIEvent::FlameCurveMoveKey {
                param,
                keyframe_id,
                new_time,
                new_value,
            } => {
                apply_move_key(
                    world,
                    target_entity,
                    *param,
                    *keyframe_id,
                    *new_time,
                    *new_value,
                );
            }
            UIEvent::FlameCurveDeleteKey { param, keyframe_id } => {
                apply_delete_key(world, target_entity, *param, *keyframe_id);
            }
            UIEvent::FlameCurveSetInterpolation {
                param,
                keyframe_id,
                interpolation,
            } => {
                apply_set_interpolation(world, target_entity, *param, *keyframe_id, *interpolation);
            }
            UIEvent::FlameCurveSetTangent {
                param,
                keyframe_id,
                in_tangent,
                out_tangent,
            } => {
                apply_set_tangent(
                    world,
                    target_entity,
                    *param,
                    *keyframe_id,
                    in_tangent.clone(),
                    out_tangent.clone(),
                );
            }
            _ => continue,
        }

        // Push edit history entry if EditHistory resource exists
        if let Some(mut edit_history) = world.get_resource_mut::<EditHistory>() {
            let after = world
                .get_component::<FlameTrack>(target_entity)
                .cloned()
                .unwrap_or_default();
            let entry = EditEntry {
                command: EditCommand::FlameTrackModified {
                    entity: target_entity,
                    before,
                    after,
                    description: "Flame curve edit",
                },
                after: EditCommandAfter::Empty,
            };
            edit_history.push_to_undo(entry);
        }
    }
}

fn resolve_target_entity(world: &World) -> Option<u64> {
    let flames = world.query_flames();
    if flames.is_empty() {
        return None;
    }
    let selected = world
        .get_resource::<SelectedFlameInstance>()
        .map(|s| s.0)
        .unwrap_or(0);
    let idx = selected.min(flames.len() - 1);
    Some(flames[idx])
}

fn apply_add_key(world: &mut World, entity: u64, param: FlameParam, time: f32, value: f32) {
    let mut track = match world.get_component_mut::<FlameTrack>(entity) {
        Some(existing) => {
            let mut track = existing.clone();
            drop(existing);
            track
        }
        None => FlameTrack::default(),
    };

    let mut found = false;
    for channel in &mut track.channels {
        if channel.param == param {
            channel_insert_key(channel, time, value, InterpolationType::Linear);
            found = true;
            break;
        }
    }
    if !found {
        let mut channel = FlameChannel {
            param,
            keys: vec![],
            next_keyframe_id: 1,
        };
        channel_insert_key(&mut channel, time, value, InterpolationType::Linear);
        track.channels.push(channel);
    }

    world.insert_component(entity, track);
}

fn apply_move_key(
    world: &mut World,
    entity: u64,
    param: FlameParam,
    keyframe_id: KeyframeId,
    new_time: f32,
    new_value: f32,
) {
    let mut track = match world.get_component_mut::<FlameTrack>(entity) {
        Some(existing) => {
            let mut track = existing.clone();
            drop(existing);
            track
        }
        None => return,
    };

    for channel in &mut track.channels {
        if channel.param == param {
            for key in &mut channel.keys {
                if key.id == keyframe_id {
                    key.time = new_time;
                    key.value = new_value;
                    break;
                }
            }
            channel
                .keys
                .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            break;
        }
    }

    world.insert_component(entity, track);
}

fn apply_delete_key(world: &mut World, entity: u64, param: FlameParam, keyframe_id: KeyframeId) {
    let mut track = match world.get_component_mut::<FlameTrack>(entity) {
        Some(existing) => {
            let mut track = existing.clone();
            drop(existing);
            track
        }
        None => return,
    };

    let mut found_channel = false;
    let mut to_remove: Vec<usize> = Vec::new();
    for (i, channel) in track.channels.iter_mut().enumerate() {
        if channel.param == param {
            channel.keys.retain(|key| key.id != keyframe_id);
            found_channel = true;
        }
        if channel.keys.is_empty() {
            to_remove.push(i);
        }
    }
    for i in to_remove.into_iter().rev() {
        track.channels.remove(i);
    }

    world.insert_component(entity, track);
}

fn apply_set_interpolation(
    world: &mut World,
    entity: u64,
    param: FlameParam,
    keyframe_id: KeyframeId,
    interpolation: InterpolationType,
) {
    let mut track = match world.get_component_mut::<FlameTrack>(entity) {
        Some(existing) => {
            let mut track = existing.clone();
            drop(existing);
            track
        }
        None => return,
    };

    for channel in &mut track.channels {
        if channel.param == param {
            for key in &mut channel.keys {
                if key.id == keyframe_id {
                    key.interpolation = interpolation;
                    break;
                }
            }
            break;
        }
    }

    world.insert_component(entity, track);
}

fn apply_set_tangent(
    world: &mut World,
    entity: u64,
    param: FlameParam,
    keyframe_id: KeyframeId,
    in_tangent: BezierHandle,
    out_tangent: BezierHandle,
) {
    let mut track = match world.get_component_mut::<FlameTrack>(entity) {
        Some(existing) => {
            let mut track = existing.clone();
            drop(existing);
            track
        }
        None => return,
    };

    for channel in &mut track.channels {
        if channel.param == param {
            for key in &mut channel.keys {
                if key.id == keyframe_id {
                    key.in_tangent = in_tangent;
                    key.out_tangent = out_tangent;
                    break;
                }
            }
            break;
        }
    }

    world.insert_component(entity, track);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::systems::phases::dispatch_edit_history::dispatch_edit_history_events;

    fn make_world_with_flame() -> World {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert_component(entity, crate::ecs::resource::FlameEffect::default());
        world.insert_component(
            entity,
            FlameTrack {
                channels: vec![FlameChannel {
                    param: FlameParam::Height,
                    keys: vec![],
                    next_keyframe_id: 1,
                }],
            },
        );
        world
    }

    #[test]
    fn test_add_move_undo_redo() {
        let mut world = make_world_with_flame();
        let entity = world.query_flames()[0];

        // Insert EditHistory resource for undo/redo
        world.insert_resource(EditHistory::new(10));

        // Add a key at time=1.0, value=2.0 via dispatch
        dispatch_flame_curve_events(
            &[UIEvent::FlameCurveAddKey {
                param: FlameParam::Height,
                time: 1.0,
                value: 2.0,
            }],
            &mut world,
        );

        // Get the actual keyframe id from the track
        let track = world.get_component::<FlameTrack>(entity).unwrap();
        let id = track.channels[0].keys[0].id;

        // Verify add was applied
        let keys: Vec<_> = track.channels[0].keys.iter().collect();
        assert_eq!(keys.len(), 1);
        assert!((keys[0].time - 1.0).abs() < 1e-6);
        assert!((keys[0].value - 2.0).abs() < 1e-6);

        // Move the key to time=3.0, value=4.0 via dispatch
        dispatch_flame_curve_events(
            &[UIEvent::FlameCurveMoveKey {
                param: FlameParam::Height,
                keyframe_id: id,
                new_time: 3.0,
                new_value: 4.0,
            }],
            &mut world,
        );

        // Verify move was applied
        let track = world.get_component::<FlameTrack>(entity).unwrap();
        let keys: Vec<_> = track.channels[0].keys.iter().collect();
        assert_eq!(keys.len(), 1);
        assert!((keys[0].time - 3.0).abs() < 1e-6);
        assert!((keys[0].value - 4.0).abs() < 1e-6);

        // Undo: should restore to after_add state (move undone)
        dispatch_edit_history_events(&[UIEvent::Undo], &mut world);
        let track = world.get_component::<FlameTrack>(entity).unwrap();
        let keys: Vec<_> = track.channels[0].keys.iter().collect();
        assert_eq!(keys.len(), 1);
        assert!((keys[0].time - 1.0).abs() < 1e-6);
        assert!((keys[0].value - 2.0).abs() < 1e-6);

        // Redo: should reapply move
        dispatch_edit_history_events(&[UIEvent::Redo], &mut world);
        let track = world.get_component::<FlameTrack>(entity).unwrap();
        let keys: Vec<_> = track.channels[0].keys.iter().collect();
        assert_eq!(keys.len(), 1);
        assert!((keys[0].time - 3.0).abs() < 1e-6);
        assert!((keys[0].value - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_delete_undo_restoration() {
        let mut world = make_world_with_flame();
        let entity = world.query_flames()[0];

        // Insert EditHistory resource for undo/redo
        world.insert_resource(EditHistory::new(10));

        // Add a key at time=1.0, value=2.0 via dispatch
        dispatch_flame_curve_events(
            &[UIEvent::FlameCurveAddKey {
                param: FlameParam::Height,
                time: 1.0,
                value: 2.0,
            }],
            &mut world,
        );

        // Get the actual keyframe id from the track
        let track = world.get_component::<FlameTrack>(entity).unwrap();
        let id = track.channels[0].keys[0].id;

        // Verify key exists
        assert_eq!(track.channels[0].keys.len(), 1);
        assert!((track.channels[0].keys[0].time - 1.0).abs() < 1e-6);

        // Delete the key via dispatch
        dispatch_flame_curve_events(
            &[UIEvent::FlameCurveDeleteKey {
                param: FlameParam::Height,
                keyframe_id: id,
            }],
            &mut world,
        );

        // Verify key was deleted (channel removed since empty)
        let track = world.get_component::<FlameTrack>(entity).unwrap();
        assert!(track.channels.is_empty());

        // Undo: should restore to before_delete state
        dispatch_edit_history_events(&[UIEvent::Undo], &mut world);
        let track = world.get_component::<FlameTrack>(entity).unwrap();

        // Verify restoration
        assert_eq!(track.channels.len(), 1);
        assert_eq!(track.channels[0].keys.len(), 1);
        assert!((track.channels[0].keys[0].time - 1.0).abs() < 1e-6);
        assert!((track.channels[0].keys[0].value - 2.0).abs() < 1e-6);
    }
}
