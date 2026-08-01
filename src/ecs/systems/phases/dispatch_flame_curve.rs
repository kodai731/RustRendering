use crate::asset::AssetStorage;
use crate::ecs::events::UIEvent;
use crate::ecs::resource::{ClipLibrary, CurveEditorState, EditHistory, TimelineState};
use crate::ecs::systems::flame_clip_systems::{
    ensure_flame_clip, flame_clip_clear_keys, flame_clip_insert_debug_keys, flame_clip_insert_key,
};
use crate::ecs::systems::resolve_selected_flame;
use crate::ecs::world::World;
use thyllore_anim_core::editable::SourceClipId;

/// Flame keyframe events, applied to the flame entity's clip (scalar curves).
/// Undo goes through the shared `ClipModified` path, so flame edits merge and
/// revert exactly like bone clip edits.
pub fn dispatch_flame_clip_events(
    events: &[UIEvent],
    world: &mut World,
    assets: &mut AssetStorage,
) {
    for event in events {
        match event {
            UIEvent::AddFlame => {
                let flame_count = world.query_flames().len();
                if flame_count < thyllore_vulkan_core::resource::MAX_FLAME_INSTANCES {
                    let effect = crate::ecs::component::FlameEffect {
                        position: cgmath::Vector3::new(1.5 * flame_count as f32, 0.0, 0.0),
                        ..crate::ecs::component::FlameEffect::default()
                    };
                    let name = format!("Flame {}", flame_count + 1);
                    crate::ecs::systems::flame_clip_systems::spawn_flame_with_clip(
                        world, assets, &name, effect,
                    );
                }
            }
            UIEvent::InsertFlameKey { param, value } => {
                let Some((clip_id, _)) = resolve_flame_clip(world, assets) else {
                    continue;
                };
                let current_time = world
                    .get_resource::<TimelineState>()
                    .map(|t| t.current_time)
                    .unwrap_or(0.0);
                edit_clip(world, clip_id, "Flame key edit", |clip| {
                    flame_clip_insert_key(clip, *param, current_time, *value);
                });
            }
            UIEvent::InsertFlameDebugKeys { seed } => {
                let Some((clip_id, flame_entity)) = resolve_flame_clip(world, assets) else {
                    continue;
                };
                edit_clip(world, clip_id, "Flame debug keys", |clip| {
                    flame_clip_insert_debug_keys(
                        clip,
                        *seed,
                        crate::ecs::systems::TIMELINE_FALLBACK_DURATION_SECONDS,
                    );
                });
                extend_flame_instance_to_clip_duration(world, flame_entity, clip_id);
            }
            UIEvent::ClearFlameKeys => {
                let Some(clip_id) = existing_flame_clip(world) else {
                    continue;
                };
                edit_clip(world, clip_id, "Flame keys clear", |clip| {
                    flame_clip_clear_keys(clip);
                });
            }
            UIEvent::OpenFlameCurveEditor => {
                let Some((clip_id, _)) = resolve_flame_clip(world, assets) else {
                    continue;
                };
                if let Some(mut timeline) = world.get_resource_mut::<TimelineState>() {
                    timeline.current_clip_id = Some(clip_id);
                    timeline.selected_keyframes.clear();
                }
                let scalar_props: Vec<_> = world
                    .get_resource::<ClipLibrary>()
                    .and_then(|lib| {
                        lib.get(clip_id).map(|clip| {
                            clip.scalar_curves.iter().map(|c| c.property_type).collect()
                        })
                    })
                    .unwrap_or_default();
                if let Some(mut editor) = world.get_resource_mut::<CurveEditorState>() {
                    editor.is_open = true;
                    editor.needs_focus = true;
                    editor.select_scalars();
                    for prop in scalar_props {
                        editor.visible_curves.insert(prop);
                    }
                    editor.view_initialized = false;
                }
            }
            _ => continue,
        }
    }
}

fn resolve_flame_clip(world: &mut World, assets: &mut AssetStorage) -> Option<(SourceClipId, u64)> {
    let flame_entity = resolve_selected_flame(world)?;
    let clip_id = ensure_flame_clip(world, assets, flame_entity);
    Some((clip_id, flame_entity))
}

fn extend_flame_instance_to_clip_duration(
    world: &mut World,
    flame_entity: crate::ecs::world::Entity,
    clip_id: SourceClipId,
) {
    let Some(duration) = world
        .get_resource::<ClipLibrary>()
        .and_then(|lib| lib.get(clip_id).map(|clip| clip.duration))
    else {
        return;
    };
    if let Some(schedule) =
        world.get_component_mut::<crate::ecs::component::ClipSchedule>(flame_entity)
    {
        for inst in schedule
            .instances
            .iter_mut()
            .filter(|i| i.source_id == clip_id)
        {
            inst.clip_out = inst.clip_out.max(duration);
        }
    }
}

fn existing_flame_clip(world: &World) -> Option<SourceClipId> {
    let flame_entity = resolve_selected_flame(world)?;
    super::super::flame_clip_systems::find_flame_clip_id(world, flame_entity)
}

fn edit_clip(
    world: &mut World,
    clip_id: SourceClipId,
    description: &'static str,
    apply: impl FnOnce(&mut crate::animation::editable::EditableAnimationClip),
) {
    let (before, after) = {
        let mut clip_library = world.resource_mut::<ClipLibrary>();
        let Some(before) = clip_library.get(clip_id).cloned() else {
            return;
        };
        let Some(clip) = clip_library.get_mut(clip_id) else {
            return;
        };
        apply(clip);
        (before, clip.clone())
    };

    if let Some(mut history) = world.get_resource_mut::<EditHistory>() {
        crate::ecs::systems::edit_history_systems::edit_history_push_clip_mergeable(
            &mut history,
            clip_id,
            before,
            after,
            description,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::component::{FlameEffect, FlameParam};
    use crate::ecs::systems::flame_clip_systems::find_flame_clip_id;
    use crate::ecs::systems::phases::dispatch_edit_history::dispatch_edit_history_events;

    fn make_world_with_flame() -> (World, AssetStorage) {
        let mut world = World::new();
        crate::ecs::systems::spawn_flame(
            &mut world,
            crate::ecs::systems::DEFAULT_FLAME_NAME,
            FlameEffect::default(),
        );
        world.insert_resource(ClipLibrary::new());
        world.insert_resource(TimelineState::new());
        world.insert_resource(EditHistory::new(10));
        (world, AssetStorage::new())
    }

    #[test]
    fn test_insert_key_creates_clip_and_schedule() {
        let (mut world, mut assets) = make_world_with_flame();
        let entity = world.query_flames()[0];

        dispatch_flame_clip_events(
            &[UIEvent::InsertFlameKey {
                param: FlameParam::Height,
                value: 2.5,
            }],
            &mut world,
            &mut assets,
        );

        let clip_id = find_flame_clip_id(&world, entity).expect("flame clip scheduled");
        let lib = world.get_resource::<ClipLibrary>().unwrap();
        let clip = lib.get(clip_id).expect("clip registered");
        let curve = clip
            .get_scalar_curve(FlameParam::Height.property_type())
            .expect("scalar curve");
        assert_eq!(curve.keyframes.len(), 1);
        assert!((curve.keyframes[0].value - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_insert_then_undo_restores_empty_clip() {
        let (mut world, mut assets) = make_world_with_flame();
        let entity = world.query_flames()[0];

        dispatch_flame_clip_events(
            &[UIEvent::InsertFlameKey {
                param: FlameParam::Height,
                value: 2.5,
            }],
            &mut world,
            &mut assets,
        );
        let clip_id = find_flame_clip_id(&world, entity).unwrap();

        dispatch_edit_history_events(&[UIEvent::Undo], &mut world);
        {
            let lib = world.get_resource::<ClipLibrary>().unwrap();
            assert!(lib.get(clip_id).unwrap().scalar_curves.is_empty());
        }

        dispatch_edit_history_events(&[UIEvent::Redo], &mut world);
        let lib = world.get_resource::<ClipLibrary>().unwrap();
        let clip = lib.get(clip_id).unwrap();
        assert_eq!(clip.scalar_curves.len(), 1);
    }

    #[test]
    fn test_add_flame_creates_clip_and_schedule_by_default() {
        let (mut world, mut assets) = make_world_with_flame();

        dispatch_flame_clip_events(&[UIEvent::AddFlame], &mut world, &mut assets);

        let flames = world.query_flames();
        assert_eq!(flames.len(), 2);
        let new_flame = flames[1];
        let clip_id =
            find_flame_clip_id(&world, new_flame).expect("new flame has a scheduled clip");
        let lib = world.get_resource::<ClipLibrary>().unwrap();
        let clip = lib.get(clip_id).expect("clip registered");
        assert_eq!(
            clip.name,
            crate::ecs::systems::flame_clip_systems::FLAME_CLIP_NAME
        );
        assert!(clip.scalar_curves.is_empty());
    }

    #[test]
    fn test_spawn_flame_with_clip_schedules_each_flame_separately() {
        let mut world = World::new();
        world.insert_resource(ClipLibrary::new());
        let mut assets = AssetStorage::new();

        let a = crate::ecs::systems::flame_clip_systems::spawn_flame_with_clip(
            &mut world,
            &mut assets,
            "Flame",
            FlameEffect::default(),
        );
        let b = crate::ecs::systems::flame_clip_systems::spawn_flame_with_clip(
            &mut world,
            &mut assets,
            "Flame 2",
            FlameEffect::default(),
        );

        let clip_a = find_flame_clip_id(&world, a).unwrap();
        let clip_b = find_flame_clip_id(&world, b).unwrap();
        assert_ne!(clip_a, clip_b, "each flame owns its own clip");
    }
}
