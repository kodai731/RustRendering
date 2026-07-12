use crate::animation::editable::SourceClipId;
use crate::asset::AssetStorage;
use crate::ecs::component::ClipSchedule;
use crate::ecs::events::UIEvent;
use crate::ecs::resource::{ClipLibrary, EditHistory};
use crate::ecs::world::World;

pub fn dispatch_clip_browser_ecs_events(
    events: &[UIEvent],
    world: &mut World,
    assets: &mut AssetStorage,
) {
    for event in events {
        match event {
            UIEvent::ClipInstanceAdd {
                entity,
                source_id,
                start_time,
            } => {
                let duration = {
                    let clip_library = world.resource::<ClipLibrary>();
                    clip_library
                        .get(*source_id)
                        .map(|c| c.duration)
                        .unwrap_or(1.0)
                };

                if let Some(schedule) = world.get_component_mut::<ClipSchedule>(*entity) {
                    crate::ecs::systems::clip_schedule_systems::clip_schedule_add_instance(
                        schedule, *source_id, duration,
                    );

                    if let Some(last) = schedule.instances.last_mut() {
                        last.start_time = *start_time;
                    }
                }
            }

            UIEvent::ClipBrowserCreateEmpty => {
                let mut clip_library = world.resource_mut::<ClipLibrary>();
                let editable = crate::animation::editable::EditableAnimationClip::new(
                    0,
                    "New Clip".to_string(),
                );
                let id =
                    crate::ecs::systems::clip_library_systems::clip_library_register_and_activate(
                        &mut clip_library,
                        assets,
                        editable,
                    );

                let source = clip_library.source_clips.get(&id).cloned();
                drop(clip_library);

                if let Some(source) = source {
                    record_clip_added(world, id, source);
                }
                log!("Created empty clip (id={})", id);
            }

            UIEvent::ClipBrowserDuplicate(source_id) => {
                let mut clip_library = world.resource_mut::<ClipLibrary>();
                if let Some(original) = clip_library.get(*source_id).cloned() {
                    let mut duplicate = original;
                    duplicate.name = format!("{} (copy)", duplicate.name);
                    let new_id =
                        crate::ecs::systems::clip_library_systems::clip_library_register_and_activate(
                            &mut clip_library,
                            assets,
                            duplicate,
                        );

                    let source = clip_library.source_clips.get(&new_id).cloned();
                    drop(clip_library);

                    if let Some(source) = source {
                        record_clip_added(world, new_id, source);
                    }
                    log!("Duplicated clip {} -> {}", source_id, new_id);
                }
            }

            UIEvent::ClipBrowserDelete(source_id) => {
                let ref_count = count_source_references(*source_id, world);
                if ref_count == 0 {
                    let removed_source = {
                        let clip_library = world.resource::<ClipLibrary>();
                        clip_library.source_clips.get(source_id).cloned()
                    };

                    let mut clip_library = world.resource_mut::<ClipLibrary>();
                    clip_library.remove(*source_id);
                    drop(clip_library);

                    if let Some(removed) = removed_source {
                        record_clip_removed(world, *source_id, removed);
                    }
                    log!("Deleted clip (id={})", source_id);
                } else {
                    log!(
                        "Cannot delete clip {}: {} references remain",
                        source_id,
                        ref_count
                    );
                }
            }

            UIEvent::ResampleSelectedModelAnimations { fps } => {
                match crate::ecs::systems::clip_library_systems::clip_library_resample_selected_model(
                    world, assets, *fps,
                ) {
                    Some(count) => log!(
                        "Created {} resampled clip(s) of selected model at {} fps",
                        count,
                        fps
                    ),
                    None => log_warn!("Resample: no model with animations is selected"),
                }
            }

            _ => {}
        }
    }
}

fn record_clip_added(
    world: &mut World,
    clip_id: SourceClipId,
    source: crate::animation::editable::SourceClip,
) {
    if world.contains_resource::<EditHistory>() {
        let mut edit_history = world.resource_mut::<EditHistory>();
        edit_history.push_clip_added(clip_id, source, "add clip");
    }
}

fn record_clip_removed(
    world: &mut World,
    clip_id: SourceClipId,
    removed: crate::animation::editable::SourceClip,
) {
    if world.contains_resource::<EditHistory>() {
        let mut edit_history = world.resource_mut::<EditHistory>();
        edit_history.push_clip_removed(clip_id, removed, "delete clip");
    }
}

fn count_source_references(source_id: SourceClipId, world: &World) -> usize {
    let entities = world.component_entities::<ClipSchedule>();
    let mut count = 0;
    for entity in entities {
        if let Some(schedule) = world.get_component::<ClipSchedule>(entity) {
            count += schedule
                .instances
                .iter()
                .filter(|i| i.source_id == source_id)
                .count();
        }
    }
    count
}
