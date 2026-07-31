use crate::animation::editable::{EditableAnimationClip, SourceClip};
use crate::ecs::component::ClipSchedule;
use crate::ecs::events::UIEvent;
use crate::ecs::resource::{ClipLibrary, EditCommand, EditCommandAfter, EditEntry, EditHistory};
use crate::ecs::world::{Entity, World};

pub fn dispatch_edit_history_events(events: &[UIEvent], world: &mut World) {
    for event in events {
        match event {
            UIEvent::Undo => dispatch_undo(world),
            UIEvent::Redo => dispatch_redo(world),
            _ => {}
        }
    }
}

fn dispatch_undo(world: &mut World) {
    let entry = {
        let mut edit_history = world.resource_mut::<EditHistory>();
        if !edit_history.can_undo() {
            return;
        }
        edit_history.pop_undo()
    };
    let Some(entry) = entry else { return };

    match &entry.command {
        EditCommand::ClipModified {
            clip_id,
            before,
            description,
        } => {
            let current = {
                let mut clip_library = world.resource_mut::<ClipLibrary>();
                let current = clip_library
                    .get(*clip_id)
                    .cloned()
                    .unwrap_or_else(|| before.clone());
                if let Some(clip) = clip_library.get_mut(*clip_id) {
                    *clip = before.clone();
                }
                current
            };

            let redo_entry = EditEntry {
                command: EditCommand::ClipModified {
                    clip_id: *clip_id,
                    before: current,
                    description,
                },
                after: entry.after.clone(),
            };

            world.resource_mut::<EditHistory>().push_to_redo(redo_entry);
            log!("Undo: {}", description);
        }

        EditCommand::ScheduleModified {
            entity,
            before,
            description,
        } => {
            let current = world
                .get_component::<ClipSchedule>(*entity)
                .cloned()
                .unwrap_or_else(|| before.clone());
            if let Some(schedule) = world.get_component_mut::<ClipSchedule>(*entity) {
                *schedule = before.clone();
            }

            let redo_entry = EditEntry {
                command: EditCommand::ScheduleModified {
                    entity: *entity,
                    before: current,
                    description,
                },
                after: entry.after.clone(),
            };

            world.resource_mut::<EditHistory>().push_to_redo(redo_entry);
            log!("Undo: {}", description);
        }

        EditCommand::ClipAdded {
            clip_id,
            description,
        } => {
            let removed_source = {
                let mut clip_library = world.resource_mut::<ClipLibrary>();
                clip_library.source_clips.remove(clip_id)
            };

            let redo_entry = EditEntry {
                command: EditCommand::ClipAdded {
                    clip_id: *clip_id,
                    description,
                },
                after: match removed_source {
                    Some(source) => EditCommandAfter::ClipCreated(source),
                    None => entry.after.clone(),
                },
            };

            world.resource_mut::<EditHistory>().push_to_redo(redo_entry);
            log!("Undo: {}", description);
        }

        EditCommand::ClipRemoved {
            clip_id,
            removed,
            description,
        } => {
            restore_source_clip(world, *clip_id, removed.clone());

            let redo_entry = EditEntry {
                command: EditCommand::ClipRemoved {
                    clip_id: *clip_id,
                    removed: removed.clone(),
                    description,
                },
                after: EditCommandAfter::Empty,
            };

            world.resource_mut::<EditHistory>().push_to_redo(redo_entry);
            log!("Undo: {}", description);
        }
    }
}

fn dispatch_redo(world: &mut World) {
    let entry = {
        let mut edit_history = world.resource_mut::<EditHistory>();
        if !edit_history.can_redo() {
            return;
        }
        edit_history.pop_redo()
    };
    let Some(entry) = entry else { return };

    match (&entry.command, &entry.after) {
        (
            EditCommand::ClipModified {
                clip_id,
                description,
                ..
            },
            EditCommandAfter::Clip(after_clip),
        ) => {
            redo_clip_modified(world, *clip_id, after_clip, &entry.after, description);
        }

        (
            EditCommand::ScheduleModified {
                entity,
                description,
                ..
            },
            EditCommandAfter::Schedule(after_schedule),
        ) => {
            redo_schedule_modified(world, *entity, after_schedule, &entry.after, description);
        }

        (
            EditCommand::ClipAdded {
                clip_id,
                description,
            },
            EditCommandAfter::ClipCreated(source),
        ) => {
            redo_clip_added(world, *clip_id, source, description);
        }

        (
            EditCommand::ClipRemoved {
                clip_id,
                removed,
                description,
            },
            EditCommandAfter::Empty,
        ) => {
            redo_clip_removed(world, *clip_id, removed, description);
        }

        _ => {}
    }
}

fn redo_clip_modified(
    world: &mut World,
    clip_id: u64,
    after_clip: &EditableAnimationClip,
    after: &EditCommandAfter,
    description: &'static str,
) {
    let current = {
        let mut clip_library = world.resource_mut::<ClipLibrary>();
        let current = clip_library
            .get(clip_id)
            .cloned()
            .unwrap_or_else(|| after_clip.clone());
        if let Some(clip) = clip_library.get_mut(clip_id) {
            *clip = after_clip.clone();
        }
        current
    };

    let undo_entry = EditEntry {
        command: EditCommand::ClipModified {
            clip_id,
            before: current,
            description,
        },
        after: after.clone(),
    };

    world.resource_mut::<EditHistory>().push_to_undo(undo_entry);
    log!("Redo: {}", description);
}

fn redo_schedule_modified(
    world: &mut World,
    entity: Entity,
    after_schedule: &ClipSchedule,
    after: &EditCommandAfter,
    description: &'static str,
) {
    let current = world
        .get_component::<ClipSchedule>(entity)
        .cloned()
        .unwrap_or_else(|| after_schedule.clone());
    if let Some(schedule) = world.get_component_mut::<ClipSchedule>(entity) {
        *schedule = after_schedule.clone();
    }

    let undo_entry = EditEntry {
        command: EditCommand::ScheduleModified {
            entity,
            before: current,
            description,
        },
        after: after.clone(),
    };

    world.resource_mut::<EditHistory>().push_to_undo(undo_entry);
    log!("Redo: {}", description);
}

fn redo_clip_added(
    world: &mut World,
    clip_id: u64,
    source: &SourceClip,
    description: &'static str,
) {
    restore_source_clip(world, clip_id, source.clone());

    let undo_entry = EditEntry {
        command: EditCommand::ClipAdded {
            clip_id,
            description,
        },
        after: EditCommandAfter::ClipCreated(source.clone()),
    };

    world.resource_mut::<EditHistory>().push_to_undo(undo_entry);
    log!("Redo: {}", description);
}

fn redo_clip_removed(
    world: &mut World,
    clip_id: u64,
    removed: &SourceClip,
    description: &'static str,
) {
    let mut clip_library = world.resource_mut::<ClipLibrary>();
    clip_library.source_clips.remove(&clip_id);

    let undo_entry = EditEntry {
        command: EditCommand::ClipRemoved {
            clip_id,
            removed: removed.clone(),
            description,
        },
        after: EditCommandAfter::Empty,
    };

    world.resource_mut::<EditHistory>().push_to_undo(undo_entry);
    log!("Redo: {}", description);
}

fn restore_source_clip(world: &mut World, clip_id: u64, source: SourceClip) {
    let mut clip_library = world.resource_mut::<ClipLibrary>();
    clip_library.source_clips.insert(clip_id, source);
    clip_library.mark_dirty(clip_id);
}
