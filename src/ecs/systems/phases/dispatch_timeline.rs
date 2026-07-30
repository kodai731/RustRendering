use crate::asset::AssetStorage;
use crate::ecs::component::ClipSchedule;
use crate::ecs::events::UIEvent;
use crate::ecs::resource::CurveEditorState;
use crate::ecs::resource::{
    BonePoseOverride, ClipLibrary, CurveEditorBuffer, EditHistory, KeyframeCopyBuffer,
    TimelineState,
};
use crate::ecs::systems::{
    edit_history_push_clip_mergeable, process_bone_set_key, process_keyframe_clipboard_events,
    timeline_process_events,
};
use crate::ecs::world::World;

use super::dispatch_spring_bone::transition_to_baked_override_if_needed;

pub fn dispatch_timeline_events(events: &[UIEvent], world: &mut World, assets: &AssetStorage) {
    let mut timeline_state = world.resource_mut::<TimelineState>();
    let mut clip_library = world.resource_mut::<ClipLibrary>();

    let clip_id = timeline_state.current_clip_id;
    let before_clip = clip_id.and_then(|id| clip_library.get(id).cloned());

    let modified = timeline_process_events(events, &mut timeline_state, &mut *clip_library);

    if modified {
        if let (Some(cid), Some(before)) = (clip_id, before_clip) {
            if let Some(after) = clip_library.get(cid).cloned() {
                if world.contains_resource::<EditHistory>() {
                    let mut edit_history = world.resource_mut::<EditHistory>();
                    edit_history_push_clip_mergeable(
                        &mut edit_history,
                        cid,
                        before,
                        after,
                        "timeline clip edit",
                    );
                }
            }
        }
    }

    drop(clip_library);
    drop(timeline_state);

    if modified {
        transition_to_baked_override_if_needed(world);
    }

    for event in events {
        match event {
            UIEvent::TimelineSelectClip(source_id) => {
                let lib = world.resource::<ClipLibrary>();
                let duration = lib.get(*source_id).map(|c| c.duration).unwrap_or(1.0);
                let asset_id = lib.get_asset_id_for_source(*source_id);
                log!(
                    "[ClipSelect] source_id={}, asset_id={:?}, duration={:.3}",
                    source_id,
                    asset_id,
                    duration,
                );
                drop(lib);

                let schedule_entities = world.component_entities::<ClipSchedule>();
                for entity in &schedule_entities {
                    if let Some(schedule) = world.get_component_mut::<ClipSchedule>(*entity) {
                        crate::ecs::systems::clip_schedule_systems::clip_schedule_switch_source(
                            schedule, *source_id, duration,
                        );
                    }
                }
            }

            UIEvent::TimelinePlay => {
                if let Some(mut overrides) = world.get_resource_mut::<BonePoseOverride>() {
                    overrides.clear();
                }
            }

            _ => {}
        }
    }

    dispatch_bone_set_key_events(events, world, assets);
}

fn dispatch_bone_set_key_events(events: &[UIEvent], world: &mut World, assets: &AssetStorage) {
    if !events.iter().any(|e| matches!(e, UIEvent::BoneSetKey)) {
        return;
    }

    let overrides = match world.get_resource::<BonePoseOverride>() {
        Some(r) => r.overrides.clone(),
        None => return,
    };
    if overrides.is_empty() {
        return;
    }

    let skeleton_id = world
        .get_resource::<crate::ecs::resource::gizmo::BoneGizmoData>()
        .and_then(|bg| bg.cached_skeleton_id);
    let Some(skel_id) = skeleton_id else { return };
    let Some(skeleton) = assets.get_skeleton_by_skeleton_id(skel_id) else {
        return;
    };
    let skeleton = skeleton.clone();

    let timeline_state = world.resource::<TimelineState>();
    let mut clip_library = world.resource_mut::<ClipLibrary>();

    let clip_id = timeline_state.current_clip_id;
    let before_clip = clip_id.and_then(|id| clip_library.get(id).cloned());

    let modified = process_bone_set_key(&overrides, &mut clip_library, &timeline_state, &skeleton);

    if modified {
        if let (Some(cid), Some(before)) = (clip_id, before_clip) {
            if let Some(after) = clip_library.get(cid).cloned() {
                if world.contains_resource::<EditHistory>() {
                    let mut edit_history = world.resource_mut::<EditHistory>();
                    edit_history.push_clip_edit(cid, before, after, "bone set key");
                }
            }
        }

        if let Some(mut pose_overrides) = world.get_resource_mut::<BonePoseOverride>() {
            pose_overrides.clear();
        }
    }
}

pub fn dispatch_keyframe_clipboard_events(events: &[UIEvent], world: &mut World) {
    let has_paste = events.iter().any(|e| {
        matches!(
            e,
            UIEvent::TimelinePasteKeyframes { .. } | UIEvent::TimelineMirrorPaste { .. }
        )
    });

    let timeline_state = world.resource::<TimelineState>();
    let mut clip_library = world.resource_mut::<ClipLibrary>();
    let mut copy_buffer = world.resource_mut::<KeyframeCopyBuffer>();

    let clip_id = timeline_state.current_clip_id;
    let before_clip = if has_paste {
        clip_id.and_then(|id| clip_library.get(id).cloned())
    } else {
        None
    };

    process_keyframe_clipboard_events(
        events,
        &*timeline_state,
        &mut *clip_library,
        &mut *copy_buffer,
    );

    if has_paste {
        if let (Some(cid), Some(before)) = (clip_id, before_clip) {
            if let Some(after) = clip_library.get(cid).cloned() {
                drop(clip_library);
                drop(timeline_state);
                drop(copy_buffer);
                if world.contains_resource::<EditHistory>() {
                    let mut edit_history = world.resource_mut::<EditHistory>();
                    edit_history.push_clip_edit(cid, before, after, "paste keyframes");
                }
                return;
            }
        }
    }
}

pub fn dispatch_buffer_events(events: &[UIEvent], world: &mut World) {
    for event in events {
        match event {
            UIEvent::TimelineCaptureBuffer => {
                let timeline_state = world.resource::<TimelineState>();
                let clip_library = world.resource::<ClipLibrary>();
                let curve_editor = world.resource::<CurveEditorState>();
                let mut curve_buffer = world.resource_mut::<CurveEditorBuffer>();

                if let Some(bone_id) = curve_editor.selected_bone_id() {
                    if let Some(clip_id) = timeline_state.current_clip_id {
                        if let Some(clip) = clip_library.get(clip_id) {
                            crate::ecs::systems::curve_editor_capture_buffer(
                                &mut curve_buffer,
                                clip,
                                bone_id,
                                &curve_editor.visible_curves,
                                clip.duration,
                                100,
                            );
                        }
                    }
                }
            }

            UIEvent::TimelineSwapBuffer => {
                let curve_editor = world.resource::<CurveEditorState>();
                let timeline_state = world.resource::<TimelineState>();
                let mut clip_library = world.resource_mut::<ClipLibrary>();
                let mut curve_buffer = world.resource_mut::<CurveEditorBuffer>();

                let clip_id = timeline_state.current_clip_id;
                let before_clip = clip_id.and_then(|id| clip_library.get(id).cloned());

                if let Some(bone_id) = curve_editor.selected_bone_id() {
                    if let Some(cid) = clip_id {
                        if let Some(clip) = clip_library.get_mut(cid) {
                            crate::ecs::systems::curve_editor_swap_buffer(
                                &mut curve_buffer,
                                clip,
                                bone_id,
                            );
                        }
                    }
                }
                if let (Some(cid), Some(before)) = (clip_id, before_clip) {
                    if let Some(after) = clip_library.get(cid).cloned() {
                        drop(clip_library);
                        drop(curve_buffer);
                        drop(timeline_state);
                        drop(curve_editor);
                        if world.contains_resource::<EditHistory>() {
                            let mut edit_history = world.resource_mut::<EditHistory>();
                            edit_history.push_clip_edit(cid, before, after, "swap buffer");
                        }
                    }
                }
            }

            _ => {}
        }
    }
}
