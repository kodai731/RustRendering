use std::collections::HashMap;

use crate::animation::editable::{
    apply_tangent_by_type, clip_add_keyframe, clip_recalculate_duration, curve_add_keyframe,
    curve_recalculate_auto_tangent_at, curve_remove_keyframe, curve_set_keyframe_time,
    initialize_weighted_handle_lengths, EditableAnimationClip, InterpolationType, KeyframeId,
    PropertyCurve, PropertyType, SourceClipId, TangentWeightMode,
};
use crate::animation::{BoneId, BoneLocalPose};
use crate::ecs::component::ClipSchedule;
use crate::ecs::events::UIEvent;
use crate::ecs::resource::{ClipLibrary, CurveTrackRef, TimelineState};
use crate::ecs::world::World;

fn ensure_bezier_for_tangent(curve: &mut PropertyCurve, keyframe_id: KeyframeId) {
    if let Some(idx) = curve.keyframes.iter().position(|k| k.id == keyframe_id) {
        curve.keyframes[idx].interpolation = InterpolationType::Bezier;
    }
}

fn resolve_curve_mut(
    clip: &mut EditableAnimationClip,
    track: CurveTrackRef,
    property_type: PropertyType,
) -> Option<&mut PropertyCurve> {
    match track {
        CurveTrackRef::Bone(bone_id) => clip
            .tracks
            .get_mut(&bone_id)
            .map(|t| t.get_curve_mut(property_type)),
        CurveTrackRef::Scalar => clip.get_scalar_curve_mut(property_type),
    }
}

pub fn timeline_process_events(
    events: &[UIEvent],
    timeline_state: &mut TimelineState,
    clip_library: &mut ClipLibrary,
) -> bool {
    let mut clip_modified = false;

    for event in events {
        match event {
            UIEvent::TimelinePlay => timeline_state.playing = true,
            UIEvent::TimelinePause => timeline_state.playing = false,
            UIEvent::TimelineStop => {
                timeline_state.playing = false;
                timeline_state.current_time = 0.0;
            }
            UIEvent::TimelineSetTime(time) => {
                timeline_state.playing = false;
                timeline_state.set_time(*time);
            }
            UIEvent::TimelineSetSpeed(speed) => timeline_state.speed = *speed,
            UIEvent::TimelineToggleLoop => timeline_state.looping = !timeline_state.looping,
            UIEvent::TimelineSelectClip(clip_id) => {
                timeline_select_clip(timeline_state, clip_library, *clip_id);
            }
            UIEvent::TimelineToggleTrack(bone_id) => {
                timeline_toggle_track_expanded(timeline_state, *bone_id);
            }
            UIEvent::TimelineExpandTrack(bone_id) => timeline_state.expand_track(*bone_id),
            UIEvent::TimelineCollapseTrack(bone_id) => timeline_state.collapse_track(*bone_id),
            UIEvent::TimelineSelectKeyframe {
                track,
                property_type,
                keyframe_id,
                modifier,
            } => {
                use crate::ecs::resource::SelectedKeyframe;
                let selected = SelectedKeyframe::new(*track, *property_type, *keyframe_id);
                timeline_apply_selection(timeline_state, selected, *modifier);
            }
            UIEvent::TimelineSetKeyframeSelection {
                keyframes,
                modifier,
            } => {
                dispatch_set_keyframe_selection(timeline_state, keyframes, *modifier);
            }
            UIEvent::TimelineSetSnapToFrame(enabled) => {
                timeline_state.snap_settings.snap_to_frame = *enabled;
            }
            UIEvent::TimelineSetSnapToKey(enabled) => {
                timeline_state.snap_settings.snap_to_key = *enabled;
            }
            UIEvent::TimelineSetFrameRate(rate) => {
                timeline_state.snap_settings.frame_rate = *rate;
            }
            UIEvent::TimelineZoomIn { max_zoom } => {
                timeline_zoom_in(timeline_state, *max_zoom);
            }
            UIEvent::TimelineZoomOut { min_zoom } => {
                timeline_zoom_out(timeline_state, *min_zoom);
            }
            _ => {}
        }
    }

    clip_modified |= dispatch_keyframe_edit_events(events, timeline_state, clip_library);
    clip_modified |= dispatch_tangent_edit_events(events, timeline_state, clip_library);

    clip_modified
}

fn dispatch_set_keyframe_selection(
    timeline_state: &mut TimelineState,
    keyframes: &[crate::ecs::resource::SelectedKeyframe],
    modifier: crate::ecs::resource::SelectionModifier,
) {
    use crate::ecs::resource::SelectionModifier;
    match modifier {
        SelectionModifier::Replace => {
            timeline_state.selected_keyframes.clear();
            for kf in keyframes {
                timeline_state.selected_keyframes.insert(kf.clone());
            }
        }
        SelectionModifier::Add => {
            for kf in keyframes {
                timeline_state.selected_keyframes.insert(kf.clone());
            }
        }
        SelectionModifier::Toggle => {
            for kf in keyframes {
                if timeline_state.selected_keyframes.contains(kf) {
                    timeline_state.selected_keyframes.remove(kf);
                } else {
                    timeline_state.selected_keyframes.insert(kf.clone());
                }
            }
        }
    }
}

fn dispatch_keyframe_edit_events(
    events: &[UIEvent],
    timeline_state: &mut TimelineState,
    clip_library: &mut ClipLibrary,
) -> bool {
    let mut clip_modified = false;
    let Some(clip_id) = timeline_state.current_clip_id else {
        return false;
    };

    for event in events {
        match event {
            UIEvent::TimelineAddKeyframe {
                track,
                property_type,
                time,
                value,
            } => {
                if let Some(clip) = clip_library.get_mut(clip_id) {
                    match track {
                        CurveTrackRef::Bone(bone_id) => {
                            clip_add_keyframe(clip, *bone_id, *property_type, *time, *value);
                        }
                        CurveTrackRef::Scalar => {
                            // Scalar curves are keyed by Custom codes; a bone
                            // property type here would create an unnamed curve
                            // that nothing ever samples.
                            if !matches!(property_type, PropertyType::Custom(_)) {
                                continue;
                            }
                            let curve = clip.get_or_add_scalar_curve(*property_type);
                            curve_add_keyframe(curve, *time, *value);
                        }
                    }
                    clip_recalculate_duration(clip);
                    clip_modified = true;
                }
            }

            UIEvent::TimelineMoveSelectedKeyframes { time_delta } => {
                if let Some(clip) = clip_library.get_mut(clip_id) {
                    for sel in &timeline_state.selected_keyframes {
                        if let Some(curve) = resolve_curve_mut(clip, sel.track, sel.property_type) {
                            if let Some(kf) =
                                curve.keyframes.iter_mut().find(|k| k.id == sel.keyframe_id)
                            {
                                kf.time = (kf.time + time_delta).max(0.0);
                            }
                        }
                    }
                    clip_recalculate_duration(clip);
                    clip_modified = true;
                }
            }

            UIEvent::TimelineDeleteSelectedKeyframes => {
                let selected: Vec<_> = timeline_state.selected_keyframes.iter().cloned().collect();
                if !selected.is_empty() {
                    if let Some(clip) = clip_library.get_mut(clip_id) {
                        for sel in &selected {
                            if let Some(curve) =
                                resolve_curve_mut(clip, sel.track, sel.property_type)
                            {
                                curve_remove_keyframe(curve, sel.keyframe_id);
                            }
                        }
                        clip_recalculate_duration(clip);
                        clip_modified = true;
                    }
                }
                timeline_state.clear_selection();
            }

            UIEvent::TimelineMoveKeyframe {
                track,
                property_type,
                keyframe_id,
                new_time,
                new_value,
            } => {
                if let Some(clip) = clip_library.get_mut(clip_id) {
                    if let Some(curve) = resolve_curve_mut(clip, *track, *property_type) {
                        curve_set_keyframe_time(curve, *keyframe_id, *new_time);
                        curve.set_keyframe_value(*keyframe_id, *new_value);
                    }
                    clip_recalculate_duration(clip);
                    clip_modified = true;
                }
            }

            UIEvent::TimelineDeleteKeyframe {
                track,
                property_type,
                keyframe_id,
            } => {
                if let Some(clip) = clip_library.get_mut(clip_id) {
                    if let Some(curve) = resolve_curve_mut(clip, *track, *property_type) {
                        curve_remove_keyframe(curve, *keyframe_id);
                    }
                    clip_recalculate_duration(clip);
                    clip_modified = true;
                }
            }

            _ => {}
        }
    }

    clip_modified
}

fn dispatch_tangent_edit_events(
    events: &[UIEvent],
    timeline_state: &TimelineState,
    clip_library: &mut ClipLibrary,
) -> bool {
    let mut clip_modified = false;
    let Some(clip_id) = timeline_state.current_clip_id else {
        return false;
    };

    for event in events {
        match event {
            UIEvent::TimelineSetKeyframeInterpolation {
                track,
                property_type,
                keyframe_id,
                interpolation,
            } => {
                if let Some(clip) = clip_library.get_mut(clip_id) {
                    if let Some(curve) = resolve_curve_mut(clip, *track, *property_type) {
                        curve.set_keyframe_interpolation(*keyframe_id, *interpolation);
                        clip_modified = true;
                    }
                }
            }

            UIEvent::TimelineSetKeyframeTangent {
                track,
                property_type,
                keyframe_id,
                in_tangent,
                out_tangent,
            } => {
                if let Some(clip) = clip_library.get_mut(clip_id) {
                    if let Some(curve) = resolve_curve_mut(clip, *track, *property_type) {
                        if let Some(kf) = curve.get_keyframe_mut(*keyframe_id) {
                            kf.tangent_type = crate::animation::editable::TangentType::Manual;
                        }
                        curve.set_keyframe_tangents(
                            *keyframe_id,
                            in_tangent.clone(),
                            out_tangent.clone(),
                        );
                        clip_modified = true;
                    }
                }
            }

            UIEvent::TimelineSetTangentType {
                track,
                property_type,
                keyframe_id,
                tangent_type,
            } => {
                if let Some(clip) = clip_library.get_mut(clip_id) {
                    if let Some(curve) = resolve_curve_mut(clip, *track, *property_type) {
                        ensure_bezier_for_tangent(curve, *keyframe_id);
                        if let Some(idx) = curve.keyframes.iter().position(|k| k.id == *keyframe_id)
                        {
                            curve.keyframes[idx].tangent_type = *tangent_type;
                            apply_tangent_by_type(&mut curve.keyframes, idx);
                        }
                        clip_modified = true;
                    }
                }
            }

            UIEvent::TimelineSetTangentWeightMode {
                track,
                property_type,
                keyframe_id,
                weight_mode,
            } => {
                if let Some(clip) = clip_library.get_mut(clip_id) {
                    if let Some(curve) = resolve_curve_mut(clip, *track, *property_type) {
                        curve.set_keyframe_weight_mode(*keyframe_id, *weight_mode);
                        if *weight_mode == TangentWeightMode::Weighted {
                            if let Some(idx) =
                                curve.keyframes.iter().position(|k| k.id == *keyframe_id)
                            {
                                let dt = compute_average_keyframe_interval(curve).max(0.1);
                                initialize_weighted_handle_lengths(&mut curve.keyframes[idx], dt);
                            }
                        }
                        clip_modified = true;
                    }
                }
            }

            _ => {}
        }
    }

    clip_modified
}

fn compute_average_keyframe_interval(curve: &PropertyCurve) -> f32 {
    if curve.keyframes.len() <= 1 {
        return 1.0;
    }
    let first = curve.keyframes.first().expect("guarded by len > 1").time;
    let last = curve.keyframes.last().expect("guarded by len > 1").time;
    (last - first) / (curve.keyframes.len() as f32 - 1.0)
}

fn timeline_select_clip(
    timeline_state: &mut TimelineState,
    clip_library: &ClipLibrary,
    clip_id: SourceClipId,
) {
    if let Some(clip) = clip_library.get(clip_id) {
        timeline_state.current_clip_id = Some(clip_id);
        timeline_state.current_time = 0.0;
        timeline_state.selected_keyframes.clear();
        timeline_state.expanded_tracks.clear();

        if let Some((&first_bone_id, _)) = clip.tracks.iter().next() {
            timeline_state.expand_track(first_bone_id);
        }

        timeline_apply_fit_zoom(timeline_state, clip.duration);

        log!(
            "Timeline: Selected clip '{}' (id={}, duration={:.2}s, tracks={})",
            clip.name,
            clip_id,
            clip.duration,
            clip.track_count()
        );
    }
}

pub fn timeline_apply_fit_zoom(timeline_state: &mut TimelineState, clip_duration: f32) {
    timeline_state.scroll_offset = 0.0;

    let visible_width = timeline_state.last_visible_width;
    if visible_width < 1.0 || clip_duration <= 0.0 {
        return;
    }

    let fit_zoom =
        visible_width / (clip_duration * crate::platform::ui::timeline_window::PIXELS_PER_SECOND);
    timeline_state.zoom_level = fit_zoom.clamp(0.01, 100.0);
}

pub fn timeline_apply_selection(
    state: &mut TimelineState,
    keyframe: crate::ecs::resource::SelectedKeyframe,
    modifier: crate::ecs::resource::SelectionModifier,
) {
    use crate::ecs::resource::SelectionModifier;
    match modifier {
        SelectionModifier::Replace => {
            state.selected_keyframes.clear();
            state.selected_keyframes.insert(keyframe);
        }
        SelectionModifier::Add => {
            state.selected_keyframes.insert(keyframe);
        }
        SelectionModifier::Toggle => {
            if state.selected_keyframes.contains(&keyframe) {
                state.selected_keyframes.remove(&keyframe);
            } else {
                state.selected_keyframes.insert(keyframe);
            }
        }
    }
}

pub fn timeline_toggle_track_expanded(state: &mut TimelineState, bone_id: BoneId) {
    if state.expanded_tracks.contains(&bone_id) {
        state.expanded_tracks.remove(&bone_id);
    } else {
        state.expanded_tracks.insert(bone_id);
    }
}

pub fn timeline_zoom_in(state: &mut TimelineState, max_zoom: f32) {
    state.zoom_level = (state.zoom_level * 1.2).min(max_zoom);
}

pub fn timeline_zoom_out(state: &mut TimelineState, min_zoom: f32) {
    state.zoom_level = (state.zoom_level / 1.2).max(min_zoom);
}

/// Timeline (start, end) a dragged clip block should be drawn at while the drag
/// is in progress. Mirrors the clamps the commit events apply on release
/// (`ClipInstanceMove`/`TrimStart`/`TrimEnd`), so the live preview always shows
/// exactly what releasing the mouse would produce.
pub fn clip_drag_preview_times(
    drag_type: &crate::ecs::resource::ClipDragType,
    original_value: f32,
    delta_time: f32,
    inst_start: f32,
    inst_end: f32,
    clip_in: f32,
    clip_out: f32,
) -> (f32, f32) {
    use crate::ecs::resource::ClipDragType;

    let span = clip_out - clip_in;
    let seconds_per_clip_second = if span > 1e-6 {
        (inst_end - inst_start) / span
    } else {
        1.0
    };

    match drag_type {
        ClipDragType::Move => {
            let start = (original_value + delta_time).max(0.0);
            (start, start + (inst_end - inst_start))
        }
        ClipDragType::TrimStart => {
            let new_clip_in = (original_value + delta_time).clamp(0.0, clip_out);
            (
                inst_start,
                inst_start + (clip_out - new_clip_in) * seconds_per_clip_second,
            )
        }
        ClipDragType::TrimEnd => {
            let new_clip_out = (original_value + delta_time).max(clip_in);
            (
                inst_start,
                inst_start + (new_clip_out - clip_in) * seconds_per_clip_second,
            )
        }
    }
}

/// Range the timeline spans when no clip supplies one, so the ruler, the transport and
/// playback all agree on how far the playhead may travel.
pub const TIMELINE_FALLBACK_DURATION_SECONDS: f32 = 5.0;

/// Range the playhead travels over. Always positive: scrubbing and playback stay usable
/// before a clip exists.
pub fn timeline_effective_duration(
    timeline_state: &TimelineState,
    clip_library: &ClipLibrary,
) -> f32 {
    let clip_duration = timeline_state
        .current_clip_id
        .and_then(|id| clip_library.get(id))
        .map(|c| c.duration)
        .unwrap_or(0.0);

    if clip_duration > 0.0 {
        clip_duration
    } else {
        TIMELINE_FALLBACK_DURATION_SECONDS
    }
}

pub fn timeline_update(
    timeline_state: &mut TimelineState,
    clip_library: &ClipLibrary,
    delta_time: f32,
) {
    if !timeline_state.playing {
        return;
    }

    let duration = timeline_effective_duration(timeline_state, clip_library);
    let new_time = timeline_state.current_time + delta_time * timeline_state.speed;

    if timeline_state.looping {
        timeline_state.current_time = new_time % duration;
    } else if new_time >= duration {
        timeline_state.current_time = duration;
        timeline_state.playing = false;
    } else {
        timeline_state.current_time = new_time;
    }
}

pub fn process_clip_instance_events(events: &[UIEvent], world: &mut World) {
    let mut deselect_after: Option<(
        crate::ecs::world::Entity,
        crate::animation::editable::ClipInstanceId,
    )> = None;

    for event in events {
        match event {
            UIEvent::ClipInstanceSelect {
                entity,
                instance_id,
            } => {
                dispatch_clip_instance_select(world, *entity, *instance_id);
            }

            UIEvent::ClipInstanceDeselect => {
                world.resource_mut::<TimelineState>().selected_clip_instance = None;
            }

            UIEvent::ClipInstanceMove {
                entity,
                instance_id,
                new_start_time,
            } => {
                modify_clip_instance(world, *entity, *instance_id, |inst| {
                    inst.start_time = *new_start_time;
                });
            }

            UIEvent::ClipInstanceTrimStart {
                entity,
                instance_id,
                new_clip_in,
            } => {
                modify_clip_instance(world, *entity, *instance_id, |inst| {
                    inst.clip_in = new_clip_in.clamp(0.0, inst.clip_out);
                });
            }

            UIEvent::ClipInstanceTrimEnd {
                entity,
                instance_id,
                new_clip_out,
            } => {
                modify_clip_instance(world, *entity, *instance_id, |inst| {
                    inst.clip_out = new_clip_out.max(inst.clip_in);
                });
            }

            UIEvent::ClipInstanceToggleMute {
                entity,
                instance_id,
            } => {
                modify_clip_instance(world, *entity, *instance_id, |inst| {
                    inst.muted = !inst.muted;
                });
            }

            UIEvent::ClipInstanceDelete {
                entity,
                instance_id,
            } => {
                if let Some(schedule) = world.get_component_mut::<ClipSchedule>(*entity) {
                    super::clip_schedule_systems::clip_schedule_remove_instance(
                        schedule,
                        *instance_id,
                    );
                }
                deselect_after = Some((*entity, *instance_id));
            }

            UIEvent::ClipInstanceSetWeight {
                entity,
                instance_id,
                weight,
            } => {
                modify_clip_instance(world, *entity, *instance_id, |inst| {
                    inst.weight = weight.clamp(0.0, 1.0);
                });
            }

            UIEvent::ClipInstanceSetBlendMode {
                entity,
                instance_id,
                blend_mode,
            } => {
                modify_clip_instance(world, *entity, *instance_id, |inst| {
                    inst.blend_mode = *blend_mode;
                });
            }

            _ => {}
        }
    }

    dispatch_clip_group_events(events, world);

    if let Some((entity, instance_id)) = deselect_after {
        let mut ts = world.resource_mut::<TimelineState>();
        if let Some((sel_entity, sel_id)) = ts.selected_clip_instance {
            if sel_entity == entity && sel_id == instance_id {
                ts.selected_clip_instance = None;
            }
        }
    }
}

fn dispatch_clip_instance_select(
    world: &mut World,
    entity: crate::ecs::world::Entity,
    instance_id: crate::animation::editable::ClipInstanceId,
) {
    let source_id = world
        .get_component::<ClipSchedule>(entity)
        .and_then(|schedule| {
            schedule
                .instances
                .iter()
                .find(|i| i.instance_id == instance_id)
                .map(|i| i.source_id)
        });

    let mut ts = world.resource_mut::<TimelineState>();
    ts.selected_clip_instance = Some((entity, instance_id));

    if let Some(source_id) = source_id {
        if ts.current_clip_id != Some(source_id) {
            let clip_library = world.resource::<ClipLibrary>();
            if let Some(clip) = clip_library.get(source_id) {
                ts.current_clip_id = Some(source_id);
                ts.current_time = 0.0;
                ts.selected_keyframes.clear();
                ts.expanded_tracks.clear();

                if let Some((&first_bone_id, _)) = clip.tracks.iter().next() {
                    ts.expand_track(first_bone_id);
                }

                let duration = clip.duration;
                timeline_apply_fit_zoom(&mut ts, duration);
            }
        }
    }
}

fn dispatch_clip_group_events(events: &[UIEvent], world: &mut World) {
    for event in events {
        match event {
            UIEvent::ClipGroupCreate { entity, name } => {
                if let Some(schedule) = world.get_component_mut::<ClipSchedule>(*entity) {
                    super::clip_schedule_systems::clip_schedule_create_group(
                        schedule,
                        name.clone(),
                    );
                }
            }

            UIEvent::ClipGroupDelete { entity, group_id } => {
                if let Some(schedule) = world.get_component_mut::<ClipSchedule>(*entity) {
                    super::clip_schedule_systems::clip_schedule_remove_group(schedule, *group_id);
                }
            }

            UIEvent::ClipGroupAddInstance {
                entity,
                group_id,
                instance_id,
            } => {
                if let Some(schedule) = world.get_component_mut::<ClipSchedule>(*entity) {
                    super::clip_schedule_systems::clip_schedule_add_to_group(
                        schedule,
                        *group_id,
                        *instance_id,
                    );
                }
            }

            UIEvent::ClipGroupRemoveInstance {
                entity,
                group_id,
                instance_id,
            } => {
                if let Some(schedule) = world.get_component_mut::<ClipSchedule>(*entity) {
                    super::clip_schedule_systems::clip_schedule_remove_from_group(
                        schedule,
                        *group_id,
                        *instance_id,
                    );
                }
            }

            UIEvent::ClipGroupToggleMute { entity, group_id } => {
                if let Some(schedule) = world.get_component_mut::<ClipSchedule>(*entity) {
                    if let Some(group) = schedule.groups.iter_mut().find(|g| g.id == *group_id) {
                        group.muted = !group.muted;
                    }
                }
            }

            UIEvent::ClipGroupSetWeight {
                entity,
                group_id,
                weight,
            } => {
                if let Some(schedule) = world.get_component_mut::<ClipSchedule>(*entity) {
                    if let Some(group) = schedule.groups.iter_mut().find(|g| g.id == *group_id) {
                        group.weight = weight.clamp(0.0, 1.0);
                    }
                }
            }

            _ => {}
        }
    }
}

fn modify_clip_instance(
    world: &mut World,
    entity: crate::ecs::world::Entity,
    instance_id: crate::animation::editable::ClipInstanceId,
    f: impl FnOnce(&mut crate::animation::editable::ClipInstance),
) {
    if let Some(schedule) = world.get_component_mut::<ClipSchedule>(entity) {
        if let Some(inst) = schedule
            .instances
            .iter_mut()
            .find(|i| i.instance_id == instance_id)
        {
            f(inst);
        }
    }
}

pub fn process_bone_set_key(
    overrides: &HashMap<BoneId, BoneLocalPose>,
    clip_library: &mut ClipLibrary,
    timeline_state: &TimelineState,
    skeleton: &crate::animation::Skeleton,
) -> bool {
    let Some(clip_id) = timeline_state.current_clip_id else {
        return false;
    };
    let Some(clip) = clip_library.get_mut(clip_id) else {
        return false;
    };

    if overrides.is_empty() {
        return false;
    }

    let time = timeline_state.current_time;

    for (&bone_id, local_pose) in overrides {
        let bone_name = skeleton
            .get_bone(bone_id)
            .map(|b| b.name.clone())
            .unwrap_or_else(|| format!("bone_{}", bone_id));

        if !clip.tracks.contains_key(&bone_id) {
            clip.add_track(bone_id, bone_name.clone());
        }

        let euler = crate::math::quaternion_to_euler_degrees(&local_pose.rotation);

        let t = &local_pose.translation;
        let s = &local_pose.scale;
        clip_add_keyframe(clip, bone_id, PropertyType::TranslationX, time, t.x);
        clip_add_keyframe(clip, bone_id, PropertyType::TranslationY, time, t.y);
        clip_add_keyframe(clip, bone_id, PropertyType::TranslationZ, time, t.z);
        clip_add_keyframe(clip, bone_id, PropertyType::RotationX, time, euler.x);
        clip_add_keyframe(clip, bone_id, PropertyType::RotationY, time, euler.y);
        clip_add_keyframe(clip, bone_id, PropertyType::RotationZ, time, euler.z);
        clip_add_keyframe(clip, bone_id, PropertyType::ScaleX, time, s.x);
        clip_add_keyframe(clip, bone_id, PropertyType::ScaleY, time, s.y);
        clip_add_keyframe(clip, bone_id, PropertyType::ScaleZ, time, s.z);
    }

    clip_recalculate_duration(clip);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::editable::{
        EditableAnimationClip, PropertyType, SourceClip, SourceClipId,
    };
    use crate::ecs::resource::{SelectedKeyframe, SelectionModifier};

    fn setup_test_clip() -> (TimelineState, ClipLibrary) {
        let clip_id: SourceClipId = 1;
        let mut clip = EditableAnimationClip::new(clip_id, "test".to_string());
        let bone_id = 0;
        clip.add_track(bone_id, "bone0".to_string());

        let kf1_id =
            clip_add_keyframe(&mut clip, bone_id, PropertyType::TranslationX, 0.5, 1.0).unwrap();
        let kf2_id =
            clip_add_keyframe(&mut clip, bone_id, PropertyType::TranslationX, 1.0, 2.0).unwrap();
        let kf3_id =
            clip_add_keyframe(&mut clip, bone_id, PropertyType::TranslationY, 0.8, 3.0).unwrap();
        clip_recalculate_duration(&mut clip);

        let mut library = ClipLibrary::new();
        library
            .source_clips
            .insert(clip_id, SourceClip::new(clip_id, clip));

        let mut state = TimelineState::new();
        state.current_clip_id = Some(clip_id);

        state.selected_keyframes.insert(SelectedKeyframe::for_bone(
            bone_id,
            PropertyType::TranslationX,
            kf1_id,
        ));
        state.selected_keyframes.insert(SelectedKeyframe::for_bone(
            bone_id,
            PropertyType::TranslationX,
            kf2_id,
        ));
        state.selected_keyframes.insert(SelectedKeyframe::for_bone(
            bone_id,
            PropertyType::TranslationY,
            kf3_id,
        ));

        (state, library)
    }

    #[test]
    fn timeline_advances_without_a_clip() {
        let library = ClipLibrary::new();
        let mut state = TimelineState::new();
        state.playing = true;

        timeline_update(&mut state, &library, 0.5);

        assert!((state.current_time - 0.5).abs() < 1e-5);
    }

    #[test]
    fn timeline_without_a_clip_loops_at_the_fallback_duration() {
        let library = ClipLibrary::new();
        let mut state = TimelineState::new();
        state.playing = true;
        state.looping = true;
        state.current_time = TIMELINE_FALLBACK_DURATION_SECONDS - 0.25;

        timeline_update(&mut state, &library, 0.5);

        assert!((state.current_time - 0.25).abs() < 1e-5);
    }

    #[test]
    fn timeline_without_a_clip_stops_at_the_fallback_duration_when_not_looping() {
        let library = ClipLibrary::new();
        let mut state = TimelineState::new();
        state.playing = true;
        state.looping = false;
        state.current_time = TIMELINE_FALLBACK_DURATION_SECONDS - 0.25;

        timeline_update(&mut state, &library, 0.5);

        assert!((state.current_time - TIMELINE_FALLBACK_DURATION_SECONDS).abs() < 1e-5);
        assert!(!state.playing);
    }

    #[test]
    fn timeline_uses_the_clip_duration_when_one_is_selected() {
        let (mut state, library) = setup_test_clip();
        let clip_duration = library.get(1).unwrap().duration;
        assert!(clip_duration > 0.0 && clip_duration < TIMELINE_FALLBACK_DURATION_SECONDS);

        assert!(
            (timeline_effective_duration(&state, &library) - clip_duration).abs() < 1e-5,
            "effective duration must follow the clip, not the fallback"
        );

        state.playing = true;
        state.looping = true;
        state.current_time = clip_duration - 0.1;
        timeline_update(&mut state, &library, 0.2);

        assert!((state.current_time - 0.1).abs() < 1e-5);
    }

    #[test]
    fn move_selected_keyframes_shifts_time() {
        let (mut state, mut library) = setup_test_clip();
        let events = vec![UIEvent::TimelineMoveSelectedKeyframes { time_delta: 0.25 }];

        let modified = timeline_process_events(&events, &mut state, &mut library);
        assert!(modified);

        let clip = library.get(1).unwrap();
        let track = clip.tracks.get(&0).unwrap();
        let tx_curve = track.get_curve(PropertyType::TranslationX);

        let times: Vec<f32> = tx_curve.keyframes.iter().map(|k| k.time).collect();
        assert!(
            (times[0] - 0.75).abs() < 0.01,
            "Expected 0.75, got {}",
            times[0]
        );
        assert!(
            (times[1] - 1.25).abs() < 0.01,
            "Expected 1.25, got {}",
            times[1]
        );

        let ty_curve = track.get_curve(PropertyType::TranslationY);
        let ty_time = ty_curve.keyframes[0].time;
        assert!(
            (ty_time - 1.05).abs() < 0.01,
            "Expected 1.05, got {}",
            ty_time
        );
    }

    #[test]
    fn move_selected_keyframes_clamps_at_zero() {
        let (mut state, mut library) = setup_test_clip();
        let events = vec![UIEvent::TimelineMoveSelectedKeyframes { time_delta: -2.0 }];

        timeline_process_events(&events, &mut state, &mut library);

        let clip = library.get(1).unwrap();
        let track = clip.tracks.get(&0).unwrap();
        let tx_curve = track.get_curve(PropertyType::TranslationX);

        for kf in &tx_curve.keyframes {
            assert!(kf.time >= 0.0, "Time should be >= 0, got {}", kf.time);
        }
    }

    #[test]
    fn set_keyframe_selection_replace() {
        let (mut state, mut library) = setup_test_clip();

        let new_sel = vec![SelectedKeyframe::for_bone(0, PropertyType::ScaleX, 999)];
        let events = vec![UIEvent::TimelineSetKeyframeSelection {
            keyframes: new_sel,
            modifier: SelectionModifier::Replace,
        }];

        timeline_process_events(&events, &mut state, &mut library);

        assert_eq!(state.selected_keyframes.len(), 1);
        assert!(state
            .selected_keyframes
            .contains(&SelectedKeyframe::for_bone(0, PropertyType::ScaleX, 999)));
    }

    #[test]
    fn set_keyframe_selection_add() {
        let (mut state, mut library) = setup_test_clip();
        let original_count = state.selected_keyframes.len();

        let new_sel = vec![SelectedKeyframe::for_bone(0, PropertyType::ScaleX, 999)];
        let events = vec![UIEvent::TimelineSetKeyframeSelection {
            keyframes: new_sel,
            modifier: SelectionModifier::Add,
        }];

        timeline_process_events(&events, &mut state, &mut library);

        assert_eq!(state.selected_keyframes.len(), original_count + 1);
    }

    #[test]
    fn set_keyframe_selection_toggle() {
        let (mut state, mut library) = setup_test_clip();

        // Toggle off an existing keyframe, toggle on a new one
        let existing = state.selected_keyframes.iter().next().unwrap().clone();
        let new_kf = SelectedKeyframe::for_bone(0, PropertyType::ScaleX, 999);
        let events = vec![UIEvent::TimelineSetKeyframeSelection {
            keyframes: vec![existing.clone(), new_kf.clone()],
            modifier: SelectionModifier::Toggle,
        }];

        let before_count = state.selected_keyframes.len();
        timeline_process_events(&events, &mut state, &mut library);

        // One removed, one added → count stays same
        assert_eq!(state.selected_keyframes.len(), before_count);
        assert!(!state.selected_keyframes.contains(&existing));
        assert!(state.selected_keyframes.contains(&new_kf));
    }

    #[test]
    fn scalar_add_keyframe_rejects_bone_property_types() {
        let (mut state, mut library) = setup_test_clip();

        let events = vec![
            UIEvent::TimelineAddKeyframe {
                track: CurveTrackRef::Scalar,
                property_type: PropertyType::TranslationX,
                time: 0.5,
                value: 1.0,
            },
            UIEvent::TimelineAddKeyframe {
                track: CurveTrackRef::Scalar,
                property_type: PropertyType::Custom(0),
                time: 0.5,
                value: 1.0,
            },
        ];
        timeline_process_events(&events, &mut state, &mut library);

        let clip = library.get(state.current_clip_id.unwrap()).unwrap();
        assert_eq!(clip.scalar_curves.len(), 1);
        assert_eq!(clip.scalar_curves[0].property_type, PropertyType::Custom(0));
    }

    fn spawn_zero_length_clip_instance(world: &mut World) -> crate::ecs::world::Entity {
        use crate::animation::editable::ClipInstance;
        use crate::ecs::component::ClipSchedule;

        let entity = world.spawn();
        let mut schedule = ClipSchedule::default();
        schedule.next_instance_id = 2;
        schedule.instances.push(ClipInstance::new(1, 1, 0.0));
        world.insert_component(entity, schedule);
        entity
    }

    fn instance_clip_range(world: &World, entity: crate::ecs::world::Entity) -> (f32, f32) {
        let schedule = world
            .get_component::<crate::ecs::component::ClipSchedule>(entity)
            .unwrap();
        let inst = &schedule.instances[0];
        (inst.clip_in, inst.clip_out)
    }

    #[test]
    fn trim_end_extends_a_zero_length_clip_freely() {
        let mut world = World::new();
        let entity = spawn_zero_length_clip_instance(&mut world);

        let events = vec![UIEvent::ClipInstanceTrimEnd {
            entity,
            instance_id: 1,
            new_clip_out: 3.0,
        }];
        process_clip_instance_events(&events, &mut world);

        let (clip_in, clip_out) = instance_clip_range(&world, entity);
        assert!((clip_in - 0.0).abs() < 1e-6);
        assert!((clip_out - 3.0).abs() < 1e-6);
    }

    #[test]
    fn trim_end_never_drops_below_clip_in() {
        let mut world = World::new();
        let entity = spawn_zero_length_clip_instance(&mut world);

        let events = vec![
            UIEvent::ClipInstanceTrimStart {
                entity,
                instance_id: 1,
                new_clip_in: 0.0,
            },
            UIEvent::ClipInstanceTrimEnd {
                entity,
                instance_id: 1,
                new_clip_out: -2.0,
            },
        ];
        process_clip_instance_events(&events, &mut world);

        let (clip_in, clip_out) = instance_clip_range(&world, entity);
        assert!(clip_out >= clip_in);
    }

    #[test]
    fn trim_start_never_exceeds_clip_out() {
        let mut world = World::new();
        let entity = spawn_zero_length_clip_instance(&mut world);

        let events = vec![
            UIEvent::ClipInstanceTrimEnd {
                entity,
                instance_id: 1,
                new_clip_out: 2.0,
            },
            UIEvent::ClipInstanceTrimStart {
                entity,
                instance_id: 1,
                new_clip_in: 5.0,
            },
        ];
        process_clip_instance_events(&events, &mut world);

        let (clip_in, clip_out) = instance_clip_range(&world, entity);
        assert!((clip_in - 2.0).abs() < 1e-6);
        assert!((clip_out - 2.0).abs() < 1e-6);
    }

    #[test]
    fn drag_preview_trim_end_extends_zero_length_clip() {
        use crate::ecs::resource::ClipDragType;
        let (start, end) =
            clip_drag_preview_times(&ClipDragType::TrimEnd, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0);
        assert!((start - 0.0).abs() < 1e-6);
        assert!((end - 3.0).abs() < 1e-6);
    }

    #[test]
    fn drag_preview_mirrors_commit_clamps() {
        use crate::ecs::resource::ClipDragType;

        // TrimEnd never drops below clip_in
        let (_, end) =
            clip_drag_preview_times(&ClipDragType::TrimEnd, 2.0, -5.0, 1.0, 3.0, 1.0, 2.0);
        assert!((end - 1.0).abs() < 1e-6);

        // TrimStart clamps into [0, clip_out]; block start stays fixed
        let (start, end) =
            clip_drag_preview_times(&ClipDragType::TrimStart, 0.5, 9.0, 1.0, 3.0, 0.5, 2.0);
        assert!((start - 1.0).abs() < 1e-6);
        assert!((end - 1.0).abs() < 1e-6);

        // Move clamps at timeline zero and preserves width
        let (start, end) =
            clip_drag_preview_times(&ClipDragType::Move, 1.0, -5.0, 1.0, 3.0, 0.0, 2.0);
        assert!((start - 0.0).abs() < 1e-6);
        assert!((end - 2.0).abs() < 1e-6);
    }

    #[test]
    fn drag_preview_respects_playback_speed_scale() {
        use crate::ecs::resource::ClipDragType;
        // 1 clip-second spans 2 timeline-seconds (speed 0.5): extending
        // clip_out by 1 extends the block by 2.
        let (start, end) =
            clip_drag_preview_times(&ClipDragType::TrimEnd, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0);
        assert!((start - 1.0).abs() < 1e-6);
        assert!((end - 5.0).abs() < 1e-6);
    }
}
