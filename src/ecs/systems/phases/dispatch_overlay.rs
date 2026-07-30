use crate::ecs::component::{channel_insert_key, FlameChannel, FlameParam, FlameTrack, FlameTrail};
use crate::ecs::events::UIEvent;
use crate::ecs::resource::gizmo::BoneGizmoData;
use crate::ecs::resource::{
    AutoExposure, DepthOfField, FlameCurveWindowState, FlameEffect, FlameRenderSettings,
    GridMeshData, HierarchyState, MessageLog, OnionSkinningConfig, PhysicalCameraParameters,
    SelectedFlameInstance, TimelineState, TransformGizmoState, WeightHeatmapState,
};
use crate::ecs::world::{Animator, World};
use thyllore_anim_core::editable::InterpolationType;

pub fn dispatch_overlay_events(events: &[UIEvent], world: &mut World) {
    for event in events {
        match event {
            UIEvent::SetBoneGizmoVisible(visible) => {
                if let Some(mut gizmo) = world.get_resource_mut::<BoneGizmoData>() {
                    gizmo.visible = *visible;
                }
            }
            UIEvent::SetWeightHeatmapEnabled(enabled) => {
                log!("UIEvent::SetWeightHeatmapEnabled({})", enabled);
                if let Some(mut heatmap) = world.get_resource_mut::<WeightHeatmapState>() {
                    heatmap.enabled = *enabled;
                } else {
                    log_warn!("WeightHeatmapState resource missing when toggling heatmap");
                }
            }
            UIEvent::SetTransformGizmoMode(mode) => {
                if let Some(mut state) = world.get_resource_mut::<TransformGizmoState>() {
                    state.mode = *mode;
                }
            }
            UIEvent::SetTransformGizmoSpace(space) => {
                if let Some(mut state) = world.get_resource_mut::<TransformGizmoState>() {
                    state.coordinate_space = *space;
                }
            }
            UIEvent::UpdateTransformGizmoState(new_state) => {
                if let Some(mut state) = world.get_resource_mut::<TransformGizmoState>() {
                    *state = *new_state.clone();
                }
            }
            UIEvent::UpdateDepthOfField(new_dof) => {
                if let Some(mut dof) = world.get_resource_mut::<DepthOfField>() {
                    *dof = new_dof.clone();
                }
            }
            UIEvent::UpdatePhysicalCamera(new_params) => {
                if let Some(mut params) = world.get_resource_mut::<PhysicalCameraParameters>() {
                    *params = new_params.clone();
                }
            }
            UIEvent::UpdateAutoExposure(new_ae) => {
                if let Some(mut ae) = world.get_resource_mut::<AutoExposure>() {
                    *ae = new_ae.clone();
                }
            }
            UIEvent::UpdateOnionSkinning(new_config) => {
                if new_config.enabled {
                    auto_select_animator_entity(world);
                }
                if let Some(mut config) = world.get_resource_mut::<OnionSkinningConfig>() {
                    *config = new_config.clone();
                }
            }
            UIEvent::UpdateFlameEffect(effect) => {
                let flames = world.query_flames();
                if flames.is_empty() {
                    continue;
                }
                let selected = world
                    .get_resource::<SelectedFlameInstance>()
                    .map(|s| s.0)
                    .unwrap_or(0);
                let idx = selected.min(flames.len() - 1);
                let target = flames[idx];
                if let Some(mut current) = world.get_component_mut::<FlameEffect>(target) {
                    *current = effect.as_ref().clone();
                }
            }
            UIEvent::UpdateFlameTrailEnabled(enabled) => {
                let flames = world.query_flames();
                if flames.is_empty() {
                    continue;
                }
                let selected = world
                    .get_resource::<SelectedFlameInstance>()
                    .map(|s| s.0)
                    .unwrap_or(0);
                let idx = selected.min(flames.len() - 1);
                let target = flames[idx];
                if let Some(mut trail) = world.get_component_mut::<FlameTrail>(target) {
                    trail.state.enabled = *enabled;
                } else {
                    world.insert_component(
                        target,
                        FlameTrail {
                            state: thyllore_render_core::FlameTrailState {
                                enabled: *enabled,
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                    );
                }
            }
            UIEvent::UpdateFlameTrailFade(fade) => {
                let flames = world.query_flames();
                if flames.is_empty() {
                    continue;
                }
                let selected = world
                    .get_resource::<SelectedFlameInstance>()
                    .map(|s| s.0)
                    .unwrap_or(0);
                let idx = selected.min(flames.len() - 1);
                let target = flames[idx];
                if let Some(mut trail) = world.get_component_mut::<FlameTrail>(target) {
                    trail.state.fade_seconds = *fade;
                } else {
                    world.insert_component(
                        target,
                        FlameTrail {
                            state: thyllore_render_core::FlameTrailState {
                                fade_seconds: *fade,
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                    );
                }
            }
            UIEvent::AddFlame => {
                let flames = world.query_flames();
                if flames.len() < thyllore_vulkan_core::resource::MAX_FLAME_INSTANCES {
                    let e = world.spawn();
                    world.insert_component(
                        e,
                        FlameEffect {
                            position: cgmath::Vector3::new(1.5 * flames.len() as f32, 0.0, 0.0),
                            ..FlameEffect::default()
                        },
                    );
                    world.insert_component(
                        e,
                        crate::ecs::world::Name(format!("Flame {}", flames.len() + 1)),
                    );
                }
            }
            UIEvent::InsertFlameKey { param, value } => {
                let current_time = world
                    .get_resource::<TimelineState>()
                    .map(|t| t.current_time)
                    .unwrap_or(0.0);
                let flames = world.query_flames();
                if flames.is_empty() {
                    continue;
                }
                let selected = world
                    .get_resource::<SelectedFlameInstance>()
                    .map(|s| s.0)
                    .unwrap_or(0);
                let idx = selected.min(flames.len() - 1);
                let target = flames[idx];
                let mut track =
                    if let Some(existing) = world.get_component_mut::<FlameTrack>(target) {
                        let mut track = existing.clone();
                        drop(existing);
                        track
                    } else {
                        FlameTrack::default()
                    };
                // Find or create the channel for this param
                let mut found = false;
                for channel in &mut track.channels {
                    if channel.param == *param {
                        // Update existing key or insert new one
                        let mut inserted = false;
                        for k in &mut channel.keys {
                            if (k.time - current_time).abs() < 1e-6 {
                                k.value = *value;
                                inserted = true;
                                break;
                            }
                        }
                        if !inserted {
                            channel_insert_key(
                                channel,
                                current_time,
                                *value,
                                InterpolationType::Linear,
                            );
                        }
                        found = true;
                        break;
                    }
                }
                if !found {
                    let mut channel = FlameChannel {
                        param: *param,
                        keys: vec![],
                        next_keyframe_id: 1,
                    };
                    channel_insert_key(
                        &mut channel,
                        current_time,
                        *value,
                        InterpolationType::Linear,
                    );
                    track.channels.push(channel);
                }
            }
            UIEvent::DeleteFlameKeysAt { time } => {
                let flames = world.query_flames();
                if flames.is_empty() {
                    continue;
                }
                let selected = world
                    .get_resource::<SelectedFlameInstance>()
                    .map(|s| s.0)
                    .unwrap_or(0);
                let idx = selected.min(flames.len() - 1);
                let target = flames[idx];
                let mut track = match world.get_component_mut::<FlameTrack>(target) {
                    Some(existing) => {
                        let mut track = existing.clone();
                        drop(existing);
                        track
                    }
                    None => continue,
                };
                let mut to_remove: Vec<usize> = Vec::new();
                for (i, channel) in track.channels.iter_mut().enumerate() {
                    channel.keys.retain(|key| (key.time - time).abs() > 0.02);
                    if channel.keys.is_empty() {
                        to_remove.push(i);
                    }
                }
                for i in to_remove.into_iter().rev() {
                    track.channels.remove(i);
                }
                if track.channels.is_empty() {
                    world.remove_component::<FlameTrack>(target);
                } else {
                    world.insert_component(target, track);
                }
            }
            UIEvent::ClearFlameKeys => {
                let flames = world.query_flames();
                if flames.is_empty() {
                    continue;
                }
                let selected = world
                    .get_resource::<SelectedFlameInstance>()
                    .map(|s| s.0)
                    .unwrap_or(0);
                let idx = selected.min(flames.len() - 1);
                let target = flames[idx];
                world.remove_component::<FlameTrack>(target);
            }
            UIEvent::SelectFlameInstance(index) => {
                let flames = world.query_flames();
                if flames.is_empty() {
                    continue;
                }
                let clamped = (*index as usize).min(flames.len() - 1);
                if let Some(mut selected) = world.get_resource_mut::<SelectedFlameInstance>() {
                    selected.0 = clamped;
                }
            }
            UIEvent::UpdateFlameRenderSettings(new_settings) => {
                if let Some(mut settings) = world.get_resource_mut::<FlameRenderSettings>() {
                    *settings = new_settings.clone();
                }
            }
            UIEvent::SetGridShowYAxis(show) => {
                if let Some(mut grid) = world.get_resource_mut::<GridMeshData>() {
                    grid.show_y_axis_grid = *show;
                }
            }
            UIEvent::ClearMessageLog => {
                if let Some(mut log) = world.get_resource_mut::<MessageLog>() {
                    crate::ecs::systems::message_log_clear_buffer(&mut log);
                }
            }
            UIEvent::ToggleFlameCurves => {
                if let Some(mut state) = world.get_resource_mut::<FlameCurveWindowState>() {
                    state.open = !state.open;
                }
            }
            UIEvent::ToggleFlameCurveParam { param } => {
                if let Some(mut state) = world.get_resource_mut::<FlameCurveWindowState>() {
                    if state.hidden_params.remove(param) {
                        // was hidden, now shown
                    } else {
                        // was shown, now hidden
                        state.hidden_params.insert(*param);
                    }
                }
            }
            UIEvent::MoveFlameKey {
                param,
                old_time,
                new_time,
                new_value,
            } => {
                let flames = world.query_flames();
                if flames.is_empty() {
                    continue;
                }
                let selected = world
                    .get_resource::<SelectedFlameInstance>()
                    .map(|s| s.0)
                    .unwrap_or(0);
                let idx = selected.min(flames.len() - 1);
                let target = flames[idx];
                let mut track = match world.get_component_mut::<FlameTrack>(target) {
                    Some(existing) => {
                        let mut track = existing.clone();
                        drop(existing);
                        track
                    }
                    None => continue,
                };
                // Find the channel for this param and remove the key within 1e-4 of old_time
                let mut found_channel = false;
                for channel in &mut track.channels {
                    if channel.param == *param {
                        channel
                            .keys
                            .retain(|key| (key.time - old_time).abs() > 1e-4);
                        // Insert new key at (new_time.max(0.0), new_value), replacing any existing key within 1e-4 of new_time
                        let clamped_time = new_time.max(0.0);
                        let mut inserted = false;
                        for k in &mut channel.keys {
                            if (k.time - clamped_time).abs() <= 1e-4 {
                                k.time = clamped_time;
                                k.value = *new_value;
                                inserted = true;
                                break;
                            }
                        }
                        if !inserted {
                            channel_insert_key(
                                channel,
                                clamped_time,
                                *new_value,
                                InterpolationType::Linear,
                            );
                        }
                        found_channel = true;
                        break;
                    }
                }
                if !found_channel {
                    // Channel doesn't exist yet, create it with the new key
                    let mut channel = FlameChannel {
                        param: *param,
                        keys: vec![],
                        next_keyframe_id: 1,
                    };
                    channel_insert_key(
                        &mut channel,
                        new_time.max(0.0),
                        *new_value,
                        InterpolationType::Linear,
                    );
                    track.channels.push(channel);
                }
                world.insert_component(target, track);
            }
            UIEvent::DeleteFlameKeyExact { param, time } => {
                let flames = world.query_flames();
                if flames.is_empty() {
                    continue;
                }
                let selected = world
                    .get_resource::<SelectedFlameInstance>()
                    .map(|s| s.0)
                    .unwrap_or(0);
                let idx = selected.min(flames.len() - 1);
                let target = flames[idx];
                let mut track = match world.get_component_mut::<FlameTrack>(target) {
                    Some(existing) => {
                        let mut track = existing.clone();
                        drop(existing);
                        track
                    }
                    None => continue,
                };
                // Find the channel for this param and remove keys within 1e-4 of time
                let mut found_channel = false;
                let mut to_remove: Vec<usize> = Vec::new();
                for (i, channel) in track.channels.iter_mut().enumerate() {
                    if channel.param == *param {
                        channel.keys.retain(|key| (key.time - time).abs() > 1e-4);
                        found_channel = true;
                    }
                    if channel.keys.is_empty() {
                        to_remove.push(i);
                    }
                }
                for i in to_remove.into_iter().rev() {
                    track.channels.remove(i);
                }
                // Keep the track component even if channels are empty (simpler than At variant)
                world.insert_component(target, track);
            }
            _ => {}
        }
    }
}

fn auto_select_animator_entity(world: &mut World) {
    let already_selected = world
        .get_resource::<HierarchyState>()
        .and_then(|h| h.selected_entity)
        .is_some();
    if already_selected {
        return;
    }

    let first_animator = world.iter_components::<Animator>().next().map(|(e, _)| e);
    if let Some(entity) = first_animator {
        let mut hierarchy = world.resource_mut::<HierarchyState>();
        crate::ecs::systems::hierarchy_select(&mut hierarchy, entity);
    }
}
