use cgmath::Vector3;

use crate::asset::AssetStorage;
use crate::ecs::component::CameraAimTarget;
use crate::ecs::component::{CameraParam, CAMERA_DOMAIN};
use crate::ecs::events::UIEvent;
use crate::ecs::resource::gizmo::{BoneGizmoData, BoneSelectionState};
use crate::ecs::resource::CurveEditorState;
use crate::ecs::resource::HelmState;
use crate::ecs::resource::{ActiveCamera, Camera, ClipLibrary, HierarchyState, TimelineState};
use crate::ecs::systems::{
    camera_move_to_look_at, collapse_entity, compute_camera_direction, compute_camera_position,
    compute_camera_right, compute_camera_up, ensure_entity_clip, expand_entity,
    hierarchy_collapse_bone, hierarchy_deselect_all, hierarchy_deselect_bone,
    hierarchy_expand_bone, hierarchy_select, hierarchy_select_bone, hierarchy_toggle_selection,
    plan_camera_shot, rename_entity, resolve_mesh_bone_id, resolve_transform_entity,
    scalar_clip_insert_key, update_entity_scale, update_entity_translation, update_entity_visible,
};
use crate::ecs::world::{Children, Entity, Transform, World};
use thyllore_ml_core::copilot::camera_direction::{
    caption::build_movement_caption,
    generate_camera_poses,
    keyframes::{poses_to_keyframe_tuples, transform_poses_to_world, CameraKeyParam},
};

pub fn dispatch_hierarchy_events(
    events: &[UIEvent],
    world: &mut World,
    assets: &mut AssetStorage,
) -> Vec<super::super::ui_event_systems::DeferredAction> {
    let deferred = dispatch_hierarchy_entity_events(events, world, assets);
    dispatch_hierarchy_bone_events(events, world, assets);
    sync_curve_editor_on_selection(events, world, assets);
    deferred
}

/// glTF camera convention: local +X right, +Y up, looks down -Z.
fn camera_to_world_matrix(camera: &Camera) -> [[f32; 4]; 4] {
    let right = compute_camera_right(camera);
    let up = compute_camera_up(camera);
    let backward = -compute_camera_direction(camera);
    let position = compute_camera_position(camera);
    [
        [right.x, up.x, backward.x, position.x],
        [right.y, up.y, backward.y, position.y],
        [right.z, up.z, backward.z, position.z],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn camera_param_from_key(param: CameraKeyParam) -> CameraParam {
    match param {
        CameraKeyParam::TranslationX => CameraParam::TranslationX,
        CameraKeyParam::TranslationY => CameraParam::TranslationY,
        CameraKeyParam::TranslationZ => CameraParam::TranslationZ,
        CameraKeyParam::RotationX => CameraParam::RotationX,
        CameraKeyParam::RotationY => CameraParam::RotationY,
        CameraKeyParam::RotationZ => CameraParam::RotationZ,
    }
}

fn dispatch_hierarchy_entity_events(
    events: &[UIEvent],
    world: &mut World,
    assets: &mut AssetStorage,
) -> Vec<super::super::ui_event_systems::DeferredAction> {
    let mut deferred = Vec::new();

    for event in events {
        match event {
            UIEvent::SelectEntity(entity) => {
                let mut hierarchy_state = world.resource_mut::<HierarchyState>();
                hierarchy_select(&mut hierarchy_state, *entity);
            }

            UIEvent::DeselectAll => {
                let mut hierarchy_state = world.resource_mut::<HierarchyState>();
                hierarchy_deselect_all(&mut hierarchy_state);
            }

            UIEvent::ToggleEntitySelection(entity) => {
                let mut hierarchy_state = world.resource_mut::<HierarchyState>();
                hierarchy_toggle_selection(&mut hierarchy_state, *entity);
            }

            UIEvent::ExpandEntity(entity) => {
                expand_entity(world, *entity);
            }

            UIEvent::CollapseEntity(entity) => {
                collapse_entity(world, *entity);
            }

            UIEvent::SetSearchFilter(filter) => {
                let mut hierarchy_state = world.resource_mut::<HierarchyState>();
                hierarchy_state.search_filter = filter.clone();
            }

            UIEvent::SetHierarchyDisplayMode(mode) => {
                let mut hierarchy_state = world.resource_mut::<HierarchyState>();
                hierarchy_state.display_mode = *mode;
            }

            UIEvent::SetEntityVisible(entity, visible) => {
                update_entity_visible(world, *entity, *visible);
            }

            UIEvent::SetEntityTranslation(entity, translation) => {
                update_entity_translation(world, *entity, *translation);
            }

            UIEvent::SetEntityRotation(entity, rotation) => {
                let target = resolve_transform_entity(world, *entity);
                if let Some(transform) = world.get_component_mut::<Transform>(target) {
                    transform.rotation = *rotation;
                }
            }

            UIEvent::SetEntityScale(entity, scale) => {
                update_entity_scale(world, *entity, *scale);
            }

            UIEvent::RenameEntity(entity, new_name) => {
                rename_entity(world, *entity, new_name.clone());
            }

            UIEvent::DeleteSelectedEntities => {
                let hierarchy_state = world.resource::<HierarchyState>();
                let selected = hierarchy_state.selected_entity;
                let multi = hierarchy_state.multi_selection.clone();
                drop(hierarchy_state);

                let mut targets: Vec<Entity> = multi.into_iter().collect();
                if let Some(entity) = selected {
                    if !targets.contains(&entity) {
                        targets.push(entity);
                    }
                }

                let mut all_to_delete = Vec::new();
                for entity in &targets {
                    collect_entity_tree(world, *entity, &mut all_to_delete);
                }

                if !all_to_delete.is_empty() {
                    let mut hierarchy_state = world.resource_mut::<HierarchyState>();
                    hierarchy_state.selected_entity = None;
                    hierarchy_state.multi_selection.clear();

                    deferred.push(
                        super::super::ui_event_systems::DeferredAction::DeleteEntities {
                            entities: all_to_delete,
                        },
                    );
                }
            }

            UIEvent::FocusOnEntity(entity) => {
                let transform_entity = resolve_transform_entity(world, *entity);
                let target = world
                    .get_component::<Transform>(transform_entity)
                    .map(|t| t.translation);

                if let Some(target) = target {
                    let offset = Vector3::new(5.0, 3.0, 5.0);
                    let mut camera = world.resource_mut::<Camera>();
                    camera_move_to_look_at(&mut camera, target, offset);
                }
            }

            UIEvent::CameraShot {
                preset,
                speed,
                target,
            } => {
                let is_look_at_orbit = match preset {
                    crate::helm::components::tool_call::ShotPreset::LookAtSelection
                    | crate::helm::components::tool_call::ShotPreset::OrbitAroundSelection => true,
                    _ => false,
                };

                if is_look_at_orbit {
                    let active_camera = world.resource::<ActiveCamera>();
                    if let (Some(camera_entity), Some(target_entity)) = (active_camera.0, target) {
                        drop(active_camera);
                        let resolved_target = resolve_transform_entity(world, *target_entity);
                        world.insert_component(
                            camera_entity,
                            CameraAimTarget::look_at(resolved_target),
                        );
                        continue;
                    }
                }

                let is_dolly_crane = match preset {
                    crate::helm::components::tool_call::ShotPreset::DollyIn
                    | crate::helm::components::tool_call::ShotPreset::DollyOut
                    | crate::helm::components::tool_call::ShotPreset::CraneUp
                    | crate::helm::components::tool_call::ShotPreset::CraneDown => true,
                    _ => false,
                };

                if is_dolly_crane {
                    let active_camera = world.resource::<ActiveCamera>();
                    if let Some(camera_entity) = active_camera.0 {
                        drop(active_camera);

                        let camera = world.resource::<Camera>();
                        let forward = crate::ecs::systems::compute_camera_direction(&camera);
                        let current_pos = crate::ecs::systems::compute_camera_position(&camera);
                        drop(camera);

                        let duration = match speed {
                            crate::helm::components::tool_call::SpeedPreset::Slow => 2.0,
                            crate::helm::components::tool_call::SpeedPreset::Normal => 1.0,
                            crate::helm::components::tool_call::SpeedPreset::Fast => 0.5,
                        };

                        let target_pos = match preset {
                            crate::helm::components::tool_call::ShotPreset::DollyIn => {
                                current_pos + forward * 3.0
                            }
                            crate::helm::components::tool_call::ShotPreset::DollyOut => {
                                current_pos - forward * 3.0
                            }
                            crate::helm::components::tool_call::ShotPreset::CraneUp => {
                                Vector3::new(current_pos.x, current_pos.y + 3.0, current_pos.z)
                            }
                            crate::helm::components::tool_call::ShotPreset::CraneDown => {
                                Vector3::new(current_pos.x, current_pos.y - 3.0, current_pos.z)
                            }
                            _ => unreachable!(),
                        };

                        let timeline = world.resource::<TimelineState>();
                        let current_time = timeline.current_time;
                        let end_time = current_time + duration;
                        drop(timeline);

                        let clip_id = ensure_entity_clip(
                            world,
                            assets,
                            camera_entity,
                            &crate::ecs::component::CAMERA_DOMAIN,
                        );

                        for param in [
                            crate::ecs::component::CameraParam::TranslationX,
                            crate::ecs::component::CameraParam::TranslationY,
                            crate::ecs::component::CameraParam::TranslationZ,
                        ] {
                            let property_type = param.property_type();
                            let current_value = (crate::ecs::component::CAMERA_DOMAIN.read)(
                                world,
                                camera_entity,
                                property_type,
                            )
                            .unwrap_or(match param {
                                crate::ecs::component::CameraParam::TranslationX => current_pos.x,
                                crate::ecs::component::CameraParam::TranslationY => current_pos.y,
                                crate::ecs::component::CameraParam::TranslationZ => current_pos.z,
                                _ => 0.0,
                            });
                            let target_value = match param {
                                crate::ecs::component::CameraParam::TranslationX => target_pos.x,
                                crate::ecs::component::CameraParam::TranslationY => target_pos.y,
                                crate::ecs::component::CameraParam::TranslationZ => target_pos.z,
                                _ => 0.0,
                            };

                            super::dispatch_scalar_curve::edit_clip(
                                world,
                                clip_id,
                                "Camera shot",
                                |clip| {
                                    scalar_clip_insert_key(
                                        clip,
                                        property_type,
                                        current_time,
                                        current_value,
                                    );
                                    scalar_clip_insert_key(
                                        clip,
                                        property_type,
                                        end_time,
                                        target_value,
                                    );
                                },
                            );
                        }

                        continue;
                    }
                }
                let target_pos = if let Some(entity) = target {
                    let transform_entity = resolve_transform_entity(world, *entity);
                    world
                        .get_component::<Transform>(transform_entity)
                        .map(|t| t.translation)
                } else {
                    None
                };

                let camera = world.resource::<Camera>();
                let tween = plan_camera_shot(&camera, *preset, *speed, target_pos);
                drop(camera);

                let mut motion = world.resource_mut::<crate::ecs::systems::CameraShotMotion>();
                motion.active = Some(tween);
            }

            UIEvent::CameraDirection {
                utterance,
                target: _,
            } => {
                let active_camera = world.resource::<ActiveCamera>();
                let camera_entity = match active_camera.0 {
                    Some(e) => e,
                    None => {
                        let mut state = world.resource_mut::<HelmState>();
                        state.feedback = Some(crate::ecs::resource::CommandFeedback::Report(
                            "camera_direction: no active camera".to_string(),
                        ));
                        continue;
                    }
                };
                drop(active_camera);

                let paths =
                    match thyllore_ml_core::model_path::resolve_camera_direction_model_paths() {
                        Some(p) => p,
                        None => {
                            let mut state = world.resource_mut::<HelmState>();
                            state.feedback = Some(crate::ecs::resource::CommandFeedback::Report(
                                "camera_direction: model not found".to_string(),
                            ));
                            continue;
                        }
                    };

                let Some(caption) = build_movement_caption(utterance) else {
                    let mut state = world.resource_mut::<HelmState>();
                    state.feedback = Some(crate::ecs::resource::CommandFeedback::Report(format!(
                        "camera_direction: no movement keyword recognized in '{}'",
                        utterance
                    )));
                    continue;
                };

                let poses = match generate_camera_poses(&paths, &caption) {
                    Ok(p) => p,
                    Err(e) => {
                        let mut state = world.resource_mut::<HelmState>();
                        state.feedback =
                            Some(crate::ecs::resource::CommandFeedback::DispatchError(
                                format!("camera_direction: {}", e),
                            ));
                        continue;
                    }
                };

                let camera_to_world = {
                    let camera = world.resource::<Camera>();
                    camera_to_world_matrix(&camera)
                };
                let world_poses = transform_poses_to_world(&camera_to_world, &poses);
                let tuples = poses_to_keyframe_tuples(&world_poses, 30.0, 5);

                let current_time = world.resource::<TimelineState>().current_time;

                let clip_id = ensure_entity_clip(world, assets, camera_entity, &CAMERA_DOMAIN);

                let n = tuples.len();

                super::dispatch_scalar_curve::edit_clip(
                    world,
                    clip_id,
                    "Camera direction",
                    |clip| {
                        for (time, param, value) in tuples {
                            scalar_clip_insert_key(
                                clip,
                                camera_param_from_key(param).property_type(),
                                current_time + time,
                                value,
                            );
                        }
                    },
                );

                let mut state = world.resource_mut::<HelmState>();
                state.feedback = Some(crate::ecs::resource::CommandFeedback::Executed(format!(
                    "camera_direction: {} keys from '{}'",
                    n, caption
                )));
            }

            _ => {}
        }
    }

    deferred
}

fn dispatch_hierarchy_bone_events(events: &[UIEvent], world: &mut World, assets: &AssetStorage) {
    for event in events {
        match event {
            UIEvent::SelectBone(bone_id) => {
                let descendants: Vec<usize> = assets
                    .skeletons
                    .values()
                    .next()
                    .map(|skel_asset| {
                        skel_asset
                            .skeleton
                            .collect_descendants(*bone_id)
                            .into_iter()
                            .map(|id| id as usize)
                            .collect()
                    })
                    .unwrap_or_default();

                {
                    let mut hierarchy_state = world.resource_mut::<HierarchyState>();
                    hierarchy_select_bone(&mut hierarchy_state, *bone_id);
                }

                if let Some(mut selection) = world.get_resource_mut::<BoneSelectionState>() {
                    let bone_idx = *bone_id as usize;
                    selection.selected_bone_indices.clear();
                    selection.selected_bone_indices.insert(bone_idx);
                    for desc_idx in descendants {
                        selection.selected_bone_indices.insert(desc_idx);
                    }
                    selection.active_bone_index = Some(bone_idx);
                }
            }

            UIEvent::DeselectBone => {
                {
                    let mut hierarchy_state = world.resource_mut::<HierarchyState>();
                    hierarchy_deselect_bone(&mut hierarchy_state);
                }

                if let Some(mut selection) = world.get_resource_mut::<BoneSelectionState>() {
                    selection.selected_bone_indices.clear();
                    selection.active_bone_index = None;
                }
            }

            UIEvent::ExpandBone(bone_id) => {
                let mut hierarchy_state = world.resource_mut::<HierarchyState>();
                hierarchy_expand_bone(&mut hierarchy_state, *bone_id);
            }

            UIEvent::CollapseBone(bone_id) => {
                let mut hierarchy_state = world.resource_mut::<HierarchyState>();
                hierarchy_collapse_bone(&mut hierarchy_state, *bone_id);
            }

            UIEvent::SetBoneDisplayStyle(style) => {
                if let Some(mut bone_gizmo) = world.get_resource_mut::<BoneGizmoData>() {
                    bone_gizmo.display_style = *style;
                }
            }

            UIEvent::SetBoneInFront(in_front) => {
                if let Some(mut bone_gizmo) = world.get_resource_mut::<BoneGizmoData>() {
                    bone_gizmo.in_front = *in_front;
                }
            }

            UIEvent::SetBoneDistanceScaling(enabled) => {
                if let Some(mut bone_gizmo) = world.get_resource_mut::<BoneGizmoData>() {
                    bone_gizmo.distance_scaling_enabled = *enabled;
                }
            }

            UIEvent::SetBoneDistanceScaleFactor(factor) => {
                if let Some(mut bone_gizmo) = world.get_resource_mut::<BoneGizmoData>() {
                    bone_gizmo.distance_scaling_factor = *factor;
                }
            }

            _ => {}
        }
    }
}

fn sync_curve_editor_on_selection(events: &[UIEvent], world: &mut World, assets: &AssetStorage) {
    let is_open = world
        .get_resource::<CurveEditorState>()
        .map(|s| s.is_open)
        .unwrap_or(false);
    if !is_open {
        return;
    }

    for event in events {
        match event {
            UIEvent::SelectEntity(entity) => {
                let clip_library = world.resource::<ClipLibrary>();
                let source_id = world.resource::<TimelineState>().current_clip_id;
                let bone_id =
                    resolve_mesh_bone_id(world, *entity, assets, &clip_library, source_id);
                drop(clip_library);

                if let Some(bone_id) = bone_id {
                    let mut editor = world.resource_mut::<CurveEditorState>();
                    editor.select_bone(bone_id);
                }
            }

            UIEvent::SelectBone(bone_id) => {
                let has_track = {
                    let clip_library = world.resource::<ClipLibrary>();
                    let source_id = world.resource::<TimelineState>().current_clip_id;
                    source_id
                        .and_then(|id| clip_library.get(id))
                        .map(|clip| clip.tracks.contains_key(bone_id))
                        .unwrap_or(false)
                };

                if has_track {
                    let mut editor = world.resource_mut::<CurveEditorState>();
                    editor.select_bone(*bone_id);
                }
            }

            _ => {}
        }
    }
}

fn collect_entity_tree(world: &World, entity: Entity, out: &mut Vec<Entity>) {
    if out.contains(&entity) {
        return;
    }
    out.push(entity);

    let children = world
        .get_component::<Children>(entity)
        .map(|c| c.0.clone())
        .unwrap_or_default();

    for child in children {
        collect_entity_tree(world, child, out);
    }
}
