use cgmath::Vector3;

use crate::asset::AssetStorage;
use crate::ecs::events::UIEvent;
use crate::ecs::resource::gizmo::{BoneGizmoData, BoneSelectionState};
use crate::ecs::resource::CurveEditorState;
use crate::ecs::resource::{Camera, ClipLibrary, HierarchyState, TimelineState};
use crate::ecs::systems::{
    camera_move_to_look_at, collapse_entity, expand_entity, hierarchy_collapse_bone,
    hierarchy_deselect_all, hierarchy_deselect_bone, hierarchy_expand_bone, hierarchy_select,
    hierarchy_select_bone, hierarchy_toggle_selection, rename_entity, resolve_mesh_bone_id,
    resolve_transform_entity, update_entity_scale, update_entity_translation,
    update_entity_visible,
};
use crate::ecs::world::{Children, Entity, Transform, World};

pub fn dispatch_hierarchy_events(
    events: &[UIEvent],
    world: &mut World,
    assets: &AssetStorage,
) -> Vec<super::super::ui_event_systems::DeferredAction> {
    let deferred = dispatch_hierarchy_entity_events(events, world);
    dispatch_hierarchy_bone_events(events, world, assets);
    sync_curve_editor_on_selection(events, world, assets);
    deferred
}

fn dispatch_hierarchy_entity_events(
    events: &[UIEvent],
    world: &mut World,
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
