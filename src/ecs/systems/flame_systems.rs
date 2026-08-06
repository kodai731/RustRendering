use crate::app::FrameContext;
use crate::asset::AssetStorage;
use crate::ecs::component::{
    apply_flame_param_value, EntityIcon, FlameBoneAttachment, FlameEffect, FlameParam, FlameTrail,
    FLAME_DOMAIN,
};
use crate::ecs::resource::{
    BatchRun, FlameRenderSettings, FlameTemporalSnapshot, FlameTemporalState, HierarchyState,
    LightState, ProjectionData, TimelineState,
};
use crate::ecs::world::{Entity, Transform, World};
use thyllore_render_core::{advance_flame_time, advance_flame_trail};

pub const DEFAULT_FLAME_NAME: &str = "Flame";

/// Spawns a flame as a regular scene entity so the hierarchy, inspector and transform gizmo
/// can all reach it through the same components they use for every other object.
pub fn spawn_flame(world: &mut World, name: &str, effect: FlameEffect) -> Entity {
    let transform = Transform {
        translation: effect.position,
        rotation: effect.rotation,
        ..Default::default()
    };

    world
        .entity()
        .with_name(name)
        .with_transform(transform)
        .with_editor_display(EntityIcon::Flame, false)
        .with_flame(effect)
        .build()
}

/// Spawn a flame entity together with its (empty) animation clip and schedule
/// instance, so every created flame is animatable and shows a Timeline clip
/// lane immediately instead of waiting for the first inserted key.
pub fn spawn_flame_with_clip(
    world: &mut World,
    assets: &mut AssetStorage,
    name: &str,
    effect: FlameEffect,
) -> Entity {
    let entity = spawn_flame(world, name, effect);
    super::scalar_clip_systems::ensure_entity_clip(world, assets, entity, &FLAME_DOMAIN);
    entity
}

/// The flame the UI and the flame events act on: the selected entity when it is a flame,
/// otherwise the first one. Keeping this in one place is what lets the hierarchy selection
/// stay the single source of truth.
pub fn resolve_selected_flame(world: &World) -> Option<Entity> {
    let selected = world
        .get_resource::<HierarchyState>()
        .and_then(|state| state.selected_entity);

    if let Some(entity) = selected {
        if world.get_component::<FlameEffect>(entity).is_some() {
            return Some(entity);
        }
    }

    world.query_flames().first().copied()
}

/// Position and rotation live on the Transform; the effect only mirrors them for the UBO.
pub fn write_flame_transform(
    world: &mut World,
    entity: Entity,
    translation: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
) {
    if let Some(transform) = world.get_component_mut::<Transform>(entity) {
        transform.translation = translation;
        transform.rotation = rotation;
    }
}

const STABLE_FRAME_HISTORY_WEIGHT: f32 = 0.85;

pub fn flame_time_advance(ctx: &mut FrameContext) {
    let light_position = ctx
        .world
        .get_resource::<LightState>()
        .map(|ls| ls.light_position);

    // Collect translations from Transform components to avoid borrow conflicts
    let flame_entities = ctx.world.query_flames();
    let transforms: Vec<(Entity, crate::ecs::world::Transform)> = flame_entities
        .iter()
        .filter_map(|&e| {
            ctx.world
                .get_component::<crate::ecs::world::Transform>(e)
                .map(|t| (e, t.clone()))
        })
        .collect();

    // Collect each flame's scheduled clip (scalar keyframe curves) up front to
    // avoid borrow conflicts while mutating FlameEffect below.
    let flame_clips: Vec<(Entity, crate::animation::editable::EditableAnimationClip)> = {
        let clip_library = ctx
            .world
            .get_resource::<crate::ecs::resource::ClipLibrary>();
        flame_entities
            .iter()
            .filter_map(|&e| {
                let clip_id = super::scalar_clip_systems::find_entity_clip_id(ctx.world, e)?;
                let clip = clip_library.as_ref()?.get(clip_id)?;
                Some((e, clip.clone()))
            })
            .collect()
    };
    let has_batch_run = ctx.world.contains_resource::<BatchRun>();
    let batch_frames_rendered = if has_batch_run {
        Some(ctx.world.resource::<BatchRun>().frames_rendered as f32)
    } else {
        None
    };
    let timeline_current_time = if has_batch_run {
        None
    } else {
        ctx.world
            .get_resource::<TimelineState>()
            .map(|ts| ts.current_time)
    };

    for &entity in &flame_entities {
        if let Some(mut effect) = ctx.world.get_component_mut::<FlameEffect>(entity) {
            if let Some(frames_rendered) = batch_frames_rendered {
                effect.time = frames_rendered * (1.0 / 60.0);
            } else if let Some(timeline_time) = timeline_current_time {
                effect.time = timeline_time * effect.time_scale + effect.time_offset;
            } else {
                advance_flame_time(&mut effect, ctx.delta_time);
            }
            if let Some(lp) = light_position {
                effect.light_position_world = lp;
            }
            // Sync position and rotation from Transform if present
            if let Some((_e, transform)) = transforms.iter().find(|(e, _)| *e == entity) {
                effect.position = transform.translation;
                effect.rotation = transform.rotation;
            }
            // Apply scheduled clip scalar curves (flame keyframes) if present
            if let Some((_e, clip)) = flame_clips.iter().find(|(e, _)| *e == entity) {
                for (property_type, value) in
                    super::scalar_clip_systems::sampled_scalar_values(clip, effect.time)
                {
                    if let Some(param) = FlameParam::from_property_type(property_type) {
                        apply_flame_param_value(&mut effect, param, value);
                    }
                }
            }
        }
    }
}

pub fn flame_bone_attach_sync(ctx: &mut FrameContext) {
    use crate::ecs::component::resolve_bone_index;
    use crate::ecs::resource::gizmo::BoneGizmoData;

    let (skeleton_id, cached_global_transforms, bone_local_offsets, mesh_scale) = {
        let bone_gizmo = match ctx.world.get_resource::<BoneGizmoData>() {
            Some(bg) => bg,
            None => return,
        };
        let skeleton_id = match bone_gizmo.cached_skeleton_id {
            Some(id) => id,
            None => return,
        };
        (
            skeleton_id,
            bone_gizmo.cached_global_transforms.clone(),
            bone_gizmo.bone_local_offsets.clone(),
            bone_gizmo.mesh_scale,
        )
    };

    let skeleton = match ctx.assets.get_skeleton_by_skeleton_id(skeleton_id) {
        Some(s) => s,
        None => return,
    };
    let bone_names: Vec<String> = skeleton.bones.iter().map(|b| b.name.clone()).collect();

    let positions =
        crate::ecs::systems::bone_gizmo_systems::compute_display_transforms_with_skeleton(
            &cached_global_transforms,
            &bone_local_offsets,
            mesh_scale,
            Some(&skeleton),
        );

    let flame_entities: Vec<Entity> = ctx.world.query_flames();
    for &entity in &flame_entities {
        let attachment = match ctx.world.get_component::<FlameBoneAttachment>(entity) {
            Some(a) => a,
            None => continue,
        };
        let idx = match resolve_bone_index(&attachment.bone, &bone_names) {
            Some(i) => i,
            None => continue,
        };
        if idx >= positions.len() {
            continue;
        }
        let translation =
            cgmath::Vector3::new(positions[idx][0], positions[idx][1], positions[idx][2]);
        if let Some(mut transform) = ctx
            .world
            .get_component_mut::<crate::ecs::world::Transform>(entity)
        {
            transform.translation = translation;
        } else {
            ctx.world.insert_component(
                entity,
                crate::ecs::world::Transform {
                    translation,
                    rotation: cgmath::Quaternion::new(0.0, 0.0, 0.0, 1.0),
                    scale: cgmath::Vector3::new(1.0, 1.0, 1.0),
                },
            );
        }
    }
}

pub fn flame_trail_advance(ctx: &mut FrameContext) {
    let flame_entities = ctx.world.query_flames();
    let has_batch_run = ctx.world.contains_resource::<BatchRun>();
    let batch_frames_rendered = if has_batch_run {
        Some(ctx.world.resource::<BatchRun>().frames_rendered as f32)
    } else {
        None
    };
    let timeline_current_time = if has_batch_run {
        None
    } else {
        ctx.world
            .get_resource::<TimelineState>()
            .map(|ts| ts.current_time)
    };

    for &entity in &flame_entities {
        // Read position before mutable borrow
        let position = match ctx.world.get_component::<FlameEffect>(entity) {
            Some(effect) => effect.position,
            None => continue,
        };
        let mut trail = match ctx.world.get_component_mut::<FlameTrail>(entity) {
            Some(t) => t,
            None => continue,
        };
        if !trail.state.enabled {
            continue;
        }
        let delta = if let Some(current_frame) = batch_frames_rendered {
            let last_frame = trail.last_timeline_time.unwrap_or(current_frame);
            current_frame - last_frame
        } else if let Some(timeline_time) = timeline_current_time {
            let last = trail.last_timeline_time.unwrap_or(timeline_time);
            timeline_time - last
        } else {
            ctx.delta_time
        };
        let pos: [f32; 3] = [position.x, position.y, position.z];
        advance_flame_trail(&mut trail.state, pos, delta);
        trail.last_timeline_time = batch_frames_rendered.or(timeline_current_time);
    }
}

/// Reusing the previous frame's shading is only valid while the camera and the flame
/// parameters hold still. Batch runs never reuse history so a single-frame screenshot
/// stays deterministic.
pub fn flame_temporal_accumulate(ctx: &mut FrameContext) {
    let flame_entities = ctx.world.query_flames();
    let count = flame_entities.len();

    if count == 0 {
        return;
    }

    // If there are 2 or more instances, only increment temporal_weight for all
    if count >= 2 {
        for &entity in &flame_entities {
            if let Some(mut effect) = ctx.world.get_component_mut::<FlameEffect>(entity) {
                effect.temporal_weight = 0.0;
                effect.frame_index = effect.frame_index.wrapping_add(1);
            }
        }
        return;
    }

    // Single instance: perform original snapshot comparison logic
    let entity = flame_entities[0];
    // Collect data first to avoid borrow conflicts
    let view = ctx.world.resource::<ProjectionData>().view;
    let settings = *ctx.world.resource::<FlameRenderSettings>();
    let has_batch_run = ctx.world.contains_resource::<BatchRun>();
    let old_effect = ctx.world.get_component::<FlameEffect>(entity).cloned();
    let Some(old_effect) = old_effect else {
        return;
    };

    let snapshot = FlameTemporalSnapshot {
        view,
        appearance: strip_per_frame_state(&old_effect),
        settings,
    };

    // Get matches_previous_frame before taking mutable borrow
    let state = ctx.world.resource::<FlameTemporalState>();
    let matches_previous_frame = state.previous.as_ref() == Some(&snapshot);
    drop(state);

    // Update the temporal state
    let mut state = ctx.world.resource_mut::<FlameTemporalState>();
    state.previous = Some(snapshot);
    drop(state);

    // Read batch frames_rendered before the mutable borrow to avoid conflicting borrows
    let batch_frames_rendered = if has_batch_run {
        Some(ctx.world.resource::<BatchRun>().frames_rendered)
    } else {
        None
    };

    // Now apply the changes to the component
    if let Some(mut effect) = ctx.world.get_component_mut::<FlameEffect>(entity) {
        effect.frame_index = if let Some(fr) = batch_frames_rendered {
            fr
        } else {
            old_effect.frame_index.wrapping_add(1)
        };
        effect.temporal_weight = if matches_previous_frame && !has_batch_run {
            STABLE_FRAME_HISTORY_WEIGHT
        } else {
            0.0
        };
    }
}

/// Fields that advance on their own every frame must be excluded, otherwise the snapshot
/// never compares equal and history would be discarded unconditionally.
fn strip_per_frame_state(effect: &FlameEffect) -> FlameEffect {
    let mut appearance = effect.clone();
    appearance.time = 0.0;
    appearance.temporal_weight = 0.0;
    appearance.frame_index = 0;
    appearance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::component::EditorDisplay;
    use crate::ecs::world::{GlobalTransform, Name};

    fn spawn_default_flame(world: &mut World, name: &str) -> Entity {
        spawn_flame(world, name, FlameEffect::default())
    }

    #[test]
    fn spawned_flame_carries_the_components_the_editor_queries() {
        let mut world = World::new();
        let entity = spawn_default_flame(&mut world, DEFAULT_FLAME_NAME);

        assert_eq!(
            world.get_component::<Name>(entity).map(|n| n.0.clone()),
            Some(DEFAULT_FLAME_NAME.to_string())
        );
        assert!(world.get_component::<Transform>(entity).is_some());
        assert!(world.get_component::<GlobalTransform>(entity).is_some());
        assert!(world.get_component::<EditorDisplay>(entity).is_some());
        assert!(world.get_component::<FlameEffect>(entity).is_some());
    }

    #[test]
    fn spawned_flame_appears_as_a_hierarchy_root() {
        let mut world = World::new();
        let entity = spawn_default_flame(&mut world, DEFAULT_FLAME_NAME);

        assert!(world.get_root_entities().contains(&entity));
    }

    #[test]
    fn spawn_mirrors_the_effect_position_onto_the_transform() {
        let mut world = World::new();
        let effect = FlameEffect {
            position: cgmath::Vector3::new(1.5, 2.5, -3.5),
            ..FlameEffect::default()
        };
        let entity = spawn_flame(&mut world, "Flame 2", effect);

        let transform = world.get_component::<Transform>(entity).unwrap();
        assert_eq!(transform.translation, cgmath::Vector3::new(1.5, 2.5, -3.5));
    }

    #[test]
    fn selection_falls_back_to_the_first_flame_when_nothing_is_selected() {
        let mut world = World::new();
        world.insert_resource(HierarchyState::default());
        let first = spawn_default_flame(&mut world, "Flame 1");
        spawn_default_flame(&mut world, "Flame 2");

        assert_eq!(resolve_selected_flame(&world), Some(first));
    }

    #[test]
    fn selection_follows_the_hierarchy_when_a_flame_is_selected() {
        let mut world = World::new();
        world.insert_resource(HierarchyState::default());
        spawn_default_flame(&mut world, "Flame 1");
        let second = spawn_default_flame(&mut world, "Flame 2");

        world
            .get_resource_mut::<HierarchyState>()
            .unwrap()
            .selected_entity = Some(second);

        assert_eq!(resolve_selected_flame(&world), Some(second));
    }

    #[test]
    fn selecting_a_non_flame_entity_keeps_editing_the_first_flame() {
        let mut world = World::new();
        world.insert_resource(HierarchyState::default());
        let first = spawn_default_flame(&mut world, "Flame 1");
        let other = world.entity().with_name("Not a flame").build();

        world
            .get_resource_mut::<HierarchyState>()
            .unwrap()
            .selected_entity = Some(other);

        assert_eq!(resolve_selected_flame(&world), Some(first));
    }

    #[test]
    fn write_flame_transform_moves_the_transform_not_the_effect() {
        let mut world = World::new();
        let entity = spawn_default_flame(&mut world, DEFAULT_FLAME_NAME);
        let translation = cgmath::Vector3::new(4.0, 0.0, 2.0);
        let rotation = cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0);

        write_flame_transform(&mut world, entity, translation, rotation);

        let transform = world.get_component::<Transform>(entity).unwrap();
        assert_eq!(transform.translation, translation);
    }

    #[test]
    fn resolve_returns_none_without_any_flame() {
        let mut world = World::new();
        world.insert_resource(HierarchyState::default());

        assert_eq!(resolve_selected_flame(&world), None);
    }
}
