use crate::app::FrameContext;
use crate::ecs::component::{apply_flame_track, FlameBoneAttachment, FlameTrack, FlameTrail};
use crate::ecs::resource::{
    BatchRun, FlameEffect, FlameRenderSettings, FlameTemporalSnapshot, FlameTemporalState,
    LightState, ProjectionData, TimelineState,
};
use crate::ecs::world::Entity;
use thyllore_render_core::{advance_flame_time, advance_flame_trail};

// Batch runs use a fixed timestep so noise-mode screenshots stay bit-deterministic.
const BATCH_FIXED_DELTA_SECONDS: f32 = 1.0 / 60.0;

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

    // Collect FlameTrack components to avoid borrow conflicts
    let flame_tracks: Vec<(Entity, FlameTrack)> = flame_entities
        .iter()
        .filter_map(|&e| {
            ctx.world
                .get_component::<FlameTrack>(e)
                .map(|t| (e, t.clone()))
        })
        .collect();

    let has_batch_run = ctx.world.contains_resource::<BatchRun>();
    let timeline_current_time = if has_batch_run {
        None
    } else {
        ctx.world
            .get_resource::<TimelineState>()
            .map(|ts| ts.current_time)
    };

    for &entity in &flame_entities {
        if let Some(mut effect) = ctx.world.get_component_mut::<FlameEffect>(entity) {
            if has_batch_run {
                advance_flame_time(&mut effect, BATCH_FIXED_DELTA_SECONDS);
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
            // Apply FlameTrack keyframe channels if present
            if let Some((_e, track)) = flame_tracks.iter().find(|(e, _)| *e == entity) {
                apply_flame_track(track, effect.time, &mut effect);
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
        let delta = if has_batch_run {
            BATCH_FIXED_DELTA_SECONDS
        } else if let Some(timeline_time) = timeline_current_time {
            let last = trail.last_timeline_time.unwrap_or(timeline_time);
            timeline_time - last
        } else {
            ctx.delta_time
        };
        let pos: [f32; 3] = [position.x, position.y, position.z];
        advance_flame_trail(&mut trail.state, pos, delta);
        trail.last_timeline_time = timeline_current_time;
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

    // Now apply the changes to the component
    if let Some(mut effect) = ctx.world.get_component_mut::<FlameEffect>(entity) {
        effect.frame_index = old_effect.frame_index.wrapping_add(1);
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
