use crate::app::FrameContext;
use crate::ecs::component::{apply_flame_param_value, FlameEffect, FlameParam};
use crate::ecs::resource::{BatchRun, LightState, TimelineState};
use crate::ecs::world::Entity;
use thyllore_render_core::advance_flame_time;

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
                let clip_id =
                    crate::ecs::systems::scalar_clip_systems::find_entity_clip_id(ctx.world, e)?;
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
                    crate::ecs::systems::scalar_clip_systems::sampled_scalar_values(
                        clip,
                        effect.time,
                    )
                {
                    if let Some(param) = FlameParam::from_property_type(property_type) {
                        apply_flame_param_value(&mut effect, param, value);
                    }
                }
            }
        }
    }
}
