use crate::app::FrameContext;
use crate::ecs::component::{apply_water_param_value, WaterParam, WaterTorusEffect};
use crate::ecs::resource::{BatchRun, TimelineState};
use crate::ecs::world::Entity;
use thyllore_effect_core::advance_water_time;

pub fn water_time_advance(ctx: &mut FrameContext) {
    // Collect translations from Transform components to avoid borrow conflicts
    let water_entities = ctx.world.query_waters();
    let transforms: Vec<(Entity, crate::ecs::world::Transform)> = water_entities
        .iter()
        .filter_map(|&e| {
            ctx.world
                .get_component::<crate::ecs::world::Transform>(e)
                .map(|t| (e, t.clone()))
        })
        .collect();

    // Collect each water's scheduled clip (scalar keyframe curves) up front to
    // avoid borrow conflicts while mutating WaterTorusEffect below.
    let water_clips: Vec<(Entity, crate::animation::editable::EditableAnimationClip)> = {
        let clip_library = ctx
            .world
            .get_resource::<crate::ecs::resource::ClipLibrary>();
        water_entities
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

    for &entity in &water_entities {
        if let Some(mut effect) = ctx.world.get_component_mut::<WaterTorusEffect>(entity) {
            if let Some(frames_rendered) = batch_frames_rendered {
                effect.time = frames_rendered * (1.0 / 60.0);
            } else if let Some(timeline_time) = timeline_current_time {
                effect.time = timeline_time * effect.time_scale + effect.time_offset;
            } else {
                advance_water_time(&mut effect, ctx.delta_time);
            }
            // Sync position and rotation from Transform if present
            if let Some((_e, transform)) = transforms.iter().find(|(e, _)| *e == entity) {
                effect.position = transform.translation;
                effect.rotation = transform.rotation;
            }
            // Apply scheduled clip scalar curves (water keyframes) if present
            if let Some((_e, clip)) = water_clips.iter().find(|(e, _)| *e == entity) {
                for (property_type, value) in
                    crate::ecs::systems::scalar_clip_systems::sampled_scalar_values(
                        clip,
                        effect.time,
                    )
                {
                    if let Some(param) = WaterParam::from_property_type(property_type) {
                        apply_water_param_value(&mut effect, param, value);
                    }
                }
            }
        }
    }
}
