use crate::app::FrameContext;
use crate::ecs::component::{apply_water_param_value, WaterParam, WaterTorusEffect};
use crate::ecs::resource::{BatchRun, TimelineState, WaterRenderSettings};
use crate::ecs::world::Entity;
use thyllore_effect_core::advance_water_time;

#[derive(Clone, Copy)]
pub struct TimelineSample {
    pub current_time: f32,
    pub playing: bool,
}

#[derive(Clone, Copy)]
pub struct WaterTimeSources {
    pub batch_fixed_time: Option<f32>,
    pub batch_frames_rendered: Option<f32>,
    pub timeline: Option<TimelineSample>,
    pub delta_time: f32,
    pub free_run_when_paused: bool,
}

const BATCH_FRAME_DURATION: f32 = 1.0 / 60.0;

pub fn resolve_water_time(effect: &mut WaterTorusEffect, sources: WaterTimeSources) {
    if let Some(fixed_time) = sources.batch_fixed_time {
        effect.time = fixed_time;
        return;
    }
    if let Some(frames_rendered) = sources.batch_frames_rendered {
        effect.time = frames_rendered * BATCH_FRAME_DURATION;
        return;
    }

    match sources.timeline {
        Some(timeline) if timeline.playing => {
            effect.time = timeline.current_time * effect.time_scale + effect.time_offset;
        }
        Some(timeline) => {
            if sources.free_run_when_paused {
                advance_water_time(effect, sources.delta_time);
            } else {
                effect.time = timeline.current_time * effect.time_scale + effect.time_offset;
            }
        }
        None => advance_water_time(effect, sources.delta_time),
    }
}

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
    let (batch_fixed_time, free_run_when_paused) = ctx
        .world
        .get_resource::<WaterRenderSettings>()
        .map(|settings| (settings.batch_fixed_time, settings.free_run_when_paused))
        .unwrap_or((None, WaterRenderSettings::default().free_run_when_paused));
    let timeline_sample = if has_batch_run {
        None
    } else {
        ctx.world
            .get_resource::<TimelineState>()
            .map(|ts| TimelineSample {
                current_time: ts.current_time,
                playing: ts.playing,
            })
    };
    let time_sources = WaterTimeSources {
        batch_fixed_time,
        batch_frames_rendered,
        timeline: timeline_sample,
        delta_time: ctx.delta_time,
        free_run_when_paused,
    };

    for &entity in &water_entities {
        if let Some(mut effect) = ctx.world.get_component_mut::<WaterTorusEffect>(entity) {
            resolve_water_time(&mut effect, time_sources);
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
