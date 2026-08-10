use crate::app::FrameContext;
use crate::ecs::component::{FlameEffect, FlameTrail};
use crate::ecs::resource::{BatchRun, TimelineState};
use thyllore_render_core::advance_flame_trail;

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
        let trail = match ctx.world.get_component_mut::<FlameTrail>(entity) {
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
