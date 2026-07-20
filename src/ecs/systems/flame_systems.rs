use crate::app::FrameContext;
use crate::ecs::resource::{BatchRun, FlameEffect};
use thyllore_render_core::advance_flame_time;

// Batch runs use a fixed timestep so noise-mode screenshots stay bit-deterministic.
const BATCH_FIXED_DELTA_SECONDS: f32 = 1.0 / 60.0;

pub fn flame_time_advance(ctx: &mut FrameContext) {
    let delta_time = if ctx.world.contains_resource::<BatchRun>() {
        BATCH_FIXED_DELTA_SECONDS
    } else {
        ctx.delta_time
    };

    if let Some(mut effect) = ctx.world.get_resource_mut::<FlameEffect>() {
        advance_flame_time(&mut effect, delta_time);
    }
}
