use crate::app::FrameContext;
use crate::ecs::resource::{
    BatchRun, FlameEffect, FlameRenderSettings, FlameTemporalSnapshot, FlameTemporalState,
    ProjectionData,
};
use thyllore_render_core::advance_flame_time;

// Batch runs use a fixed timestep so noise-mode screenshots stay bit-deterministic.
const BATCH_FIXED_DELTA_SECONDS: f32 = 1.0 / 60.0;

const STABLE_FRAME_HISTORY_WEIGHT: f32 = 0.85;

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

/// Reusing the previous frame's shading is only valid while the camera and the flame
/// parameters hold still. Batch runs never reuse history so a single-frame screenshot
/// stays deterministic.
pub fn flame_temporal_accumulate(ctx: &mut FrameContext) {
    let Some(mut effect) = ctx.world.get_resource_mut::<FlameEffect>() else {
        return;
    };

    let snapshot = FlameTemporalSnapshot {
        view: ctx.world.resource::<ProjectionData>().view,
        appearance: strip_per_frame_state(&effect),
        settings: *ctx.world.resource::<FlameRenderSettings>(),
    };

    let mut state = ctx.world.resource_mut::<FlameTemporalState>();
    let matches_previous_frame = state.previous.as_ref() == Some(&snapshot);
    state.previous = Some(snapshot);

    effect.frame_index = effect.frame_index.wrapping_add(1);
    effect.temporal_weight = if matches_previous_frame && !ctx.world.contains_resource::<BatchRun>()
    {
        STABLE_FRAME_HISTORY_WEIGHT
    } else {
        0.0
    };
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
