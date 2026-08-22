use crate::app::FrameContext;
use crate::ecs::component::{FlameBaked, FlameEffect, FlameTemporalAccum};
use crate::ecs::resource::{
    BatchRun, FlameRenderSettings, FlameTemporalSnapshot, FlameTemporalState, ProjectionData,
};

const STABLE_FRAME_HISTORY_WEIGHT: f32 = 0.85;

/// Reusing the previous frame's shading is only valid while the camera and the flame
/// parameters hold still. Batch runs reuse history only when asked to
/// (`--batch-flame-history`): the jitter walk is keyed on `frames_rendered`, so the
/// blended result is still deterministic, but the default keeps the static grid the
/// existing bit-identity gates were recorded with.
pub fn flame_temporal_accumulate(ctx: &mut FrameContext) {
    let flame_entities = ctx.world.query_flames();
    let count = flame_entities.len();

    if count == 0 {
        return;
    }

    // If there are 2 or more instances, only increment the frame counter for all
    if count >= 2 {
        for &entity in &flame_entities {
            let next = ctx
                .world
                .get_component::<FlameTemporalAccum>(entity)
                .map(|t| t.frame_index.wrapping_add(1))
                .unwrap_or(0);
            ctx.world.insert_component(
                entity,
                FlameTemporalAccum {
                    weight: 0.0,
                    frame_index: next,
                },
            );
        }
        return;
    }

    // Single instance: perform original snapshot comparison logic
    let entity = flame_entities[0];
    // Collect data first to avoid borrow conflicts
    let view = ctx.world.resource::<ProjectionData>().view;
    let settings = *ctx.world.resource::<FlameRenderSettings>();
    let batch_run = ctx.world.get_resource::<BatchRun>();
    let has_batch_run = batch_run.is_some();
    let history_allowed = batch_run.map_or(true, |batch| batch.flame_history);
    let old_effect = ctx.world.get_component::<FlameEffect>(entity).cloned();
    let Some(old_effect) = old_effect else {
        return;
    };
    let baked = ctx
        .world
        .get_component::<FlameBaked>(entity)
        .cloned()
        .unwrap_or_default();
    let old_temporal = ctx
        .world
        .get_component::<FlameTemporalAccum>(entity)
        .cloned()
        .unwrap_or_default();

    let snapshot = FlameTemporalSnapshot {
        view,
        appearance: strip_per_frame_state(&old_effect),
        baked,
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

    let batch_frames_rendered = if has_batch_run {
        Some(ctx.world.resource::<BatchRun>().frames_rendered)
    } else {
        None
    };

    ctx.world.insert_component(
        entity,
        FlameTemporalAccum {
            frame_index: if let Some(fr) = batch_frames_rendered {
                fr
            } else {
                old_temporal.frame_index.wrapping_add(1)
            },
            weight: if matches_previous_frame && history_allowed {
                STABLE_FRAME_HISTORY_WEIGHT
            } else {
                0.0
            },
        },
    );
}

/// The clock advances on its own every frame and must be excluded, otherwise
/// the snapshot never compares equal and history would be discarded
/// unconditionally.
fn strip_per_frame_state(effect: &FlameEffect) -> FlameEffect {
    let mut appearance = effect.clone();
    appearance.time = 0.0;
    appearance
}
