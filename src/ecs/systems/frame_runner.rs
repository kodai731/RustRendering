use anyhow::Result;
use cgmath::Vector3;

#[cfg(feature = "ml")]
use super::curve_suggestion_systems::curve_suggestion_poll_results;
#[cfg(feature = "ml")]
use super::inference_actor_systems::{inference_actor_initialize, inference_actor_poll};
use super::object_picking_systems::apply_mesh_selection;
use super::phases::{
    run_animation_phase_ecs, run_animation_phase_gpu, run_input_phase, run_onion_skin_phase,
    run_render_prep_phase, run_transform_phase_ecs, run_transform_phase_gpu,
};
use super::timeline_systems::timeline_update;
use crate::app::FrameContext;
#[cfg(feature = "ml")]
use crate::ecs::component::InferenceActorSetup;
use crate::ecs::context::EcsContext;
use crate::ecs::resource::{ClipLibrary, HierarchyState, TimelineState};
#[cfg(feature = "ml")]
use crate::ecs::resource::{CurveSuggestionState, InferenceActorState};
use crate::ecs::world::Animator;
use crate::vulkanr::resource::graphics_resource::GraphicsResources;

pub unsafe fn run_frame(ctx: &mut FrameContext) -> Result<()> {
    let mesh_positions = collect_mesh_positions(ctx.graphics);

    {
        let mut ecs_ctx = EcsContext {
            time: ctx.time,
            delta_time: ctx.delta_time,
            image_index: ctx.image_index,
            swapchain_extent: ctx.swapchain_extent,
            world: ctx.world,
            assets: ctx.assets,
            mesh_positions,
        };

        process_pending_mesh_selection(&mut ecs_ctx);
        run_input_phase(&mut ecs_ctx)?;
        run_transform_phase_ecs(&mut ecs_ctx);
    }

    run_timeline_phase(ctx);
    #[cfg(feature = "ml")]
    run_inference_actor_phase(ctx);
    let animation_updates = run_animation_phase_ecs(ctx);
    run_animation_phase_gpu(ctx, &animation_updates)?;
    run_onion_skin_phase(ctx, &animation_updates.updated_meshes)?;

    run_transform_phase_gpu(ctx)?;
    run_render_prep_phase(ctx)?;
    Ok(())
}

fn process_pending_mesh_selection(ctx: &mut EcsContext) {
    if !ctx
        .world
        .contains_resource::<crate::ecs::resource::ObjectIdReadback>()
    {
        return;
    }

    let has_result = {
        let readback = ctx.object_id_readback();
        readback.last_read_object_id.is_some()
    };

    if !has_result {
        return;
    }

    let mut readback = ctx.object_id_readback_mut();
    let readback_clone = (*readback).clone();
    drop(readback);

    let mut readback_state = readback_clone;
    apply_mesh_selection(ctx.world, ctx.assets, &mut readback_state);

    let mut readback = ctx.object_id_readback_mut();
    readback.last_read_object_id = readback_state.last_read_object_id;
    readback.is_shift = readback_state.is_shift;
    readback.is_ctrl = readback_state.is_ctrl;
}

fn run_timeline_phase(ctx: &mut FrameContext) {
    if !ctx.world.contains_resource::<TimelineState>() {
        return;
    }
    if !ctx.world.contains_resource::<ClipLibrary>() {
        return;
    }

    let selected_entity = {
        let hierarchy_state = ctx.world.resource::<HierarchyState>();
        hierarchy_state.selected_entity
    };

    {
        let mut timeline_state = ctx.world.resource_mut::<TimelineState>();
        timeline_state.target_entity = selected_entity;
    }

    let mut timeline_state = ctx.world.resource_mut::<TimelineState>();
    let clip_library = ctx.world.resource::<ClipLibrary>();
    timeline_update(&mut timeline_state, &*clip_library, ctx.delta_time);
    drop(clip_library);
    drop(timeline_state);

    sync_timeline_to_all_animators(ctx);

    sync_editable_clips_to_registry(ctx);
}

fn sync_timeline_to_all_animators(ctx: &mut FrameContext) {
    let timeline_snapshot = {
        let timeline = ctx.world.resource::<TimelineState>();
        (
            timeline.current_time,
            timeline.playing,
            timeline.speed,
            timeline.looping,
        )
    };

    let animated_entities = ctx.world.query_animated();

    for entity in animated_entities {
        if let Some(animator) = ctx.world.get_component_mut::<Animator>(entity) {
            animator.time = timeline_snapshot.0;
            animator.playing = timeline_snapshot.1;
            animator.speed = timeline_snapshot.2;
            animator.looping = timeline_snapshot.3;
        }
    }
}

fn sync_editable_clips_to_registry(ctx: &mut FrameContext) {
    let mut clip_library = ctx.world.resource_mut::<ClipLibrary>();
    super::clip_library_systems::clip_library_sync_dirty(&mut clip_library, ctx.assets);
}

#[cfg(feature = "ml")]
fn run_inference_actor_phase(ctx: &mut FrameContext) {
    if !ctx.world.contains_resource::<InferenceActorState>() {
        return;
    }

    let setups: Vec<_> = ctx
        .world
        .iter_components::<InferenceActorSetup>()
        .map(|(_, setup)| setup.clone())
        .collect();

    let mut state = ctx.world.resource_mut::<InferenceActorState>();
    for setup in &setups {
        inference_actor_initialize(setup, &mut state);
    }
    inference_actor_poll(&mut state);

    if ctx.world.contains_resource::<CurveSuggestionState>() {
        let mut suggestion_state = ctx.world.resource_mut::<CurveSuggestionState>();
        curve_suggestion_poll_results(&mut suggestion_state, &mut state);
    }
}

fn collect_mesh_positions(graphics: &GraphicsResources) -> Vec<Vector3<f32>> {
    if graphics.meshes.is_empty() {
        return Vec::new();
    }

    graphics
        .meshes
        .iter()
        .flat_map(|mesh| {
            mesh.vertex_data
                .vertices
                .iter()
                .map(|v| Vector3::new(v.pos.x, v.pos.y, v.pos.z))
        })
        .collect()
}
