//! Orchestrator phase: processes submitted utterances and confirm responses.

use crate::ecs::context::EcsContext;
use crate::ecs::resource::{ClipLibrary, OrchestratorState, TimelineState};
use crate::ecs::systems::orchestrator::dispatcher::{dispatch_tool_call, DispatchOutcome};
use crate::ecs::systems::orchestrator::name_resolver::list_entity_names;
use crate::ecs::systems::orchestrator::timeline_context::build_timeline_context;
use crate::ecs::UIEventQueue;
use crate::orchestrator::components::route::Route;
use crate::orchestrator::components::tool_call::ToolCall;
use crate::orchestrator::systems::binder::{bind_route, BindOutcome};
use crate::orchestrator::systems::normalize::normalize_utterance;
use crate::orchestrator::systems::resolution::{confirm_reason, resolve_decision, ConfirmReason, ResolvedAction};
use crate::orchestrator::systems::router::{rank_routes, route_utterance, RoutingRequest};

use crate::ecs::events::UIEvent;
use std::io::Write;

/// Construct a jsonl entry for Reject / Clarify decisions (pure function).
/// Returns `None` for variants that are not logged (MissingObjectName, AmbiguousObjectName, NoCandidate).
fn reject_log_entry(
    normalized_utterance: &str,
    raw_top_score: f32,
    feedback: &crate::orchestrator::systems::resolution::OrchestratorFeedback,
) -> Option<serde_json::Value> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    match feedback {
        crate::orchestrator::systems::resolution::OrchestratorFeedback::Rejected { best, score } => {
            Some(serde_json::json!({
                "ts": ts,
                "utterance": normalized_utterance,
                "decision": "reject",
                "raw_top_score": raw_top_score,
                "details": {
                    "best": Route::tool_name(*best),
                    "score": *score,
                },
            }))
        }
        crate::orchestrator::systems::resolution::OrchestratorFeedback::ClarifyOptions(candidates) => {
            let items: Vec<serde_json::Value> = candidates
                .iter()
                .map(|(route, score)| {
                    serde_json::json!({
                        "route": Route::tool_name(*route),
                        "score": *score,
                    })
                })
                .collect();
            Some(serde_json::json!({
                "ts": ts,
                "utterance": normalized_utterance,
                "decision": "clarify",
                "raw_top_score": raw_top_score,
                "details": items,
            }))
        }
        _ => None, // MissingObjectName, AmbiguousObjectName, NoCandidate — not logged
    }
}

/// Append a pre-built jsonl entry to the reject log file (best-effort, errors ignored).
fn write_reject_log(entry: serde_json::Value) -> std::io::Result<()> {
    let log_dir = std::path::PathBuf::from("log");
    let _ = std::fs::create_dir_all(&log_dir);
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_dir.join("orchestrator_rejects.jsonl"))?;
    writeln!(file, "{}", serde_json::to_string(&entry).unwrap_or_default())?;
    Ok(())
}

#[derive(Clone, Debug)]
enum OrchestratorRuntimeKind {
    Uninitialized,
    Ready,
    Failed(String),
}

/// Run the orchestrator phase.
pub fn run_orchestrator_phase(ctx: &mut EcsContext) {
    // Copy state values before accessing other world resources to avoid borrow conflicts.
    let (submitted_utterance, confirm_response, clarify_choice, mode, thresholds, confirm_all) = {
        let mut state = ctx.world.resource_mut::<OrchestratorState>();
        (
            state.submitted_utterance.take(),
            state.confirm_response.take(),
            state.clarify_choice.take(),
            state.mode,
            state.thresholds,
            state.confirm_all,
        )
    };

    // Handle confirm_response first.
    if let Some(response) = confirm_response {
        let outcome = {
            let mut state = ctx.world.resource_mut::<OrchestratorState>();
            if let Some((call, reason)) = state.pending.take() {
                if response {
                    Some(execute_call(ctx.world, call, &state))
                } else {
                    state.feedback = Some(
                        crate::ecs::resource::CommandFeedback::Report("cancelled".to_string()),
                    );
                    None
                }
            } else {
                None
            }
        };
        if let Some(outcome) = outcome {
            handle_dispatch_outcome(ctx.world, outcome);
        }
    }

    if let Some(route) = clarify_choice {
        let normalized_utterance = {
            let state = ctx.world.resource::<OrchestratorState>();
            state.last_utterance.clone().unwrap_or_default()
        };
        let scene_names = list_entity_names(ctx.world);
        let outcome = bind_route(route, &normalized_utterance, &scene_names);

        match outcome {
            BindOutcome::Call(call) => {
                let reason = confirm_reason(&call, false, confirm_all);
                if let Some(reason) = reason {
                    let mut state = ctx.world.resource_mut::<OrchestratorState>();
                    state.pending = Some((call, reason));
                } else {
                    let outcome = execute_call(ctx.world, call, &ctx.world.resource::<OrchestratorState>());
                    handle_dispatch_outcome(ctx.world, outcome);
                }
            }
            BindOutcome::MissingSlot { route: _, slot } => {
                let mut state = ctx.world.resource_mut::<OrchestratorState>();
                state.feedback = Some(crate::ecs::resource::CommandFeedback::Router(
                    crate::orchestrator::systems::resolution::OrchestratorFeedback::MissingObjectName { route },
                ));
            }
            BindOutcome::AmbiguousSlot { route: _, candidates } => {
                let mut state = ctx.world.resource_mut::<OrchestratorState>();
                state.feedback = Some(crate::ecs::resource::CommandFeedback::Router(
                    crate::orchestrator::systems::resolution::OrchestratorFeedback::AmbiguousObjectName { candidates },
                ));
            }
        }
    }

    // Handle submitted_utterance.
    if let Some(utterance) = submitted_utterance {
        let runtime_kind: OrchestratorRuntimeKind = {
            let state = ctx.world.resource::<OrchestratorState>();
            match &state.runtime {
                crate::ecs::resource::RuntimeSlot::Uninitialized => {
                    OrchestratorRuntimeKind::Uninitialized
                }
                crate::ecs::resource::RuntimeSlot::Ready(_) => OrchestratorRuntimeKind::Ready,
                crate::ecs::resource::RuntimeSlot::Failed(msg) => {
                    OrchestratorRuntimeKind::Failed(msg.clone())
                }
            }
        };

        match runtime_kind {
            OrchestratorRuntimeKind::Uninitialized => {
                // Try to load the runtime.
                let result = crate::ecs::resource::load_runtime(std::path::Path::new(
                    crate::ecs::resource::ROUTER_MODEL_DIR,
                ));
                let mut state = ctx.world.resource_mut::<OrchestratorState>();
                match result {
                    Ok(runtime) => {
                        state.runtime =
                            crate::ecs::resource::RuntimeSlot::Ready(Box::new(runtime));
                    }
                    Err(e) => {
                        state.runtime =
                            crate::ecs::resource::RuntimeSlot::Failed(e.clone());
                        state.feedback = Some(
                            crate::ecs::resource::CommandFeedback::Unavailable(e),
                        );
                        return;
                    }
                }
            }
            OrchestratorRuntimeKind::Ready => {
                // Process the utterance: normalize -> encode -> route -> resolve.
                let normalized = normalize_utterance(&utterance);
                {
                    let mut state = ctx.world.resource_mut::<OrchestratorState>();
                    state.last_utterance = Some(normalized.clone());
                }
                let scene_names = list_entity_names(ctx.world);

                let (decision, raw_top_score) = {
                    let mut state = ctx.world.resource_mut::<OrchestratorState>();
                    match &mut state.runtime {
                        crate::ecs::resource::RuntimeSlot::Ready(rt) => {
                            let vector: Vec<f32> = match rt.encoder.encode(&normalized) {
                                Ok(v) => v,
                                Err(e) => {
                                    drop(state);
                                    let mut state = ctx.world.resource_mut::<OrchestratorState>();
                                    state.feedback = Some(
                                        crate::ecs::resource::CommandFeedback::DispatchError(format!(
                                            "encoding failed: {}",
                                            e
                                        )),
                                    );
                                    return;
                                }
                            };

                            let raw_vector: Vec<f32> = match rt.raw_encoder.encode(&normalized) {
                                Ok(v) => v,
                                Err(e) => {
                                    drop(state);
                                    let mut state = ctx.world.resource_mut::<OrchestratorState>();
                                    state.feedback = Some(
                                        crate::ecs::resource::CommandFeedback::DispatchError(format!(
                                            "raw encoding failed: {}",
                                            e
                                        )),
                                    );
                                    return;
                                }
                            };

                            let raw_top_score = {
                                let mut ranked = rank_routes(&rt.raw_index, &raw_vector, mode);
                                ranked.pop().map(|(_, score)| score)
                            };

                            let decision = route_utterance(
                                RoutingRequest {
                                    utterance: &normalized,
                                    query_vector: &vector,
                                    mode,
                                    raw_top_score,
                                },
                                &rt.index,
                                thresholds,
                            );
                            (decision, raw_top_score)
                        }
                        _ => {
                            drop(state);
                            return;
                        }
                    }
                };

                let action = resolve_decision(decision, &normalized, &scene_names, confirm_all);
                match action {
                    ResolvedAction::Dispatch(call) => {
                        let outcome = execute_call(ctx.world, call, &ctx.world.resource::<OrchestratorState>());
                        handle_dispatch_outcome(ctx.world, outcome);
                    }
                    ResolvedAction::AwaitConfirm { call, reason } => {
                        let mut state = ctx.world.resource_mut::<OrchestratorState>();
                        state.pending = Some((call, reason));
                    }
                    ResolvedAction::Feedback(f) => {
                        if let Some(entry) = reject_log_entry(&normalized, raw_top_score.unwrap_or(0.0), &f) {
                            let _ = write_reject_log(entry);
                        }

                        let mut state = ctx.world.resource_mut::<OrchestratorState>();
                        state.feedback = Some(
                            crate::ecs::resource::CommandFeedback::Router(f),
                        );
                    }
                }
            }
            OrchestratorRuntimeKind::Failed(e) => {
                // Don't retry — feedback is already set from the failed load.
                let mut state = ctx.world.resource_mut::<OrchestratorState>();
                state.feedback = Some(
                    crate::ecs::resource::CommandFeedback::Unavailable(e),
                );
            }
        }
    }
}

/// Execute a tool call and return its outcome.
fn execute_call(
    world: &crate::ecs::world::World,
    call: ToolCall,
    state: &OrchestratorState,
) -> DispatchOutcome {
    let timeline_state = world.resource::<TimelineState>();
    let clip_library = world.resource::<ClipLibrary>();
    let timeline_ctx = build_timeline_context(&timeline_state, &clip_library);
    dispatch_tool_call(world, &timeline_ctx, &call)
}

/// Handle a dispatch outcome by updating the UI event queue and feedback.
fn handle_dispatch_outcome(world: &crate::ecs::world::World, outcome: DispatchOutcome) {
    match outcome {
        DispatchOutcome::Command(event) => {
            let tool_name = {
                let mut state = world.resource_mut::<OrchestratorState>();
                let name = if let Some(ref pending) = state.pending {
                    pending.0.tool_name().to_string()
                } else {
                    "unknown".to_string()
                };
                drop(state);
                name
            };
            let mut ui_events = world.resource_mut::<UIEventQueue>();
            ui_events.send(event);
            let mut state = world.resource_mut::<OrchestratorState>();
            state.feedback = Some(
                crate::ecs::resource::CommandFeedback::Executed(tool_name),
            );
        }
        DispatchOutcome::Report(s) => {
            let mut state = world.resource_mut::<OrchestratorState>();
            state.feedback = Some(crate::ecs::resource::CommandFeedback::Report(s));
        }
        DispatchOutcome::MotionRequest { category, speed } => {
            let speed_mult = speed.to_multiplier();
            let category_str = category.as_str();

            // (a) Load motion seed catalog
            let catalog = match crate::orchestrator::systems::motion_seed::load_motion_seed_catalog() {
                Ok(c) => c,
                Err(e) => {
                    let mut state = world.resource_mut::<OrchestratorState>();
                    state.feedback = Some(
                        crate::ecs::resource::CommandFeedback::Unavailable(e),
                    );
                    return;
                }
            };

            // (b) Collect clip names from ClipLibrary
            let clip_library = world.resource::<ClipLibrary>();
            let clip_names: Vec<(crate::animation::editable::SourceClipId, String)> = clip_library
                .source_clips
                .values()
                .map(|c| (c.id, c.name().to_string()))
                .collect();
            drop(clip_library);

            // (c) Find seed candidates
            let candidates = crate::orchestrator::systems::motion_seed::find_seed_candidates(
                &catalog,
                category,
                &clip_names,
            );

            if candidates.is_empty() {
                let mut state = world.resource_mut::<OrchestratorState>();
                state.feedback = Some(crate::ecs::resource::CommandFeedback::Report(format!(
                    "no loaded clip matches motion category '{}'",
                    category_str
                )));
                return;
            }

            // (d) Target entity: HierarchyState.selected_entity, else TimelineState.target_entity
            let target_entity = {
                let hierarchy = world.resource::<crate::ecs::resource::HierarchyState>();
                hierarchy.selected_entity
            };
            let target_entity = target_entity.or_else(|| {
                let timeline = world.resource::<TimelineState>();
                timeline.target_entity
            });

            let target_entity = match target_entity {
                Some(e) => e,
                None => {
                    let mut state = world.resource_mut::<OrchestratorState>();
                    state.feedback = Some(crate::ecs::resource::CommandFeedback::Report(
                        "select a target entity first".to_string(),
                    ));
                    return;
                }
            };

            // (e) Round-robin counter
            let mut state = world.resource_mut::<OrchestratorState>();
            let counter = state.motion_seed_counters.entry(category).or_insert(0);
            let pick_index = *counter;
            *counter += 1;
            drop(state);

            let picked = crate::orchestrator::systems::motion_seed::pick_round_robin(&candidates, pick_index);
            let (source_id, clip_name) = match picked {
                Some(p) => p,
                None => {
                    let mut state = world.resource_mut::<OrchestratorState>();
                    state.feedback = Some(crate::ecs::resource::CommandFeedback::Report(
                        "no candidate found".to_string(),
                    ));
                    return;
                }
            };

            // (f) Get timeline current_time and push UIEvent::ClipInstanceAdd
            let current_time = {
                let timeline = world.resource::<TimelineState>();
                timeline.current_time
            };

            let mut ui_events = world.resource_mut::<UIEventQueue>();
            ui_events.send(UIEvent::ClipInstanceAdd {
                entity: target_entity,
                source_id,
                start_time: current_time,
                speed: speed_mult,
            });

            let mut state = world.resource_mut::<OrchestratorState>();
            state.feedback = Some(crate::ecs::resource::CommandFeedback::Executed(format!(
                "generate_motion {}: {}",
                category_str, clip_name
            )));
        }
        DispatchOutcome::Rejected(e) => {
            let mut state = world.resource_mut::<OrchestratorState>();
            state.feedback = Some(
                crate::ecs::resource::CommandFeedback::DispatchError(format!("{:?}", e)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::events::UIEvent;
    use crate::ecs::world::{Entity, Name, World};

    fn make_world() -> World {
        let mut world = World::new();
        world.insert_resource(crate::ecs::resource::HierarchyState::default());
        world.insert_resource(TimelineState::new());
        world.insert_resource(ClipLibrary::new());
        world.insert_resource(OrchestratorState::default());
        world.insert_resource(UIEventQueue::new());
        world
    }

    /// Test that ListObjects dispatch produces Report feedback.
    #[test]
    fn test_list_objects_dispatch_produces_report_feedback() {
        let mut world = make_world();
        let entity = world.spawn();
        world.insert_component(entity, Name("Hero".to_string()));

        // Manually execute a ListObjects call and check the outcome.
        let timeline_state = world.resource::<TimelineState>();
        let clip_library = world.resource::<ClipLibrary>();
        let timeline_ctx = build_timeline_context(&timeline_state, &clip_library);
        let outcome = dispatch_tool_call(&world, &timeline_ctx, &ToolCall::ListObjects);

        match outcome {
            DispatchOutcome::Report(s) => {
                assert!(s.contains("Hero"), "expected report about entities, got: {}", s);
            }
            other => panic!("expected Report, got {:?}", other),
        }
    }

    /// Test that PlayAnimation dispatch produces a Command (UIEvent) outcome.
    #[test]
    fn test_play_animation_dispatch_produces_command() {
        let mut world = make_world();

        let timeline_state = world.resource::<TimelineState>();
        let clip_library = world.resource::<ClipLibrary>();
        let timeline_ctx = build_timeline_context(&timeline_state, &clip_library);
        let outcome = dispatch_tool_call(&world, &timeline_ctx, &ToolCall::PlayAnimation);

        match outcome {
            DispatchOutcome::Command(event) => {
                assert!(matches!(event, UIEvent::TimelinePlay), "expected TimelinePlay, got {:?}", event);
            }
            other => panic!("expected Command, got {:?}", other),
        }
    }

    /// Test that handle_dispatch_outcome for PlayAnimation pushes to UIEventQueue.
    #[test]
    fn test_play_animation_pushes_to_ui_event_queue() {
        let mut world = make_world();

        // Set up a pending call so handle_dispatch_outcome can extract the tool name.
        {
            let mut state = world.resource_mut::<OrchestratorState>();
            state.pending = Some((ToolCall::PlayAnimation, ConfirmReason::ConfirmAll));
        }

        let timeline_state = world.resource::<TimelineState>();
        let clip_library = world.resource::<ClipLibrary>();
        let timeline_ctx = build_timeline_context(&timeline_state, &clip_library);
        let outcome = dispatch_tool_call(&world, &timeline_ctx, &ToolCall::PlayAnimation);

        let events_before = {
            let queue = world.resource::<UIEventQueue>();
            queue.len()
        };

        handle_dispatch_outcome(&world, outcome);

        let events_after = {
            let queue = world.resource::<UIEventQueue>();
            queue.len()
        };

        assert_eq!(events_after - events_before, 1, "expected 1 event pushed");

        let feedback = {
            let state = world.resource::<OrchestratorState>();
            state.feedback.clone()
        };

        match feedback {
            Some(crate::ecs::resource::CommandFeedback::Executed(name)) => {
                assert_eq!(name, "play_animation", "expected play_animation tool name");
            }
            other => panic!("expected Executed feedback, got {:?}", other),
        }
    }

    /// Test that confirm_response=false clears pending and sets Report("cancelled").
    #[test]
    fn test_confirm_response_false_clears_pending() {
        let mut world = make_world();

        // Set up a pending call.
        {
            let mut state = world.resource_mut::<OrchestratorState>();
            state.pending = Some((ToolCall::PlayAnimation, ConfirmReason::ConfirmAll));
            state.confirm_response = Some(false);
        }

        // Run the phase.
        let mut ctx = EcsContext {
            time: 0.0,
            delta_time: 0.0,
            image_index: 0,
            swapchain_extent: (800, 600),
            world: &mut world,
            assets: &mut crate::asset::AssetStorage::new(),
            mesh_positions: Vec::new(),
        };
        run_orchestrator_phase(&mut ctx);

        let state = world.resource::<OrchestratorState>();
        assert!(state.pending.is_none(), "expected pending to be cleared");
        match &state.feedback {
            Some(crate::ecs::resource::CommandFeedback::Report(msg)) => {
                assert_eq!(msg, "cancelled", "expected 'cancelled' report");
            }
            other => panic!("expected Report feedback, got {:?}", other),
        }
    }

    /// Test that clarify_choice with Route::PlayAnimation enters pending state.
    #[test]
    fn test_clarify_choice_play_animation_enters_pending() {
        let mut world = make_world();

        // Set up clarify_choice with PlayAnimation and confirm_all=false.
        {
            let mut state = world.resource_mut::<OrchestratorState>();
            state.clarify_choice = Some(Route::PlayAnimation);
            state.confirm_all = false;
        }

        let events_before = {
            let queue = world.resource::<UIEventQueue>();
            queue.len()
        };

        // Run the phase.
        let mut ctx = EcsContext {
            time: 0.0,
            delta_time: 0.0,
            image_index: 0,
            swapchain_extent: (800, 600),
            world: &mut world,
            assets: &mut crate::asset::AssetStorage::new(),
            mesh_positions: Vec::new(),
        };
        run_orchestrator_phase(&mut ctx);

        let events_after = {
            let queue = world.resource::<UIEventQueue>();
            queue.len()
        };

        // PlayAnimation is Mutating, so confirm_reason returns Some(Mutating) even with confirm_all=false.
        // It should enter pending, not execute immediately.
        assert_eq!(events_after, events_before, "expected no events pushed (should be pending)");

        let state = world.resource::<OrchestratorState>();
        assert!(state.clarify_choice.is_none(), "expected clarify_choice to be consumed");
        assert!(state.pending.is_some(), "expected pending to be set for PlayAnimation");
    }

    /// Test that clarify_choice with Route::ListObjects executes immediately.
    #[test]
    fn test_clarify_choice_list_objects_immediate_execution() {
        let mut world = make_world();
        let entity = world.spawn();
        world.insert_component(entity, Name("Hero".to_string()));

        // Set up clarify_choice with ListObjects (read-only).
        {
            let mut state = world.resource_mut::<OrchestratorState>();
            state.confirm_all = false;
            state.clarify_choice = Some(Route::ListObjects);
        }

        // Run the phase.
        let mut ctx = EcsContext {
            time: 0.0,
            delta_time: 0.0,
            image_index: 0,
            swapchain_extent: (800, 600),
            world: &mut world,
            assets: &mut crate::asset::AssetStorage::new(),
            mesh_positions: Vec::new(),
        };
        run_orchestrator_phase(&mut ctx);

        let state = world.resource::<OrchestratorState>();
        assert!(state.clarify_choice.is_none(), "expected clarify_choice to be consumed");
        assert!(state.pending.is_none(), "expected no pending (should execute immediately)");
        match &state.feedback {
            Some(crate::ecs::resource::CommandFeedback::Report(msg)) => {
                assert!(msg.contains("Hero"), "expected report about entities, got: {}", msg);
            }
            other => panic!("expected Report feedback, got {:?}", other),
        }
    }

    /// Test that clarify_choice with Route::ListObjects is pending when confirm_all is on.
    #[test]
    fn test_clarify_choice_list_objects_pending_when_confirm_all_on() {
        let mut world = make_world();
        let entity = world.spawn();
        world.insert_component(entity, Name("Hero".to_string()));

        // Set up clarify_choice with ListObjects while confirm_all is true.
        {
            let mut state = world.resource_mut::<OrchestratorState>();
            state.confirm_all = true;
            state.clarify_choice = Some(Route::ListObjects);
        }

        // Run the phase.
        let mut ctx = EcsContext {
            time: 0.0,
            delta_time: 0.0,
            image_index: 0,
            swapchain_extent: (800, 600),
            world: &mut world,
            assets: &mut crate::asset::AssetStorage::new(),
            mesh_positions: Vec::new(),
        };
        run_orchestrator_phase(&mut ctx);

        let mut state = world.resource_mut::<OrchestratorState>();
        assert!(state.clarify_choice.is_none(), "expected clarify_choice to be consumed");
        let (call, reason) = state.pending.take().expect("expected pending to be set");
        assert!(matches!(call, ToolCall::ListObjects), "expected ListObjects call");
        assert!(matches!(reason, ConfirmReason::ConfirmAll), "expected ConfirmReason::ConfirmAll, got {:?}", reason);
    }

    /// Test that MotionRequest dispatch produces ClipInstanceAdd event with correct speed.
    #[test]
    fn test_motion_request_walk_produces_clip_instance_add() {
        let mut world = make_world();
        // Insert a source clip named "Walk_Loop" into ClipLibrary
        {
            let mut library = world.resource_mut::<ClipLibrary>();
            let source_id: crate::animation::editable::SourceClipId = 1;
            let clip = crate::animation::editable::EditableAnimationClip::new(source_id, "Walk_Loop".to_string());
            let source = crate::animation::editable::SourceClip::new(source_id, clip);
            library.source_clips.insert(source_id, source);
        }

        // Spawn an entity with ClipSchedule and set as selected_entity
        let entity = world.spawn();
        world.insert_component(entity, crate::ecs::component::ClipSchedule::default());
        {
            let mut hierarchy = world.resource_mut::<crate::ecs::resource::HierarchyState>();
            hierarchy.selected_entity = Some(entity);
        }

        // Dispatch MotionRequest for Walk
        let timeline_state = world.resource::<TimelineState>();
        let clip_library = world.resource::<ClipLibrary>();
        let timeline_ctx = build_timeline_context(&timeline_state, &clip_library);
        let outcome = dispatch_tool_call(
            &world,
            &timeline_ctx,
            &ToolCall::GenerateMotion(crate::orchestrator::components::tool_call::MotionCategory::Walk, crate::orchestrator::components::tool_call::SpeedPreset::Normal),
        );

        handle_dispatch_outcome(&world, outcome);

        // Assert UIEventQueue has 1 ClipInstanceAdd with speed 1.0
        {
            let queue = world.resource::<UIEventQueue>();
            assert_eq!(queue.len(), 1, "expected 1 event in queue");
            match &queue[0] {
                UIEvent::ClipInstanceAdd { entity: e, speed, .. } => {
                    assert_eq!(*e, entity, "expected target entity to match");
                    assert_eq!(*speed, 1.0, "expected speed 1.0 for Normal");
                }
                other => panic!("expected ClipInstanceAdd, got {:?}", other),
            }
        }

        // Assert feedback is Executed
        {
            let state = world.resource::<OrchestratorState>();
            match &state.feedback {
                Some(crate::ecs::resource::CommandFeedback::Executed(_)) => {}
                other => panic!("expected Executed feedback, got {:?}", other),
            }
        }

        // Assert motion_seed_counters[Walk] == 1
        {
            let state = world.resource::<OrchestratorState>();
            let counter = state.motion_seed_counters.get(&crate::orchestrator::components::tool_call::MotionCategory::Walk);
            assert_eq!(counter, Some(&1), "expected Walk counter to be 1, got {:?}", counter);
        }
    }

    /// Test that calling MotionRequest again increments counter and adds another event.
    #[test]
    fn test_motion_request_walk_second_call_increments_counter() {
        let mut world = make_world();

        // Insert a source clip named "Walk_Loop" into ClipLibrary
        {
            let mut library = world.resource_mut::<ClipLibrary>();
            let source_id: crate::animation::editable::SourceClipId = 1;
            let clip = crate::animation::editable::EditableAnimationClip::new(source_id, "Walk_Loop".to_string());
            let source = crate::animation::editable::SourceClip::new(source_id, clip);
            library.source_clips.insert(source_id, source);
        }

        // Spawn an entity with ClipSchedule and set as selected_entity
        let entity = world.spawn();
        world.insert_component(entity, crate::ecs::component::ClipSchedule::default());
        {
            let mut hierarchy = world.resource_mut::<crate::ecs::resource::HierarchyState>();
            hierarchy.selected_entity = Some(entity);
        }

        // First call
        let timeline_state = world.resource::<TimelineState>();
        let clip_library = world.resource::<ClipLibrary>();
        let timeline_ctx = build_timeline_context(&timeline_state, &clip_library);
        let outcome = dispatch_tool_call(
            &world,
            &timeline_ctx,
            &ToolCall::GenerateMotion(crate::orchestrator::components::tool_call::MotionCategory::Walk, crate::orchestrator::components::tool_call::SpeedPreset::Normal),
        );
        handle_dispatch_outcome(&world, outcome);

        // Second call
        let timeline_state = world.resource::<TimelineState>();
        let clip_library = world.resource::<ClipLibrary>();
        let timeline_ctx = build_timeline_context(&timeline_state, &clip_library);
        let outcome = dispatch_tool_call(
            &world,
            &timeline_ctx,
            &ToolCall::GenerateMotion(crate::orchestrator::components::tool_call::MotionCategory::Walk, crate::orchestrator::components::tool_call::SpeedPreset::Normal),
        );
        handle_dispatch_outcome(&world, outcome);

        // Assert queue has 2 events (round robin progression)
        {
            let queue = world.resource::<UIEventQueue>();
            assert_eq!(queue.len(), 2, "expected 2 events in queue");
        }

        // Assert motion_seed_counters[Walk] == 2
        {
            let state = world.resource::<OrchestratorState>();
            let counter = state.motion_seed_counters.get(&crate::orchestrator::components::tool_call::MotionCategory::Walk);
            assert_eq!(counter, Some(&2), "expected Walk counter to be 2, got {:?}", counter);
        }
    }

    /// Test that MotionRequest for Jump with no matching clips produces Report feedback.
    #[test]
    fn test_motion_request_jump_no_matching_clips_produces_report() {
        let mut world = make_world();

        // Spawn an entity with ClipSchedule and set as selected_entity
        let entity = world.spawn();
        world.insert_component(entity, crate::ecs::component::ClipSchedule::default());
        {
            let mut hierarchy = world.resource_mut::<crate::ecs::resource::HierarchyState>();
            hierarchy.selected_entity = Some(entity);
        }

        // Dispatch MotionRequest for Jump (no matching clips in library)
        let timeline_state = world.resource::<TimelineState>();
        let clip_library = world.resource::<ClipLibrary>();
        let timeline_ctx = build_timeline_context(&timeline_state, &clip_library);
        let outcome = dispatch_tool_call(
            &world,
            &timeline_ctx,
            &ToolCall::GenerateMotion(crate::orchestrator::components::tool_call::MotionCategory::Jump, crate::orchestrator::components::tool_call::SpeedPreset::Normal),
        );

        handle_dispatch_outcome(&world, outcome);

        // Assert feedback is Report
        {
            let state = world.resource::<OrchestratorState>();
            match &state.feedback {
                Some(crate::ecs::resource::CommandFeedback::Report(_)) => {}
                other => panic!("expected Report feedback, got {:?}", other),
            }
        }

        // Assert queue length did not increase (still 0)
        {
            let queue = world.resource::<UIEventQueue>();
            assert_eq!(queue.len(), 0, "expected no events in queue");
        }
    }

    #[test]
    fn test_reject_log_entry_rejected() {
        let feedback = crate::orchestrator::systems::resolution::OrchestratorFeedback::Rejected {
            best: crate::orchestrator::components::route::Route::GenerateMotion(
                crate::orchestrator::components::tool_call::MotionCategory::Jump,
            ),
            score: 0.35,
        };
        let entry = reject_log_entry("test utterance", 0.42, &feedback).expect("should return Some for Rejected");

        assert_eq!(entry["decision"], "reject");
        assert_eq!(entry["utterance"], "test utterance");
        let raw_score: f32 = entry["raw_top_score"].as_f64().unwrap() as f32;
        assert!((raw_score - 0.42).abs() < 1e-6, "expected raw_top_score ~0.42, got {}", raw_score);
        assert_eq!(entry["details"]["best"], "generate_motion");
        let detail_score: f32 = entry["details"]["score"].as_f64().unwrap() as f32;
        assert!((detail_score - 0.35).abs() < 1e-6, "expected score ~0.35, got {}", detail_score);
    }

    #[test]
    fn test_reject_log_entry_clarify() {
        let candidates: Vec<(crate::orchestrator::components::route::Route, f32)> = vec![
            (crate::orchestrator::components::route::Route::GenerateMotion(
                crate::orchestrator::components::tool_call::MotionCategory::Jump,
            ), 0.4),
            (crate::orchestrator::components::route::Route::PlayAnimation, 0.3),
        ];
        let feedback = crate::orchestrator::systems::resolution::OrchestratorFeedback::ClarifyOptions(candidates);
        let entry = reject_log_entry("test utterance", 0.45, &feedback).expect("should return Some for ClarifyOptions");

        assert_eq!(entry["decision"], "clarify");
        assert_eq!(entry["utterance"], "test utterance");
        let raw_score: f32 = entry["raw_top_score"].as_f64().unwrap() as f32;
        assert!((raw_score - 0.45).abs() < 1e-6, "expected raw_top_score ~0.45, got {}", raw_score);
        let details = entry["details"].as_array().unwrap();
        assert_eq!(details.len(), 2);
        assert_eq!(details[0]["route"], "generate_motion");
        let score0: f32 = details[0]["score"].as_f64().unwrap() as f32;
        assert!((score0 - 0.4).abs() < 1e-6, "expected score ~0.4, got {}", score0);
        assert_eq!(details[1]["route"], "play_animation");
        let score1: f32 = details[1]["score"].as_f64().unwrap() as f32;
        assert!((score1 - 0.3).abs() < 1e-6, "expected score ~0.3, got {}", score1);
    }

    #[test]
    fn test_write_reject_log_appends_json_entry() {
        let log_path = std::path::PathBuf::from("log/orchestrator_rejects.jsonl");
        let lines_before = if log_path.exists() {
            let content = std::fs::read_to_string(&log_path).unwrap_or_default();
            content.lines().count()
        } else {
            0
        };

        // Build a Rejected feedback entry using reject_log_entry
        let feedback = crate::orchestrator::systems::resolution::OrchestratorFeedback::Rejected {
            best: crate::orchestrator::components::route::Route::GenerateMotion(
                crate::orchestrator::components::tool_call::MotionCategory::Jump,
            ),
            score: 0.35,
        };
        let entry = reject_log_entry("test utterance", 0.42, &feedback)
            .expect("should return Some for Rejected");

        // Write the entry via write_reject_log
        write_reject_log(entry).expect("write_reject_log should succeed");

        // Assert line count increased by at least 1
        let content = std::fs::read_to_string(&log_path).expect("log file should exist after write");
        let lines_after = content.lines().count();
        assert!(
            lines_after >= lines_before + 1,
            "expected line count to increase by at least 1: before={}, after={}",
            lines_before,
            lines_after
        );

        // Assert last line is valid JSON containing "decision":"reject"
        let last_line = content.lines().last().expect("log file should have at least one line");
        let parsed: serde_json::Value =
            serde_json::from_str(last_line).expect("last line should be valid JSON");
        assert_eq!(
            parsed.get("decision").and_then(|v| v.as_str()),
            Some("reject"),
            "expected decision to be 'reject', got {:?}",
            parsed.get("decision")
        );
    }
}
