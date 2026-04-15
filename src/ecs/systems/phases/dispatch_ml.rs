#[cfg(feature = "ml")]
pub fn dispatch_curve_suggestion_events(
    events: &[crate::ecs::events::UIEvent],
    world: &mut crate::ecs::world::World,
) {
    use crate::ecs::events::UIEvent;
    use crate::ecs::resource::{
        BoneNameTokenCache, BoneTopologyCache, ClipLibrary, CurveSuggestionState,
        InferenceActorState, TimelineState,
    };
    use crate::ecs::systems::{
        curve_suggestion_apply, curve_suggestion_dismiss, curve_suggestion_submit,
    };
    use crate::ml::CURVE_COPILOT_ACTOR_ID;

    for event in events {
        match event {
            UIEvent::CurveSuggestionRequest {
                bone_id,
                property_type,
            } => {
                let timeline_state = world.resource::<TimelineState>();
                let clip_id = timeline_state.current_clip_id;
                let current_time = timeline_state.current_time;
                drop(timeline_state);

                let clip_library = world.resource::<ClipLibrary>();
                let clip_info = clip_id
                    .and_then(|id| clip_library.get(id))
                    .and_then(|clip| {
                        clip.tracks
                            .get(bone_id)
                            .map(|track| (track.get_curve(*property_type).clone(), clip.duration))
                    });
                drop(clip_library);

                if let Some((curve, clip_duration)) = clip_info {
                    let topology_cache = world.resource::<BoneTopologyCache>();
                    let name_token_cache = world.resource::<BoneNameTokenCache>();
                    let mut suggestion_state = world.resource_mut::<CurveSuggestionState>();
                    let mut inference_state = world.resource_mut::<InferenceActorState>();
                    curve_suggestion_submit(
                        &mut suggestion_state,
                        &mut inference_state,
                        CURVE_COPILOT_ACTOR_ID,
                        &curve,
                        *property_type,
                        *bone_id,
                        clip_duration,
                        current_time,
                        &topology_cache,
                        &name_token_cache,
                    );
                }
            }

            UIEvent::CurveSuggestionAccept => {
                let suggestion = {
                    let state = world.resource::<CurveSuggestionState>();
                    state.suggestions.first().cloned()
                };

                if let Some(suggestion) = suggestion {
                    let timeline_state = world.resource::<TimelineState>();
                    let clip_id = timeline_state.current_clip_id;
                    drop(timeline_state);

                    if let Some(cid) = clip_id {
                        let mut clip_library = world.resource_mut::<ClipLibrary>();
                        if let Some(clip) = clip_library.get_mut(cid) {
                            if let Some(track) = clip.tracks.get_mut(&suggestion.bone_id) {
                                let curve = track.get_curve_mut(suggestion.property_type);
                                curve_suggestion_apply(&suggestion, curve);
                            }
                        }
                    }

                    let mut state = world.resource_mut::<CurveSuggestionState>();
                    curve_suggestion_dismiss(&mut state);
                }
            }

            UIEvent::CurveSuggestionDismiss => {
                let mut state = world.resource_mut::<CurveSuggestionState>();
                curve_suggestion_dismiss(&mut state);
            }

            _ => {}
        }
    }
}

#[cfg(feature = "text-to-motion")]
pub fn dispatch_text_to_motion_events(
    events: &[crate::ecs::events::UIEvent],
    world: &mut crate::ecs::world::World,
    assets: &mut crate::asset::AssetStorage,
) {
    use crate::ecs::events::UIEvent;
    use crate::ecs::resource::{ClipLibrary, TextToMotionState, TimelineState};
    use crate::ecs::systems::{text_to_motion_cancel, text_to_motion_submit};
    use crate::grpc::GrpcThreadHandle;

    const DEFAULT_ENDPOINT: &str = "http://localhost:50051";

    for event in events {
        match event {
            UIEvent::TextToMotionGenerate {
                prompt,
                duration_seconds,
            } => {
                if !world.contains_resource::<GrpcThreadHandle>() {
                    let handle = GrpcThreadHandle::spawn(DEFAULT_ENDPOINT);
                    world.insert_resource(handle);
                    log!("TextToMotion: spawned gRPC thread ({})", DEFAULT_ENDPOINT);
                }

                let handle = world.get_resource::<GrpcThreadHandle>();
                let mut state = world.resource_mut::<TextToMotionState>();

                if let Some(handle) = handle {
                    text_to_motion_submit(&mut state, &*handle, prompt, *duration_seconds);
                }
            }

            UIEvent::TextToMotionApply => {
                let clip = {
                    let mut state = world.resource_mut::<TextToMotionState>();
                    state.generated_clip.take()
                };

                if let Some(clip) = clip {
                    let mut clip_library = world.resource_mut::<ClipLibrary>();
                    let new_id =
                        crate::ecs::systems::clip_library_systems::clip_library_register_and_activate(
                            &mut clip_library,
                            assets,
                            clip,
                        );
                    drop(clip_library);

                    let mut timeline = world.resource_mut::<TimelineState>();
                    timeline.current_clip_id = Some(new_id);

                    let mut state = world.resource_mut::<TextToMotionState>();
                    text_to_motion_cancel(&mut state);

                    log!("TextToMotion: applied clip (id={})", new_id);
                }
            }

            UIEvent::TextToMotionCancel => {
                let mut state = world.resource_mut::<TextToMotionState>();
                text_to_motion_cancel(&mut state);
                log!("TextToMotion: cancelled");
            }

            _ => {}
        }
    }
}

#[cfg(feature = "text-to-motion")]
pub fn drain_grpc_responses(
    world: &mut crate::ecs::world::World,
    assets: &crate::asset::AssetStorage,
) {
    use crate::grpc::{GrpcResponse, GrpcThreadHandle};

    let handle = match world.get_resource::<GrpcThreadHandle>() {
        Some(h) => h,
        None => return,
    };

    let mut responses = Vec::new();
    while let Some(response) = handle.try_recv() {
        responses.push(response);
    }
    drop(handle);

    for response in responses {
        match response {
            GrpcResponse::MotionGenerated {
                curves,
                generation_time_ms,
                model_used,
            } => {
                apply_motion_response(world, assets, curves, generation_time_ms, model_used);
            }

            #[cfg(feature = "auto-rig")]
            GrpcResponse::MeshGenerated {
                glb_data,
                vertex_count,
                face_count,
                generation_time_ms,
                intermediate_image_png,
            } => {
                apply_mesh_response(
                    world,
                    glb_data,
                    vertex_count,
                    face_count,
                    generation_time_ms,
                    intermediate_image_png,
                );
            }

            GrpcResponse::ServerStatus {
                ready,
                active_model,
                ..
            } => {
                use crate::ecs::resource::TextToMotionState;
                if let Some(mut state) = world.get_resource_mut::<TextToMotionState>() {
                    state.server_ready = ready;
                    state.model_used = Some(active_model);
                }
            }

            #[cfg(feature = "auto-rig")]
            GrpcResponse::MeshServerStatus { ready } => {
                handle_mesh_server_status(world, ready);
            }

            #[cfg(feature = "auto-rig")]
            GrpcResponse::RigGenerated {
                rigged_glb_data,
                joint_count,
                bone_count,
                generation_time_ms,
                ..
            } => {
                apply_rig_response(
                    world,
                    rigged_glb_data,
                    joint_count,
                    bone_count,
                    generation_time_ms,
                );
            }

            #[cfg(feature = "auto-rig")]
            GrpcResponse::RiggingServerStatus { ready, .. } => {
                handle_rigging_server_status(world, ready);
            }

            GrpcResponse::Error { message } => {
                route_grpc_error(world, &message);
            }
        }
    }
}

#[cfg(feature = "text-to-motion")]
fn apply_motion_response(
    world: &mut crate::ecs::world::World,
    assets: &crate::asset::AssetStorage,
    curves: Vec<crate::grpc::RawAnimationCurve>,
    generation_time_ms: f32,
    model_used: String,
) {
    use crate::ecs::resource::{TextToMotionState, TextToMotionStatus};
    use crate::grpc::convert_motion_response_to_clip;

    let mut state = world.resource_mut::<TextToMotionState>();
    if state.status != TextToMotionStatus::Generating {
        return;
    }

    let bone_name_to_id = assets
        .skeletons
        .values()
        .next()
        .map(|sa| sa.skeleton.bone_name_to_id.clone())
        .unwrap_or_default();

    let clip_name = format!("T2M: {}", truncate_prompt(&state.last_prompt, 30));
    let clip =
        convert_motion_response_to_clip(&curves, &clip_name, state.last_duration, &bone_name_to_id);

    log!(
        "TextToMotion: generated clip '{}' with {} tracks in {:.0}ms (model: {})",
        clip_name,
        clip.tracks.len(),
        generation_time_ms,
        model_used
    );

    state.status = TextToMotionStatus::Generated;
    state.generated_clip = Some(clip);
    state.generation_time_ms = Some(generation_time_ms);
    state.model_used = Some(model_used);
}

#[cfg(feature = "auto-rig")]
fn apply_mesh_response(
    world: &mut crate::ecs::world::World,
    glb_data: Vec<u8>,
    vertex_count: u32,
    face_count: u32,
    generation_time_ms: f32,
    intermediate_image_png: Option<Vec<u8>>,
) {
    use crate::ecs::resource::{TextToMeshState, TextToMeshStatus};

    let mut state = world.resource_mut::<TextToMeshState>();
    if state.status != TextToMeshStatus::Generating {
        return;
    }

    log!(
        "TextToMesh: received GLB ({} bytes, {} vertices, {} faces) in {:.0}ms",
        glb_data.len(),
        vertex_count,
        face_count,
        generation_time_ms
    );

    if let Some(ref png_data) = intermediate_image_png {
        let path = format!(
            "log/text_to_mesh_intermediate_{}.png",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
        if let Err(e) = std::fs::write(&path, png_data) {
            log_warn!("Failed to save intermediate image: {}", e);
        } else {
            log!("TextToMesh: saved intermediate image to {}", path);
        }
    }

    state.status = TextToMeshStatus::Generated;
    state.glb_data = Some(glb_data);
    state.vertex_count = Some(vertex_count);
    state.face_count = Some(face_count);
    state.generation_time_ms = Some(generation_time_ms);
    state.intermediate_image_png = intermediate_image_png;
}

#[cfg(feature = "auto-rig")]
fn handle_mesh_server_status(world: &mut crate::ecs::world::World, ready: bool) {
    use crate::ecs::resource::{TextToMeshState, TextToMeshStatus};
    use crate::grpc::GrpcThreadHandle;

    let mut state = world.resource_mut::<TextToMeshState>();
    if state.status != TextToMeshStatus::WaitingForServer {
        return;
    }

    if ready {
        log!("TextToMesh: server ready, submitting pending request");
        drop(state);

        let handle = world.get_resource::<GrpcThreadHandle>();
        let mut state = world.resource_mut::<TextToMeshState>();
        if let Some(handle) = handle {
            crate::ecs::systems::text_to_mesh_send_generate(&mut state, &*handle);
        }
    } else {
        state.last_status_check = Some(std::time::Instant::now());
    }
}

#[cfg(feature = "auto-rig")]
pub fn poll_mesh_server_status(world: &mut crate::ecs::world::World) {
    use crate::ecs::resource::{TextToMeshState, TextToMeshStatus};
    use crate::grpc::{GrpcRequest, GrpcThreadHandle};

    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

    let state = world.resource::<TextToMeshState>();
    if state.status != TextToMeshStatus::WaitingForServer {
        return;
    }

    let should_poll = match state.last_status_check {
        Some(last) => last.elapsed() >= POLL_INTERVAL,
        None => false,
    };
    drop(state);

    if !should_poll {
        return;
    }

    if let Some(handle) = world.get_resource::<GrpcThreadHandle>() {
        handle.send(GrpcRequest::CheckMeshStatus);
    }
}

#[cfg(feature = "text-to-motion")]
fn route_grpc_error(world: &mut crate::ecs::world::World, message: &str) {
    use crate::ecs::resource::{TextToMotionState, TextToMotionStatus};

    if let Some(mut state) = world.get_resource_mut::<TextToMotionState>() {
        if state.status == TextToMotionStatus::Generating {
            log_error!("TextToMotion: error - {}", message);
            state.status = TextToMotionStatus::Error;
            state.error_message = Some(message.to_string());
            return;
        }
    }

    #[cfg(feature = "auto-rig")]
    {
        use crate::ecs::resource::{TextToMeshState, TextToMeshStatus};
        if let Some(mut state) = world.get_resource_mut::<TextToMeshState>() {
            if state.status == TextToMeshStatus::Generating
                || state.status == TextToMeshStatus::WaitingForServer
            {
                log_error!("TextToMesh: error - {}", message);
                state.status = TextToMeshStatus::Error;
                state.error_message = Some(message.to_string());
                state.pending_request = None;
                return;
            }
        }
    }

    #[cfg(feature = "auto-rig")]
    {
        use crate::ecs::resource::{AutoRigState, AutoRigStatus};
        if let Some(mut state) = world.get_resource_mut::<AutoRigState>() {
            if state.status == AutoRigStatus::Rigging
                || state.status == AutoRigStatus::WaitingForServer
            {
                log_error!("AutoRig: error - {}", message);
                state.status = AutoRigStatus::Error;
                state.error_message = Some(message.to_string());
                state.source_glb_data = None;
                return;
            }
        }
    }

    log_warn!("gRPC error with no active request: {}", message);
}

#[cfg(feature = "text-to-motion")]
fn truncate_prompt(prompt: &str, max_len: usize) -> String {
    if prompt.len() <= max_len {
        prompt.to_string()
    } else {
        format!("{}...", &prompt[..max_len])
    }
}

#[cfg(feature = "auto-rig")]
pub fn dispatch_text_to_mesh_events(
    events: &[crate::ecs::events::UIEvent],
    world: &mut crate::ecs::world::World,
    deferred: &mut Vec<super::super::ui_event_systems::DeferredAction>,
) {
    use crate::ecs::events::UIEvent;
    use crate::ecs::resource::{TextToMeshState, TextToMeshStatus};
    use crate::ecs::systems::{text_to_mesh_cancel, text_to_mesh_submit};
    use crate::grpc::GrpcThreadHandle;

    const DEFAULT_ENDPOINT: &str = "http://localhost:50051";

    for event in events {
        match event {
            UIEvent::TextToMeshGenerate {
                prompt,
                target_faces,
                seed,
                input_mode,
                input_image_png,
                model_type,
                t2i_model_type,
            } => {
                if !world.contains_resource::<GrpcThreadHandle>() {
                    ensure_mesh_server_running(world);
                    let handle = GrpcThreadHandle::spawn(DEFAULT_ENDPOINT);
                    world.insert_resource(handle);
                    log!("TextToMesh: spawned gRPC thread ({})", DEFAULT_ENDPOINT);
                }

                let handle = world.get_resource::<GrpcThreadHandle>();
                let mut state = world.resource_mut::<TextToMeshState>();

                if let Some(handle) = handle {
                    text_to_mesh_submit(
                        &mut state,
                        &*handle,
                        prompt.clone(),
                        *target_faces,
                        *seed,
                        input_mode.clone(),
                        input_image_png.clone(),
                        model_type.clone(),
                        t2i_model_type.clone(),
                    );
                }
            }

            UIEvent::TextToMeshApply => {
                let mut state = world.resource_mut::<TextToMeshState>();
                if let Some(glb_data) = state.glb_data.take() {
                    state.status = TextToMeshStatus::Idle;
                    deferred.push(
                        super::super::ui_event_systems::DeferredAction::LoadModelFromMemory {
                            glb_data,
                        },
                    );
                    log!("TextToMesh: applying generated mesh to scene");
                }
            }

            UIEvent::TextToMeshCancel => {
                let mut state = world.resource_mut::<TextToMeshState>();
                text_to_mesh_cancel(&mut state);
                log!("TextToMesh: cancelled");
            }

            _ => {}
        }
    }
}

#[cfg(feature = "auto-rig")]
fn apply_rig_response(
    world: &mut crate::ecs::world::World,
    rigged_glb_data: Vec<u8>,
    joint_count: u32,
    bone_count: u32,
    generation_time_ms: f32,
) {
    use crate::ecs::resource::{AutoRigState, AutoRigStatus};

    let mut state = world.resource_mut::<AutoRigState>();
    if state.status != AutoRigStatus::Rigging {
        return;
    }

    log!(
        "AutoRig: received rigged GLB ({} bytes, {} joints, {} bones) in {:.0}ms",
        rigged_glb_data.len(),
        joint_count,
        bone_count,
        generation_time_ms
    );

    state.joint_count = Some(joint_count);
    state.bone_count = Some(bone_count);
    state.generation_time_ms = Some(generation_time_ms);
    state.rigged_glb_data = Some(rigged_glb_data);
    state.status = AutoRigStatus::Previewing;
}

#[cfg(feature = "auto-rig")]
fn handle_rigging_server_status(world: &mut crate::ecs::world::World, ready: bool) {
    use crate::ecs::resource::{AutoRigState, AutoRigStatus};
    use crate::grpc::GrpcThreadHandle;

    let mut state = world.resource_mut::<AutoRigState>();
    if state.status != AutoRigStatus::WaitingForServer {
        return;
    }

    if ready {
        log!("AutoRig: server ready, submitting rig request");
        drop(state);

        let handle = world.get_resource::<GrpcThreadHandle>();
        let mut state = world.resource_mut::<AutoRigState>();
        if let Some(handle) = handle {
            crate::ecs::systems::auto_rig_send_generate(&mut state, &*handle);
        }
    } else {
        state.last_status_check = Some(std::time::Instant::now());
    }
}

#[cfg(feature = "auto-rig")]
pub fn poll_rigging_server_status(world: &mut crate::ecs::world::World) {
    use crate::ecs::resource::{AutoRigState, AutoRigStatus};
    use crate::grpc::{GrpcRequest, GrpcThreadHandle};

    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

    let state = world.resource::<AutoRigState>();
    if state.status != AutoRigStatus::WaitingForServer {
        return;
    }

    let should_poll = match state.last_status_check {
        Some(last) => last.elapsed() >= POLL_INTERVAL,
        None => false,
    };
    drop(state);

    if !should_poll {
        return;
    }

    if let Some(handle) = world.get_resource::<GrpcThreadHandle>() {
        handle.send(GrpcRequest::CheckRiggingStatus);
    }
}

#[cfg(feature = "auto-rig")]
pub fn dispatch_auto_rig_events(
    events: &[crate::ecs::events::UIEvent],
    world: &mut crate::ecs::world::World,
    deferred: &mut Vec<super::super::ui_event_systems::DeferredAction>,
) {
    use crate::ecs::component::GlbSource;
    use crate::ecs::events::UIEvent;
    use crate::ecs::resource::{AutoRigState, AutoRigStatus, HierarchyState};
    use crate::ecs::systems::{auto_rig_cancel, auto_rig_submit};
    use crate::ecs::world::Parent;
    use crate::grpc::GrpcThreadHandle;

    const DEFAULT_ENDPOINT: &str = "http://localhost:50051";

    for event in events {
        match event {
            UIEvent::AutoRigGenerate { .. } => {
                let hierarchy = world.resource::<HierarchyState>();
                let selected = hierarchy.selected_entity;
                drop(hierarchy);

                let selected_entity = match selected {
                    Some(e) => e,
                    None => continue,
                };

                let parent_entity = resolve_parent_with_glb_source(world, selected_entity);
                let parent_entity = match parent_entity {
                    Some(e) => e,
                    None => {
                        log_warn!("AutoRig: selected entity has no GlbSource");
                        continue;
                    }
                };

                let glb_source = world.get_component::<GlbSource>(parent_entity);
                let glb_data = match glb_source {
                    Some(source) => match source.read_bytes() {
                        Ok(data) => data,
                        Err(e) => {
                            log_error!("AutoRig: failed to read GLB: {}", e);
                            continue;
                        }
                    },
                    None => continue,
                };

                if !world.contains_resource::<GrpcThreadHandle>() {
                    ensure_mesh_server_running(world);
                    let handle = GrpcThreadHandle::spawn(DEFAULT_ENDPOINT);
                    world.insert_resource(handle);
                    log!("AutoRig: spawned gRPC thread ({})", DEFAULT_ENDPOINT);
                }

                let handle = world.get_resource::<GrpcThreadHandle>();
                let mut state = world.resource_mut::<AutoRigState>();

                if let Some(handle) = handle {
                    state.original_glb_backup = Some(glb_data.clone());
                    auto_rig_submit(&mut state, &*handle, glb_data, parent_entity);
                }
            }

            UIEvent::AutoRigApply => {
                let mut state = world.resource_mut::<AutoRigState>();
                if state.status != AutoRigStatus::Previewing {
                    continue;
                }

                if let Some(rigged_glb) = state.rigged_glb_data.take() {
                    state.status = AutoRigStatus::Idle;
                    state.original_glb_backup = None;
                    state.target_entity = None;
                    deferred.push(
                        super::super::ui_event_systems::DeferredAction::LoadModelFromMemory {
                            glb_data: rigged_glb,
                        },
                    );
                    log!("AutoRig: applying rigged model to scene");
                }
            }

            UIEvent::AutoRigDiscard => {
                let mut state = world.resource_mut::<AutoRigState>();
                if state.status == AutoRigStatus::Previewing {
                    if let Some(original_glb) = state.original_glb_backup.take() {
                        auto_rig_cancel(&mut state);
                        deferred.push(
                            super::super::ui_event_systems::DeferredAction::LoadModelFromMemory {
                                glb_data: original_glb,
                            },
                        );
                        log!("AutoRig: discarding, reverting to original model");
                        continue;
                    }
                }
                auto_rig_cancel(&mut state);
                log!("AutoRig: cancelled");
            }

            _ => {}
        }
    }
}

#[cfg(feature = "auto-rig")]
fn resolve_parent_with_glb_source(
    world: &crate::ecs::world::World,
    entity: crate::ecs::world::Entity,
) -> Option<crate::ecs::world::Entity> {
    use crate::ecs::component::GlbSource;
    use crate::ecs::world::Parent;

    if world.get_component::<GlbSource>(entity).is_some() {
        return Some(entity);
    }

    if let Some(Parent(parent)) = world.get_component::<Parent>(entity) {
        if world.get_component::<GlbSource>(*parent).is_some() {
            return Some(*parent);
        }
    }

    None
}

#[cfg(feature = "auto-rig")]
fn ensure_mesh_server_running(world: &mut crate::ecs::world::World) {
    use crate::grpc::MeshServerProcess;

    if let Some(mut proc) = world.get_resource_mut::<MeshServerProcess>() {
        if proc.is_running() {
            return;
        }
        log_warn!("MeshServer: process exited, restarting");
    }

    let is_debug = cfg!(debug_assertions);
    match MeshServerProcess::launch() {
        Ok(proc) => {
            world.insert_resource(proc);
            log!("MeshServer: launched (debug={})", is_debug);
        }
        Err(e) => {
            log_error!("MeshServer: failed to launch: {}", e);
        }
    }
}
