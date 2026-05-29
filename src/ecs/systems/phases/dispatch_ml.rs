#[cfg(feature = "ml")]
pub fn dispatch_curve_suggestion_events(
    events: &[crate::ecs::events::UIEvent],
    world: &mut crate::ecs::world::World,
    _assets: &crate::asset::AssetStorage,
) {
    use crate::ecs::events::UIEvent;
    use crate::ecs::resource::{
        ClipLibrary, CurveSuggestionState, InferenceActorState, TimelineState,
    };
    use crate::ecs::systems::{
        curve_suggestion_apply, curve_suggestion_dismiss, curve_suggestion_submit,
        CurveSuggestionInputs,
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

                let Some(clip_id) = clip_id else {
                    continue;
                };

                let clip = {
                    let clip_library = world.resource::<ClipLibrary>();
                    clip_library.get(clip_id).cloned()
                };
                let Some(clip) = clip else {
                    continue;
                };

                let mut suggestion_state = world.resource_mut::<CurveSuggestionState>();
                let mut inference_state = world.resource_mut::<InferenceActorState>();
                curve_suggestion_submit(
                    &mut suggestion_state,
                    &mut inference_state,
                    CURVE_COPILOT_ACTOR_ID,
                    CurveSuggestionInputs { clip: &clip },
                    *property_type,
                    *bone_id,
                    current_time,
                );
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

#[cfg(feature = "auto-rig")]
pub fn dispatch_text_to_animation_events(
    events: &[crate::ecs::events::UIEvent],
    world: &mut crate::ecs::world::World,
    assets: &crate::asset::AssetStorage,
) {
    use crate::ecs::events::UIEvent;
    use crate::ecs::resource::{AutoRigState, ModelState, TextToAnimationState};
    use crate::ecs::systems::{
        auto_rig_submit, detect_rig_presence, text_to_animation_begin, text_to_animation_cancel,
        RigPresence,
    };
    use crate::grpc::{GrpcRequest, GrpcThreadHandle, TextToMotionRequest};

    const DEFAULT_ENDPOINT: &str = "http://localhost:50051";

    for event in events {
        match event {
            UIEvent::TextToAnimationGenerate {
                prompt,
                duration_seconds,
            } => {
                let entity = match resolve_entity_with_glb_source(world) {
                    Some(e) => e,
                    None => {
                        let mut state = world.resource_mut::<TextToAnimationState>();
                        crate::ecs::systems::text_to_animation_mark_error(
                            &mut state,
                            "No model with GlbSource selected".to_string(),
                        );
                        log_warn!("TextToAnimation: no selected entity with GlbSource");
                        continue;
                    }
                };

                let glb_data = match read_glb_bytes(world, entity) {
                    Some(data) => data,
                    None => {
                        let mut state = world.resource_mut::<TextToAnimationState>();
                        crate::ecs::systems::text_to_animation_mark_error(
                            &mut state,
                            "Failed to read GLB bytes".to_string(),
                        );
                        continue;
                    }
                };

                let rig_present = {
                    let model_state = world.resource::<ModelState>();
                    detect_rig_presence(&model_state, assets)
                };

                if !world.contains_resource::<GrpcThreadHandle>() {
                    ensure_mesh_server_running(world);
                    let handle = GrpcThreadHandle::spawn(DEFAULT_ENDPOINT);
                    world.insert_resource(handle);
                    log!(
                        "TextToAnimation: spawned gRPC thread ({})",
                        DEFAULT_ENDPOINT
                    );
                }

                {
                    let mut state = world.resource_mut::<TextToAnimationState>();
                    text_to_animation_begin(
                        &mut state,
                        prompt.clone(),
                        *duration_seconds,
                        entity,
                        rig_present,
                    );
                }

                let handle = match world.get_resource::<GrpcThreadHandle>() {
                    Some(h) => h,
                    None => continue,
                };

                match rig_present {
                    RigPresence::Present => {
                        log!(
                            "TextToAnimation: skipping auto-rig, sending GenerateMotion (prompt='{}', duration={}s)",
                            prompt,
                            duration_seconds
                        );
                        handle.send(GrpcRequest::GenerateMotion(TextToMotionRequest {
                            prompt: prompt.clone(),
                            duration_seconds: *duration_seconds,
                            target_fps: 30,
                            glb_data,
                        }));
                    }
                    RigPresence::Absent => {
                        log!(
                            "TextToAnimation: no rig detected, starting auto-rig first (prompt='{}')",
                            prompt
                        );
                        drop(handle);
                        let handle = world.get_resource::<GrpcThreadHandle>().unwrap();
                        let mut auto_rig_state = world.resource_mut::<AutoRigState>();
                        auto_rig_state.original_glb_backup = Some(glb_data.clone());
                        auto_rig_submit(&mut auto_rig_state, &*handle, glb_data, entity);
                    }
                }
            }

            UIEvent::TextToAnimationCancel => {
                let mut state = world.resource_mut::<TextToAnimationState>();
                text_to_animation_cancel(&mut state);
                log!("TextToAnimation: cancelled");
            }

            _ => {}
        }
    }
}

#[cfg(feature = "auto-rig")]
fn resolve_entity_with_glb_source(
    world: &crate::ecs::world::World,
) -> Option<crate::ecs::world::Entity> {
    use crate::ecs::resource::HierarchyState;

    let selected = {
        let hierarchy = world.resource::<HierarchyState>();
        hierarchy.selected_entity?
    };

    resolve_parent_with_glb_source(world, selected)
}

#[cfg(feature = "auto-rig")]
fn read_glb_bytes(
    world: &crate::ecs::world::World,
    entity: crate::ecs::world::Entity,
) -> Option<Vec<u8>> {
    use crate::ecs::component::GlbSource;

    let source = world.get_component::<GlbSource>(entity)?;
    match source.read_bytes() {
        Ok(data) => Some(data),
        Err(e) => {
            log_error!("Failed to read GLB bytes from entity: {}", e);
            None
        }
    }
}

#[cfg(feature = "auto-rig")]
fn find_first_entity_with_glb_source(
    world: &crate::ecs::world::World,
) -> Option<crate::ecs::world::Entity> {
    use crate::ecs::component::GlbSource;

    world
        .iter_components::<GlbSource>()
        .next()
        .map(|(entity, _)| entity)
}

#[cfg(feature = "auto-rig")]
pub fn dispatch_model_loaded_for_animation(
    events: &[crate::ecs::events::UIEvent],
    world: &mut crate::ecs::world::World,
) {
    use crate::ecs::events::{ModelLoadSource, UIEvent};
    use crate::ecs::resource::{TextToAnimationState, TextToAnimationStatus};
    use crate::ecs::systems::text_to_animation_advance_to_motion;
    use crate::grpc::{GrpcRequest, GrpcThreadHandle, TextToMotionRequest};

    for event in events {
        if let UIEvent::ModelLoadedFromMemory { source } = event {
            log!(
                "dispatch_model_loaded_for_animation: received ModelLoadedFromMemory({:?})",
                source
            );
            if *source != ModelLoadSource::AutoRigOutput {
                continue;
            }
            let current_status = {
                let state = world.resource::<TextToAnimationState>();
                state.status.clone()
            };
            log!(
                "dispatch_model_loaded_for_animation: TextToAnimationStatus={:?}",
                current_status
            );
            if current_status != TextToAnimationStatus::AutoRigApplying {
                continue;
            }

            let entity = match find_first_entity_with_glb_source(world) {
                Some(e) => e,
                None => {
                    let mut state = world.resource_mut::<TextToAnimationState>();
                    crate::ecs::systems::text_to_animation_mark_error(
                        &mut state,
                        "Rigged model has no GlbSource after load".to_string(),
                    );
                    continue;
                }
            };

            let glb_data = match read_glb_bytes(world, entity) {
                Some(data) => data,
                None => {
                    let mut state = world.resource_mut::<TextToAnimationState>();
                    crate::ecs::systems::text_to_animation_mark_error(
                        &mut state,
                        "Failed to read rigged GLB after load".to_string(),
                    );
                    continue;
                }
            };

            let (prompt, duration) = {
                let mut state = world.resource_mut::<TextToAnimationState>();
                text_to_animation_advance_to_motion(&mut state);
                state.target_entity = Some(entity);
                (state.prompt.clone(), state.duration_seconds)
            };

            log!(
                "TextToAnimation: rigged model loaded, sending GenerateMotion (prompt='{}')",
                prompt
            );

            if let Some(handle) = world.get_resource::<GrpcThreadHandle>() {
                handle.send(GrpcRequest::GenerateMotion(TextToMotionRequest {
                    prompt,
                    duration_seconds: duration,
                    target_fps: 30,
                    glb_data,
                }));
            }
        }
    }
}

#[cfg(feature = "text-to-motion")]
pub fn drain_grpc_responses(
    world: &mut crate::ecs::world::World,
    assets: &mut crate::asset::AssetStorage,
    deferred: &mut Vec<super::super::ui_event_systems::DeferredAction>,
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
            #[cfg(feature = "auto-rig")]
            GrpcResponse::MotionGenerated {
                curves,
                generation_time_ms,
                model_used,
            } => {
                apply_motion_response(world, assets, curves, generation_time_ms, model_used);
            }

            #[cfg(not(feature = "auto-rig"))]
            GrpcResponse::MotionGenerated { .. } => {
                log_warn!("MotionGenerated received but auto-rig feature is disabled");
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

            GrpcResponse::ServerStatus { ready, .. } => {
                #[cfg(feature = "auto-rig")]
                {
                    use crate::ecs::resource::TextToAnimationState;
                    if let Some(mut state) = world.get_resource_mut::<TextToAnimationState>() {
                        state.server_ready = ready;
                    }
                }
                #[cfg(not(feature = "auto-rig"))]
                {
                    let _ = ready;
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
                    deferred,
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

#[cfg(feature = "auto-rig")]
fn apply_motion_response(
    world: &mut crate::ecs::world::World,
    assets: &mut crate::asset::AssetStorage,
    curves: Vec<crate::grpc::RawAnimationCurve>,
    generation_time_ms: f32,
    model_used: String,
) {
    use crate::ecs::resource::{
        ClipLibrary, TextToAnimationState, TextToAnimationStatus, TimelineState,
    };
    use crate::ecs::systems::{text_to_animation_mark_done, text_to_animation_set_motion_result};
    use crate::grpc::convert_motion_response_to_clip;

    let (prompt, duration) = {
        let state = world.resource::<TextToAnimationState>();
        if state.status != TextToAnimationStatus::GeneratingMotion {
            return;
        }
        (state.prompt.clone(), state.duration_seconds)
    };

    let bone_name_to_id = assets
        .skeletons
        .values()
        .next()
        .map(|sa| sa.skeleton.bone_name_to_id.clone())
        .unwrap_or_default();

    let clip_name = format!("t2a_{}", prompt_to_snake_case(&prompt, 30));
    let clip = convert_motion_response_to_clip(&curves, &clip_name, duration, &bone_name_to_id);
    let track_count = clip.tracks.len();

    log!(
        "TextToAnimation: generated clip '{}' with {} tracks in {:.0}ms (model: {})",
        clip_name,
        track_count,
        generation_time_ms,
        model_used
    );

    {
        let mut state = world.resource_mut::<TextToAnimationState>();
        text_to_animation_set_motion_result(&mut state, clip, generation_time_ms, model_used);
    }

    let clip = {
        let mut state = world.resource_mut::<TextToAnimationState>();
        state.generated_clip.take()
    };

    if let Some(clip) = clip {
        let mut clip_library = world.resource_mut::<ClipLibrary>();
        let new_id = crate::ecs::systems::clip_library_systems::clip_library_register_and_activate(
            &mut clip_library,
            assets,
            clip,
        );
        drop(clip_library);

        let mut timeline = world.resource_mut::<TimelineState>();
        timeline.current_clip_id = Some(new_id);
        drop(timeline);

        let mut state = world.resource_mut::<TextToAnimationState>();
        text_to_animation_mark_done(&mut state);

        log!("TextToAnimation: applied clip (id={})", new_id);
    }
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
    #[cfg(feature = "auto-rig")]
    {
        use crate::ecs::resource::{TextToAnimationState, TextToAnimationStatus};
        use crate::ecs::systems::text_to_animation_mark_error;

        if let Some(mut state) = world.get_resource_mut::<TextToAnimationState>() {
            let active = matches!(
                state.status,
                TextToAnimationStatus::AutoRigging
                    | TextToAnimationStatus::AutoRigApplying
                    | TextToAnimationStatus::GeneratingMotion
                    | TextToAnimationStatus::ApplyingClip
            );
            if active {
                log_error!("TextToAnimation: error - {}", message);
                text_to_animation_mark_error(&mut state, message.to_string());
                return;
            }
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

#[cfg(feature = "auto-rig")]
fn prompt_to_snake_case(prompt: &str, max_len: usize) -> String {
    let sanitized: String = prompt
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();

    let collapsed = collapse_repeated_underscores(&sanitized);
    let trimmed = collapsed.trim_matches('_');
    let truncated: String = trimmed.chars().take(max_len).collect();
    let result = truncated.trim_end_matches('_').to_string();

    if result.is_empty() {
        "untitled".to_string()
    } else {
        result
    }
}

#[cfg(feature = "auto-rig")]
fn collapse_repeated_underscores(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_underscore = false;
    for c in input.chars() {
        if c == '_' {
            if !last_underscore {
                out.push('_');
                last_underscore = true;
            }
        } else {
            out.push(c);
            last_underscore = false;
        }
    }
    out
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
                            source: crate::ecs::events::ModelLoadSource::TextToMeshOutput,
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
    deferred: &mut Vec<super::super::ui_event_systems::DeferredAction>,
) {
    use crate::ecs::events::ModelLoadSource;
    use crate::ecs::resource::{
        AutoRigState, AutoRigStatus, TextToAnimationState, TextToAnimationStatus,
    };
    use crate::ecs::systems::text_to_animation_advance_to_apply_rig;

    let mut auto_rig = world.resource_mut::<AutoRigState>();
    if auto_rig.status != AutoRigStatus::Rigging {
        return;
    }

    log!(
        "AutoRig: received rigged GLB ({} bytes, {} joints, {} bones) in {:.0}ms",
        rigged_glb_data.len(),
        joint_count,
        bone_count,
        generation_time_ms
    );

    auto_rig.joint_count = Some(joint_count);
    auto_rig.bone_count = Some(bone_count);
    auto_rig.generation_time_ms = Some(generation_time_ms);

    let orchestrated =
        world.resource::<TextToAnimationState>().status == TextToAnimationStatus::AutoRigging;

    if orchestrated {
        auto_rig.status = AutoRigStatus::Idle;
        auto_rig.rigged_glb_data = None;
        auto_rig.original_glb_backup = None;
        auto_rig.target_entity = None;
        drop(auto_rig);

        let mut state = world.resource_mut::<TextToAnimationState>();
        text_to_animation_advance_to_apply_rig(&mut state);
        drop(state);

        deferred.push(
            super::super::ui_event_systems::DeferredAction::LoadModelFromMemory {
                glb_data: rigged_glb_data,
                source: ModelLoadSource::AutoRigOutput,
            },
        );
        log!("TextToAnimation: rigged GLB queued for load (orchestrated)");
    } else {
        auto_rig.rigged_glb_data = Some(rigged_glb_data);
        auto_rig.status = AutoRigStatus::Previewing;
    }
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
                            source: crate::ecs::events::ModelLoadSource::AutoRigOutput,
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
                                source: crate::ecs::events::ModelLoadSource::UserFile,
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
