use crate::ecs::component::InferenceActorSetup;
use crate::ecs::resource::{ActorRuntime, InferenceActorState};
use crate::ml::{
    InferenceActorId, InferenceRequest, InferenceRequestId, InferenceRequestKind, InferenceResult,
    InferenceThreadHandle,
};

pub fn inference_actor_initialize(setup: &InferenceActorSetup, state: &mut InferenceActorState) {
    if !setup.enabled {
        return;
    }

    if state.actors.contains_key(&setup.actor_id) {
        return;
    }

    let handle = match InferenceThreadHandle::spawn(
        &setup.model_path,
        setup.actor_id,
        setup.model_kind.clone(),
    ) {
        Ok(h) => h,
        Err(e) => {
            log!(
                "Failed to spawn inference actor {}: {:?}",
                setup.actor_id,
                e
            );
            return;
        }
    };

    state.actors.insert(
        setup.actor_id,
        ActorRuntime {
            thread_handle: handle,
            enabled: true,
        },
    );

    log!("Initialized inference actor {}", setup.actor_id);
}

pub fn inference_actor_poll(state: &mut InferenceActorState) {
    let actor_ids: Vec<InferenceActorId> = state.actors.keys().copied().collect();

    for actor_id in actor_ids {
        if let Some(runtime) = state.actors.get(&actor_id) {
            while let Some(result) = runtime.thread_handle.try_recv() {
                state.pending_results.push(result);
            }
        }
    }
}

pub fn inference_actor_submit(
    state: &mut InferenceActorState,
    actor_id: InferenceActorId,
    kind: InferenceRequestKind,
) -> Option<InferenceRequestId> {
    let request_id = state.next_request_id;
    state.next_request_id += 1;

    let runtime = state.actors.get(&actor_id)?;

    if !runtime.enabled {
        return None;
    }

    let request = InferenceRequest {
        request_id,
        actor_id,
        kind,
    };

    match runtime.thread_handle.send(request) {
        Ok(()) => Some(request_id),
        Err(e) => {
            log!(
                "Failed to send inference request to actor {}: {:?}",
                actor_id,
                e
            );
            None
        }
    }
}

pub fn inference_actor_take_results(state: &mut InferenceActorState) -> Vec<InferenceResult> {
    std::mem::take(&mut state.pending_results)
}
