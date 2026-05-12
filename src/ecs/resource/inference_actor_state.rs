use std::collections::HashMap;

use crate::ml::{InferenceActorId, InferenceRequestId, InferenceResult, InferenceThreadHandle};

pub struct ActorRuntime {
    pub(crate) thread_handle: InferenceThreadHandle,
    pub(crate) enabled: bool,
}

pub struct InferenceActorState {
    pub(crate) actors: HashMap<InferenceActorId, ActorRuntime>,
    pub(crate) pending_results: Vec<InferenceResult>,
    pub(crate) next_request_id: InferenceRequestId,
}

impl Default for InferenceActorState {
    fn default() -> Self {
        Self {
            actors: HashMap::new(),
            pending_results: Vec::new(),
            next_request_id: 1,
        }
    }
}

impl InferenceActorState {
    pub fn actor_max_steps(&self, actor_id: InferenceActorId) -> Option<usize> {
        self.actors.get(&actor_id)?.thread_handle.max_steps()
    }
}
