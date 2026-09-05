use crate::ecs::systems::DeferredAction;

#[derive(Default)]
pub struct PlatformEventQueue {
    pub actions: Vec<DeferredAction>,
}
