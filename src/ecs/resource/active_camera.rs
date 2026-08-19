use crate::ecs::world::Entity;

#[derive(Clone, Debug, Default)]
pub struct ActiveCamera(pub Option<Entity>);
