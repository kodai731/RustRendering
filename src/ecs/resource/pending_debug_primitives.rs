use cgmath::Vector3;

use crate::ecs::events::DebugPrimitiveKind;

#[derive(Debug, Clone, Copy)]
pub struct PendingDebugPrimitive {
    pub kind: DebugPrimitiveKind,
    pub position: Vector3<f32>,
}

/// Debug primitives restored from a scene file, spawned once the model load has settled.
#[derive(Debug, Clone, Default)]
pub struct PendingDebugPrimitives {
    pub requests: Vec<PendingDebugPrimitive>,
}

impl PendingDebugPrimitives {
    pub fn take_requests(&mut self) -> Vec<PendingDebugPrimitive> {
        std::mem::take(&mut self.requests)
    }
}
