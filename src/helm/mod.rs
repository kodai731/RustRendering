//! Text-command helm, pure domain layer.
//!
//! Holds the tool schema, the route table the embedding router indexes, and the
//! pure resolution functions those need. Nothing here depends on `World`,
//! `Entity` or graphics resources, so it is testable without ECS infrastructure.
//!
//! The ECS-facing half (dispatcher, name resolution) lives in
//! `src/ecs/systems/helm/`.

pub mod components;
pub mod systems;
