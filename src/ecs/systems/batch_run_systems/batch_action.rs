use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::{DebugViewMode, DebugViewState};
use crate::ecs::world::World;

pub trait BatchAction: std::fmt::Debug {
    fn name(&self) -> &'static str;
    fn apply(&self, world: &World);
    fn owns_dump(&self) -> bool {
        false
    }
}

pub struct BatchActionDescriptor {
    pub name: &'static str,
    pub parse: fn(&str) -> Option<anyhow::Result<Box<dyn BatchAction>>>,
}

#[derive(Debug)]
pub struct ResetCamera;

#[derive(Debug)]
pub struct ResetCameraUp;

#[derive(Debug)]
pub struct CameraToModel;

impl BatchAction for ResetCamera {
    fn name(&self) -> &'static str {
        "reset_camera"
    }
    fn apply(&self, world: &World) {
        world
            .resource_mut::<UIEventQueue>()
            .send(UIEvent::ResetCamera);
    }
}

impl BatchAction for ResetCameraUp {
    fn name(&self) -> &'static str {
        "reset_camera_up"
    }
    fn apply(&self, world: &World) {
        world
            .resource_mut::<UIEventQueue>()
            .send(UIEvent::ResetCameraUp);
    }
}

impl BatchAction for CameraToModel {
    fn name(&self) -> &'static str {
        "camera_to_model"
    }
    fn apply(&self, world: &World) {
        world
            .resource_mut::<UIEventQueue>()
            .send(UIEvent::MoveCameraToModel);
    }
}

#[derive(Debug)]
pub struct ViewMode(pub DebugViewMode);

#[derive(Debug)]
pub struct BlackBackground;

impl BatchAction for ViewMode {
    fn name(&self) -> &'static str {
        "view_mode"
    }
    fn apply(&self, world: &World) {
        world.resource_mut::<DebugViewState>().debug_view_mode = self.0;
    }
}

impl BatchAction for BlackBackground {
    fn name(&self) -> &'static str {
        "black_background"
    }
    fn apply(&self, world: &World) {
        world.resource_mut::<DebugViewState>().black_background = true;
    }
}

#[derive(Debug)]
pub struct SpawnDebugPrimitive(pub crate::ecs::events::DebugPrimitiveKind);

#[derive(Debug)]
pub struct WallProbeDump;

#[derive(Debug)]
pub struct WaterDebugDump;

impl BatchAction for SpawnDebugPrimitive {
    fn name(&self) -> &'static str {
        match self.0 {
            crate::ecs::events::DebugPrimitiveKind::Cube => "spawn_cube",
            crate::ecs::events::DebugPrimitiveKind::Sphere => "spawn_sphere",
            crate::ecs::events::DebugPrimitiveKind::Floor => "spawn_floor",
        }
    }
    fn apply(&self, world: &World) {
        world
            .resource_mut::<UIEventQueue>()
            .send(UIEvent::SpawnDebugPrimitive { kind: self.0 });
    }
}

impl BatchAction for WallProbeDump {
    fn name(&self) -> &'static str {
        "dump_wall_probe"
    }
    fn apply(&self, _world: &World) {
        // Wall probe dump is now handled synchronously in the render path
        // via batch.dump_wall_probe, so this is a no-op.
    }
    fn owns_dump(&self) -> bool {
        true
    }
}

impl BatchAction for WaterDebugDump {
    fn name(&self) -> &'static str {
        "dump_water_debug"
    }
    fn apply(&self, world: &World) {
        world
            .resource_mut::<UIEventQueue>()
            .send(UIEvent::DumpWaterDebug);
    }
    fn owns_dump(&self) -> bool {
        true
    }
}

fn parse_debug_view_mode(name: &str) -> Option<DebugViewMode> {
    match name {
        "final" => Some(DebugViewMode::Final),
        "position" => Some(DebugViewMode::Position),
        "normal" => Some(DebugViewMode::Normal),
        "shadow_mask" => Some(DebugViewMode::ShadowMask),
        "ndotl" => Some(DebugViewMode::NdotL),
        "light_direction" => Some(DebugViewMode::LightDirection),
        "view_depth" => Some(DebugViewMode::ViewDepth),
        "object_id" => Some(DebugViewMode::ObjectID),
        "selection_view" => Some(DebugViewMode::SelectionView),
        "selection_ubo" => Some(DebugViewMode::SelectionUBO),
        _ => None,
    }
}

fn parse_view_mode(s: &str) -> Option<anyhow::Result<Box<dyn BatchAction>>> {
    let mode_str = s.strip_prefix("view_mode=")?.trim();
    Some(
        parse_debug_view_mode(mode_str)
            .map(|mode| Box::new(ViewMode(mode)) as Box<dyn BatchAction>)
            .ok_or_else(|| anyhow::anyhow!("unknown view_mode '{mode_str}'")),
    )
}

fn parse_spawn_cube(s: &str) -> Option<anyhow::Result<Box<dyn BatchAction>>> {
    (s == "spawn_cube").then(|| {
        Ok(Box::new(SpawnDebugPrimitive(
            crate::ecs::events::DebugPrimitiveKind::Cube,
        )) as Box<dyn BatchAction>)
    })
}

fn parse_spawn_sphere(s: &str) -> Option<anyhow::Result<Box<dyn BatchAction>>> {
    (s == "spawn_sphere").then(|| {
        Ok(Box::new(SpawnDebugPrimitive(
            crate::ecs::events::DebugPrimitiveKind::Sphere,
        )) as Box<dyn BatchAction>)
    })
}

fn parse_spawn_floor(s: &str) -> Option<anyhow::Result<Box<dyn BatchAction>>> {
    (s == "spawn_floor").then(|| {
        Ok(Box::new(SpawnDebugPrimitive(
            crate::ecs::events::DebugPrimitiveKind::Floor,
        )) as Box<dyn BatchAction>)
    })
}

pub fn generic_descriptors() -> Vec<BatchActionDescriptor> {
    vec![
        BatchActionDescriptor {
            name: "reset_camera",
            parse: |s| {
                (s == "reset_camera").then(|| Ok(Box::new(ResetCamera) as Box<dyn BatchAction>))
            },
        },
        BatchActionDescriptor {
            name: "reset_camera_up",
            parse: |s| {
                (s == "reset_camera_up")
                    .then(|| Ok(Box::new(ResetCameraUp) as Box<dyn BatchAction>))
            },
        },
        BatchActionDescriptor {
            name: "camera_to_model",
            parse: |s| {
                (s == "camera_to_model")
                    .then(|| Ok(Box::new(CameraToModel) as Box<dyn BatchAction>))
            },
        },
        BatchActionDescriptor {
            name: "view_mode",
            parse: parse_view_mode,
        },
        BatchActionDescriptor {
            name: "black_background",
            parse: |s| {
                (s == "black_background")
                    .then(|| Ok(Box::new(BlackBackground) as Box<dyn BatchAction>))
            },
        },
        BatchActionDescriptor {
            name: "spawn_cube",
            parse: parse_spawn_cube,
        },
        BatchActionDescriptor {
            name: "spawn_sphere",
            parse: parse_spawn_sphere,
        },
        BatchActionDescriptor {
            name: "spawn_floor",
            parse: parse_spawn_floor,
        },
        BatchActionDescriptor {
            name: "dump_wall_probe",
            parse: |s| {
                (s == "dump_wall_probe")
                    .then(|| Ok(Box::new(WallProbeDump) as Box<dyn BatchAction>))
            },
        },
        BatchActionDescriptor {
            name: "dump_water_debug",
            parse: |s| {
                (s == "dump_water_debug")
                    .then(|| Ok(Box::new(WaterDebugDump) as Box<dyn BatchAction>))
            },
        },
    ]
}

pub fn batch_action_registry() -> Vec<BatchActionDescriptor> {
    let mut registry = generic_descriptors();
    registry.extend(super::flame_args::flame_action_descriptors());
    registry
}
