use crate::app::App;
use crate::ecs::events::DebugPrimitiveKind;
use crate::vulkanr::context::{CommandState, SwapchainState};
use crate::vulkanr::vulkan::*;
use anyhow::Result;

impl App {
    pub unsafe fn spawn_debug_primitive(&mut self, kind: DebugPrimitiveKind) -> Result<()> {
        let position = default_debug_primitive_position(kind);
        self.spawn_debug_primitive_at(kind, position)
    }

    pub unsafe fn spawn_debug_primitive_at(
        &mut self,
        kind: DebugPrimitiveKind,
        position: cgmath::Vector3<f32>,
    ) -> Result<()> {
        log!("Spawning debug primitive: {:?}", kind);
        self.rrdevice.device.device_wait_idle()?;

        let command_pool = self.resource::<CommandState>().pool.clone();
        let swapchain = self.resource::<SwapchainState>().swapchain.clone();

        let (load_result, part_name) = match kind {
            DebugPrimitiveKind::Cube => (
                thyllore_importer_core::primitive::build_cube_model(1.0),
                "Cube",
            ),
            DebugPrimitiveKind::Sphere => (
                thyllore_importer_core::primitive::build_uv_sphere_model(0.6, 32, 16),
                "Sphere",
            ),
            DebugPrimitiveKind::Floor => (
                thyllore_importer_core::primitive::build_box_model(
                    12.0,
                    0.2,
                    12.0,
                    [0.8, 0.8, 0.8, 1.0],
                ),
                "Floor",
            ),
        };
        let parent_entity = crate::app::model_loader::append_model_to_scene(
            &load_result,
            part_name,
            &self.instance,
            &self.rrdevice,
            &command_pool,
            &swapchain,
            &mut self.data.graphics_resources,
            &mut self.data.raytracing,
            &mut self.data.ecs_world,
            &mut self.data.ecs_assets,
        )?;

        self.data.ecs_world.insert_component(
            parent_entity,
            crate::ecs::component::DebugPrimitiveTag { kind },
        );

        let mut transform = self
            .data
            .ecs_world
            .get_component_mut::<crate::ecs::world::Transform>(parent_entity)
            .unwrap();
        transform.translation = position;

        msg_info!(
            "Debug primitive spawned: {:?} at ({:.1}, {:.1}, {:.1})",
            kind,
            position.x,
            position.y,
            position.z
        );
        Ok(())
    }

    pub unsafe fn spawn_pending_debug_primitives(&mut self) {
        let requests = match self
            .data
            .ecs_world
            .get_resource_mut::<crate::ecs::resource::PendingDebugPrimitives>()
        {
            Some(mut pending) => pending.take_requests(),
            None => return,
        };

        for request in requests {
            if let Err(e) = self.spawn_debug_primitive_at(request.kind, request.position) {
                log_error!(
                    "Failed to spawn scene debug primitive {:?}: {:?}",
                    request.kind,
                    e
                );
            }
        }
    }

    pub unsafe fn delete_entities(&mut self, entities: &[u64]) -> Result<()> {
        self.rrdevice.device.device_wait_idle()?;

        for &entity in entities {
            let mesh_ref = self
                .data
                .ecs_world
                .get_component::<crate::ecs::world::MeshRef>(entity)
                .cloned();

            if let Some(mesh_ref) = mesh_ref {
                let graphics_mesh_index = self
                    .data
                    .ecs_assets
                    .get_mesh(mesh_ref.mesh_asset_id)
                    .map(|m| m.graphics_mesh_index);

                if let Some(idx) = graphics_mesh_index {
                    if idx < self.data.graphics_resources.meshes.len() {
                        self.data.graphics_resources.meshes[idx].render_to_gbuffer = false;
                        self.data.graphics_resources.meshes[idx].destroy(&self.rrdevice);
                    }
                }
            }

            self.data.ecs_world.despawn(entity);
        }

        let command_pool = self.resource::<CommandState>().pool.clone();
        let waters = crate::app::model_loader::collect_water_instances(&self.data.ecs_world);
        let mesh_transforms = crate::app::model_loader::collect_mesh_transforms(
            &self.data.ecs_world,
            &self.data.ecs_assets,
        );
        let water_instances =
            crate::app::model_loader::collect_water_instances(&self.data.ecs_world);
        crate::app::model_loader::rebuild_acceleration_structures(
            &self.instance,
            &self.rrdevice,
            &command_pool,
            &self.data.graphics_resources,
            &mut self.data.raytracing,
            &waters,
            &mesh_transforms,
        )?;

        log!("Deleted {} entities with GPU cleanup", entities.len());
        Ok(())
    }
}

fn default_debug_primitive_position(kind: DebugPrimitiveKind) -> cgmath::Vector3<f32> {
    match kind {
        DebugPrimitiveKind::Cube => cgmath::Vector3::new(3.0, 0.5, 0.0),
        DebugPrimitiveKind::Sphere => cgmath::Vector3::new(-3.0, 0.6, 0.0),
        DebugPrimitiveKind::Floor => cgmath::Vector3::new(0.0, -1.6, 0.0),
    }
}
