use crate::ecs::resource::gizmo::LightGizmoData;
use crate::ecs::resource::DebugViewState;
use crate::ecs::resource::{Camera, LightState};
use crate::ecs::systems::camera_systems::compute_camera_position;
use crate::ecs::world::World;
use crate::vulkanr::resource::graphics_resource::GraphicsResources;
use thyllore_vulkan_core::resource::raytracing_data::RayTracingData;

pub fn log_shadow_debug_info(
    world: &World,
    raytracing: &RayTracingData,
    graphics: &GraphicsResources,
) {
    let light = world.resource::<LightState>();
    let camera = world.resource::<Camera>();
    let cam_pos = compute_camera_position(&camera);

    log!("=== Shadow Debug Info ===");
    log!(
        "Light position: ({:.2}, {:.2}, {:.2})",
        light.light_position.x,
        light.light_position.y,
        light.light_position.z
    );

    let light_gizmo = world.resource::<LightGizmoData>();
    log!(
        "Light gizmo position: ({:.2}, {:.2}, {:.2})",
        light_gizmo.position.position.x,
        light_gizmo.position.position.y,
        light_gizmo.position.position.z
    );

    log!(
        "Camera position: ({:.2}, {:.2}, {:.2})",
        cam_pos.x,
        cam_pos.y,
        cam_pos.z
    );

    log!("Shadow settings:");
    log!("  strength: {:.2}", light.shadow_strength);
    log!("  normal_offset: {:.2}", light.shadow_normal_offset);

    let debug_view = world.resource::<DebugViewState>();
    log!("  debug_view_mode: {:?}", debug_view.debug_view_mode);
    log!(
        "  distance_attenuation: {}",
        light.distance_attenuation.is_enabled()
    );

    if let Some(ref accel_struct) = raytracing.acceleration_structure {
        log!("Acceleration Structure:");
        log!("  BLAS count: {}", accel_struct.blas_list.len());
        for (i, blas) in accel_struct.blas_list.iter().enumerate() {
            log!(
                "    BLAS[{}]: AS={:?}, device_addr={:#x}",
                i,
                blas.acceleration_structure.is_some(),
                blas.device_address
            );
        }
        log!(
            "  TLAS: AS={:?}",
            accel_struct.tlas.acceleration_structure.is_some()
        );
    } else {
        log_warn!("No acceleration structure!");
    }

    log!("Vertex buffers (GPU):");
    for (i, mesh) in graphics.meshes.iter().enumerate() {
        log!(
            "  Mesh[{}]: {} vertices, {} indices",
            i,
            mesh.vertex_data.vertices.len(),
            mesh.vertex_data.indices.len()
        );
        if !mesh.vertex_data.vertices.is_empty() {
            let v = &mesh.vertex_data.vertices[0];
            log!(
                "    vertex[0].pos: ({:.2}, {:.2}, {:.2})",
                v.pos.x,
                v.pos.y,
                v.pos.z
            );
            log!(
                "    vertex[0].normal: ({:.3}, {:.3}, {:.3})",
                v.normal.x,
                v.normal.y,
                v.normal.z
            );
        }
    }

    log!("=========================");
}
