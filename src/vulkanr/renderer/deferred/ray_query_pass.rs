use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::ecs::resource::HierarchyState;
use crate::ecs::world::MeshRef;

pub unsafe fn record_ray_query_pass(app: &App, command_buffer: vk::CommandBuffer) -> Result<()> {
    let gbuffer = app
        .data
        .raytracing
        .gbuffer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("G-Buffer not initialized"))?;
    let pipeline = app
        .data
        .raytracing
        .ray_query_pipeline
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Ray Query pipeline not initialized"))?;
    let descriptor = app
        .data
        .raytracing
        .ray_query_descriptor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Ray Query descriptor set not initialized"))?;

    let normal_offset = app
        .resource::<crate::ecs::resource::LightState>()
        .shadow_normal_offset;

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, 0);

    thyllore_vulkan_core::renderer::record_ray_query_pass(
        &ctx,
        gbuffer,
        pipeline,
        descriptor,
        normal_offset,
        command_buffer,
    )
}

pub(super) fn collect_selected_mesh_ids(app: &App) -> Vec<u32> {
    let hierarchy_state = app.data.ecs_world.resource::<HierarchyState>();
    let mut selected_ids = Vec::new();

    for &entity in hierarchy_state.multi_selection.iter() {
        if let Some(mesh_ref) = app.data.ecs_world.get_component::<MeshRef>(entity) {
            if let Some(mesh_asset) = app.data.ecs_assets.get_mesh(mesh_ref.mesh_asset_id) {
                let mesh_id = (mesh_asset.graphics_mesh_index + 1) as u32;
                if !selected_ids.contains(&mesh_id) {
                    selected_ids.push(mesh_id);
                }
            }
        }
    }

    selected_ids
}
