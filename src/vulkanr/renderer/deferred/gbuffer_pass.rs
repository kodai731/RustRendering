use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::ecs::world::MeshRef;

pub unsafe fn record_gbuffer_pass(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let gbuffer = app
        .data
        .raytracing
        .gbuffer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("G-Buffer not initialized"))?;
    let pipeline = app
        .data
        .raytracing
        .gbuffer_pipeline
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("G-Buffer pipeline not initialized"))?;
    let render_targets = app.resource::<crate::vulkanr::context::RenderTargets>();
    let render_pass = render_targets.render.gbuffer_render_pass;
    let framebuffer = render_targets.render.gbuffer_framebuffer;
    drop(render_targets);

    let draw_mesh_indices = collect_gbuffer_mesh_indices(app);
    let heatmap_mode = resolve_heatmap_mode(app);

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

    thyllore_vulkan_core::renderer::record_gbuffer_pass(
        &ctx,
        gbuffer,
        pipeline,
        render_pass,
        framebuffer,
        &draw_mesh_indices,
        heatmap_mode,
        command_buffer,
    )
}

fn collect_gbuffer_mesh_indices(app: &App) -> Vec<usize> {
    let ecs_world = &app.data.ecs_world;
    let ecs_assets = &app.data.ecs_assets;

    if ecs_world.has_mesh_entities() {
        ecs_world
            .query_renderable()
            .iter()
            .filter_map(|&entity| {
                let mesh_ref = ecs_world.get_component::<MeshRef>(entity)?;
                let mesh_asset = ecs_assets.get_mesh(mesh_ref.mesh_asset_id)?;
                if !mesh_asset.render_to_gbuffer {
                    return None;
                }
                Some(mesh_asset.graphics_mesh_index)
            })
            .collect()
    } else {
        (0..app.data.graphics_resources.meshes.len()).collect()
    }
}

fn resolve_heatmap_mode(app: &App) -> u32 {
    app.data
        .ecs_world
        .get_resource::<crate::ecs::resource::WeightHeatmapState>()
        .map(|state| if state.enabled { 1 } else { 0 })
        .unwrap_or(0)
}
