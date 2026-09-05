use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::ecs::component::WindTornadoEffect;
use crate::ecs::resource::{ProjectionData, WindRenderSettings};
use crate::vulkanr::renderer::deferred::scissor::compute_bounds_scissor;
use thyllore_effect_core::{
    build_wind_ubo, inverse_view_proj_f64, wind_local_bounds_corners, WindShellParams,
};

pub unsafe fn record_wind_passes(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let (Some(wind_buffer), Some(shading_pipeline), Some(descriptor), Some(wind_ubo)) = (
        app.data.viewport.wind_buffer.as_ref(),
        app.data.raytracing.wind_shading_pipeline.as_ref(),
        app.data.raytracing.wind_descriptor.as_ref(),
        app.data.raytracing.wind_ubo.as_ref(),
    ) else {
        return Ok(());
    };

    let winds = app.data.ecs_world.query_winds();
    let instance_count = winds
        .len()
        .min(thyllore_vulkan_core::resource::MAX_WIND_INSTANCES);
    if instance_count == 0 {
        return Ok(());
    }

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);
    let inv_view_proj = app
        .data
        .ecs_world
        .get_resource::<ProjectionData>()
        .map(|projection| inverse_view_proj_f64(projection.proj, projection.view));
    let settings = app
        .data
        .ecs_world
        .get_resource::<WindRenderSettings>()
        .map(|settings| *settings)
        .unwrap_or_default();
    let push_constants = thyllore_vulkan_core::renderer::WindPushConstants::new(
        settings.shading_mode.as_shader_value(),
        settings.reference_step_count as i32,
        settings.debug_view.as_shader_value(),
    );

    for (slot, &entity) in winds.iter().take(instance_count).enumerate() {
        let Some(effect) = app
            .data
            .ecs_world
            .get_component::<WindTornadoEffect>(entity)
        else {
            continue;
        };
        let mut ubo = build_wind_ubo(&effect);
        if let Some(inv_view_proj) = inv_view_proj {
            ubo.inv_view_proj = inv_view_proj;
        }
        let params = WindShellParams::from_effect(&effect);
        let Some(scissor) = compute_bounds_scissor(
            app,
            wind_buffer.extent(),
            &ubo.model,
            wind_local_bounds_corners(&params),
        ) else {
            continue;
        };

        wind_ubo.record_update(
            &ctx.device.device,
            command_buffer,
            slot,
            &ubo,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        )?;
        thyllore_vulkan_core::renderer::record_wind_shading_pass(
            &ctx,
            wind_buffer,
            shading_pipeline,
            descriptor,
            wind_ubo.slot_offset(slot)? as u32,
            scissor,
            push_constants,
            image_index,
            command_buffer,
        )?;
    }

    Ok(())
}
