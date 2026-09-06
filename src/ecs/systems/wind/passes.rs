use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::ecs::component::WindTornadoEffect;
use crate::ecs::resource::{ProjectionData, WindRenderSettings, WindRenderTargets};
use crate::hooks::pass::{
    CoreTarget, PassStage, RenderPassNode, TargetAccess, TargetRef, TargetUse,
};
use crate::vulkanr::renderer::deferred::compute_bounds_scissor;
use thyllore_effect_core::{
    build_wind_ubo, inverse_view_proj_f64, wind_local_bounds_corners, WindShellParams, WindUBO,
};

pub struct WindPassNode;

/// Everything the wind node agrees on for one frame. `None` means nothing records this frame.
struct WindFrame {
    ubos: Vec<WindUBO>,
    scissors: Vec<Option<vk::Rect2D>>,
}

impl WindFrame {
    fn has_visible_instance(&self) -> bool {
        self.scissors.iter().any(Option::is_some)
    }
}

fn wind_frame(app: &App) -> Option<WindFrame> {
    let extent = app
        .data
        .ecs_world
        .get_resource::<WindRenderTargets>()?
        .buffer
        .extent();
    app.data.raytracing.wind_shading_pipeline.as_ref()?;
    app.data.raytracing.wind_descriptor.as_ref()?;
    app.data.raytracing.wind_ubo.as_ref()?;

    let mut winds = app.data.ecs_world.query_winds();
    winds.truncate(thyllore_vulkan_core::resource::MAX_WIND_INSTANCES);
    if winds.is_empty() {
        return None;
    }

    let inv_view_proj = app
        .data
        .ecs_world
        .get_resource::<ProjectionData>()
        .map(|projection| inverse_view_proj_f64(projection.proj, projection.view));
    let mut ubos = Vec::with_capacity(winds.len());
    let mut scissors = Vec::with_capacity(winds.len());
    for wind in winds {
        let effect = app
            .data
            .ecs_world
            .get_component::<WindTornadoEffect>(wind)?;
        let mut ubo = build_wind_ubo(&effect);
        if let Some(inv_view_proj) = inv_view_proj {
            ubo.inv_view_proj = inv_view_proj;
        }
        let params = WindShellParams::from_effect(&effect);
        scissors.push(compute_bounds_scissor(
            app,
            extent,
            &ubo.model,
            wind_local_bounds_corners(&params),
        ));
        ubos.push(ubo);
    }

    Some(WindFrame { ubos, scissors })
}

impl RenderPassNode for WindPassNode {
    fn name(&self) -> &'static str {
        "wind"
    }

    fn stage(&self) -> PassStage {
        PassStage::Effect
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        wind_frame(app)
            .filter(WindFrame::has_visible_instance)
            .map(|_| {
                vec![TargetUse::new(
                    TargetRef::Core(CoreTarget::HdrColor),
                    TargetAccess::Attachment {
                        initial_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    },
                )]
            })
            .unwrap_or_default()
    }

    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        _frame_slot: usize,
    ) -> Result<()> {
        record_wind_passes(app, command_buffer, image_index)
    }
}

unsafe fn record_wind_passes(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let Some(frame) = wind_frame(app) else {
        return Ok(());
    };
    let (Some(wind_targets), Some(shading_pipeline), Some(descriptor), Some(wind_ubo)) = (
        app.data.ecs_world.get_resource::<WindRenderTargets>(),
        app.data.raytracing.wind_shading_pipeline.as_ref(),
        app.data.raytracing.wind_descriptor.as_ref(),
        app.data.raytracing.wind_ubo.as_ref(),
    ) else {
        return Ok(());
    };
    let wind_buffer = &wind_targets.buffer;
    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

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

    for (slot, (ubo, scissor)) in frame.ubos.iter().zip(&frame.scissors).enumerate() {
        let Some(scissor) = *scissor else {
            continue;
        };
        wind_ubo.record_update(
            &ctx.device.device,
            command_buffer,
            slot,
            ubo,
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
