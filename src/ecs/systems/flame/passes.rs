use anyhow::Result;
use cgmath::{SquareMatrix, Vector3};
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::hooks::pass::{PassStage, RenderPassNode};
use crate::vulkanr::renderer::deferred::full_extent_scissor;

pub struct FlamePassNode;

impl RenderPassNode for FlamePassNode {
    fn name(&self) -> &'static str {
        "flame"
    }

    fn stage(&self) -> PassStage {
        PassStage::Effect
    }

    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        _frame_slot: usize,
    ) -> Result<()> {
        record_flame_passes(app, command_buffer, image_index)
    }
}

unsafe fn record_flame_passes(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let (Some(flame_targets), Some(shading_pipeline), Some(descriptor)) = (
        app.data
            .ecs_world
            .get_resource::<crate::ecs::resource::FlameRenderTargets>(),
        app.data.raytracing.flame_shading_pipeline.as_ref(),
        app.data.raytracing.flame_descriptor.as_ref(),
    ) else {
        return Ok(());
    };
    let flame_buffer = &flame_targets.buffer;

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

    let mut flames = app.data.ecs_world.query_flames();

    // Sort flames by descending camera distance (back-to-front) for correct overdraw
    if let Some(projection) = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::ProjectionData>()
    {
        let view_inverse = projection
            .view
            .invert()
            .unwrap_or_else(|| cgmath::Matrix4::identity());
        let camera_pos =
            cgmath::Vector3::new(view_inverse[3][0], view_inverse[3][1], view_inverse[3][2]);
        flames.sort_by(|a, b| {
            let effect_a = app
                .data
                .ecs_world
                .get_component::<crate::ecs::component::FlameEffect>(*a);
            let effect_b = app
                .data
                .ecs_world
                .get_component::<crate::ecs::component::FlameEffect>(*b);
            match (effect_a, effect_b) {
                (Some(ea), Some(eb)) => {
                    let pos_a = Vector3::new(ea.position[0], ea.position[1], ea.position[2]);
                    let dist_a = ((pos_a.x - camera_pos.x).powi(2)
                        + (pos_a.y - camera_pos.y).powi(2)
                        + (pos_a.z - camera_pos.z).powi(2))
                    .sqrt();
                    let pos_b = Vector3::new(eb.position[0], eb.position[1], eb.position[2]);
                    let dist_b = ((pos_b.x - camera_pos.x).powi(2)
                        + (pos_b.y - camera_pos.y).powi(2)
                        + (pos_b.z - camera_pos.z).powi(2))
                    .sqrt();
                    dist_b
                        .partial_cmp(&dist_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
                _ => std::cmp::Ordering::Equal,
            }
        });
    }

    let instance_count = flames
        .len()
        .min(thyllore_vulkan_core::resource::MAX_FLAME_INSTANCES);

    if instance_count == 0 {
        return Ok(());
    }

    // Calculate history_index from the first flame's frame_index (shared by all instances)
    let history_index = if let Some(first) = flames.first() {
        if let Some(temporal) = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::FlameTemporalAccum>(*first)
        {
            (temporal.frame_index as usize) & 1
        } else {
            0
        }
    } else {
        0
    };

    // Process each instance sequentially: F1_i -> F2_i before moving to i+1
    for i in 0..instance_count {
        let flame = flames[i];
        let effect = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::FlameEffect>(flame)
            .ok_or_else(|| anyhow::anyhow!("Missing FlameEffect for instance {}", i))?;

        // Build UBO for this instance (trail-aware)
        let trail = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::FlameTrail>(flame);
        let is_noise_mode = app
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::FlameRenderSettings>()
            .map(|s| s.shading_mode == thyllore_effect_core::FlameShadingMode::NoiseRaymarch)
            .unwrap_or(false);
        let baked = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::FlameBaked>(flame)
            .cloned()
            .unwrap_or_default();
        let temporal_accum = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::FlameTemporalAccum>(flame)
            .cloned()
            .unwrap_or_default();
        let ubo = thyllore_effect_core::build_flame_ubo_with_trail(
            effect,
            &baked,
            &temporal_accum,
            trail.map(|t| &t.state),
            is_noise_mode,
        );

        let Some(flame_ubo) = app.data.raytracing.flame_ubo.as_ref() else {
            return Ok(());
        };
        let ubo_dynamic_offset = flame_ubo.slot_offset(i)? as u32;
        flame_ubo.record_update(
            &ctx.device.device,
            command_buffer,
            i,
            &ubo,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        )?;

        // Build per-instance model matrix
        let model_matrix = ubo.model;

        // Compute per-instance scissor using the model matrix
        let bend_offset = [
            ubo.wind_bend.wind_direction[0] * ubo.wind_bend.bend_amount,
            ubo.wind_bend.wind_direction[1] * ubo.wind_bend.bend_amount,
        ];
        let support_scale = thyllore_effect_core::flame_shell_support_scale(
            ubo.emitter_params.kind as u32,
            ubo.emitter_params.ring_major_ratio,
            ubo.support_motion.support_margin,
        );
        let Some(scissor) = compute_flame_scissor(
            app,
            flame_buffer.extent(),
            &model_matrix,
            bend_offset,
            support_scale,
            ubo.support_motion.support_margin,
            thyllore_effect_core::FlameProxyPad {
                radial: thyllore_effect_core::flame_proxy_radial_pad(
                    ubo.branch_field.bounding_pad,
                    ubo.support_motion.meander_amp,
                ),
                top: ubo.branch_field.bounding_pad_y,
            },
        ) else {
            continue;
        };

        // Get render settings for push constants
        let settings = app
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::FlameRenderSettings>()
            .map(|settings| *settings)
            .unwrap_or_default();
        let push_constants = thyllore_vulkan_core::renderer::FlamePushConstants::new(
            settings.shading_mode.as_shader_value(),
            settings.resolved_step_count() as i32,
            settings.debug_view.as_shader_value(),
        );

        // Record shading pass for this instance (F2_i completed before next instance)
        thyllore_vulkan_core::renderer::record_flame_shading_pass(
            &ctx,
            flame_buffer,
            shading_pipeline,
            descriptor,
            history_index,
            ubo_dynamic_offset,
            scissor,
            push_constants,
            image_index,
            command_buffer,
        )?;
    }

    Ok(())
}

fn compute_flame_scissor(
    app: &App,
    extent: vk::Extent2D,
    model: &cgmath::Matrix4<f32>,
    bend_offset: [f32; 2],
    support_scale: f32,
    support_margin: f32,
    proxy_pad: thyllore_effect_core::FlameProxyPad,
) -> Option<vk::Rect2D> {
    use crate::ecs::resource::ProjectionData;
    const SCISSOR_MARGIN_PX: f32 = 2.0;

    let Some(projection) = app.data.ecs_world.get_resource::<ProjectionData>() else {
        return Some(full_extent_scissor(extent));
    };
    let view_proj = projection.proj * projection.view;

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let bounds = thyllore_effect_core::flame_local_bounds(
        bend_offset,
        support_scale,
        support_margin,
        proxy_pad,
    );
    for corner in thyllore_effect_core::flame_local_bounds_corners(&bounds) {
        let clip = view_proj * model * cgmath::vec4(corner.x, corner.y, corner.z, 1.0);
        if clip.w <= 0.0 {
            return Some(full_extent_scissor(extent));
        }
        let screen_x = (clip.x / clip.w + 1.0) * 0.5 * extent.width as f32;
        let screen_y = (clip.y / clip.w + 1.0) * 0.5 * extent.height as f32;
        min_x = min_x.min(screen_x);
        min_y = min_y.min(screen_y);
        max_x = max_x.max(screen_x);
        max_y = max_y.max(screen_y);
    }

    let min_x = (min_x - SCISSOR_MARGIN_PX).clamp(0.0, extent.width as f32);
    let min_y = (min_y - SCISSOR_MARGIN_PX).clamp(0.0, extent.height as f32);
    let max_x = (max_x + SCISSOR_MARGIN_PX).clamp(0.0, extent.width as f32);
    let max_y = (max_y + SCISSOR_MARGIN_PX).clamp(0.0, extent.height as f32);
    if max_x - min_x < 1.0 || max_y - min_y < 1.0 {
        return None;
    }

    Some(
        vk::Rect2D::builder()
            .offset(vk::Offset2D {
                x: min_x as i32,
                y: min_y as i32,
            })
            .extent(vk::Extent2D {
                width: (max_x - min_x).ceil() as u32,
                height: (max_y - min_y).ceil() as u32,
            })
            .build(),
    )
}
