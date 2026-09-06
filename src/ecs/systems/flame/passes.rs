use anyhow::Result;
use cgmath::{InnerSpace, SquareMatrix, Vector3};
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::ecs::world::Entity;
use crate::hooks::pass::{
    CoreTarget, PassStage, RenderPassNode, ShaderStage, TargetAccess, TargetRef, TargetUse,
};
use crate::vulkanr::renderer::deferred::full_extent_scissor;
use thyllore_vulkan_core::resource::RenderTargetKey;

const HISTORY_KEYS: [RenderTargetKey; 2] = [
    RenderTargetKey::EffectHistory(0),
    RenderTargetKey::EffectHistory(1),
];

pub struct FlamePassNode;

fn flame_history_index(app: &App, flames: &[Entity]) -> usize {
    flames
        .first()
        .and_then(|first| {
            app.data
                .ecs_world
                .get_component::<crate::ecs::component::FlameTemporalAccum>(*first)
        })
        .map(|temporal| (temporal.frame_index as usize) & 1)
        .unwrap_or(0)
}

/// Everything the flame node agrees on for one frame. `None` means nothing records this frame.
struct FlameFrame {
    ubos: Vec<thyllore_effect_core::FlameUBO>,
    scissors: Vec<Option<vk::Rect2D>>,
    history_index: usize,
}

impl FlameFrame {
    fn has_visible_instance(&self) -> bool {
        self.scissors.iter().any(Option::is_some)
    }
}

fn flame_frame(app: &App) -> Option<FlameFrame> {
    let extent = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::FlameRenderTargets>()?
        .buffer
        .extent();
    app.data.raytracing.flame_shading_pipeline.as_ref()?;
    app.data.raytracing.flame_descriptor.as_ref()?;
    app.data.raytracing.flame_ubo.as_ref()?;

    let mut flames = app.data.ecs_world.query_flames();
    sort_flames_back_to_front(app, &mut flames);
    flames.truncate(thyllore_vulkan_core::resource::MAX_FLAME_INSTANCES);
    if flames.is_empty() {
        return None;
    }

    let ubos: Vec<_> = flames
        .iter()
        .map(|flame| build_instance_ubo(app, *flame))
        .collect::<Option<_>>()?;
    let scissors = ubos
        .iter()
        .map(|ubo| instance_scissor(app, extent, ubo))
        .collect();
    let history_index = flame_history_index(app, &flames);

    Some(FlameFrame {
        ubos,
        scissors,
        history_index,
    })
}

fn sort_flames_back_to_front(app: &App, flames: &mut [Entity]) {
    let Some(projection) = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::ProjectionData>()
    else {
        return;
    };
    let view_inverse = projection
        .view
        .invert()
        .unwrap_or_else(cgmath::Matrix4::identity);
    let camera_pos = Vector3::new(view_inverse[3][0], view_inverse[3][1], view_inverse[3][2]);

    let camera_distance = |flame: &Entity| {
        app.data
            .ecs_world
            .get_component::<crate::ecs::component::FlameEffect>(*flame)
            .map(|effect| {
                let position =
                    Vector3::new(effect.position[0], effect.position[1], effect.position[2]);
                (position - camera_pos).magnitude()
            })
    };
    flames.sort_by(|a, b| match (camera_distance(a), camera_distance(b)) {
        (Some(dist_a), Some(dist_b)) => dist_b
            .partial_cmp(&dist_a)
            .unwrap_or(std::cmp::Ordering::Equal),
        _ => std::cmp::Ordering::Equal,
    });
}

fn build_instance_ubo(app: &App, flame: Entity) -> Option<thyllore_effect_core::FlameUBO> {
    let world = &app.data.ecs_world;
    let effect = world.get_component::<crate::ecs::component::FlameEffect>(flame)?;
    let trail = world.get_component::<crate::ecs::component::FlameTrail>(flame);
    let is_noise_mode = world
        .get_resource::<crate::ecs::resource::FlameRenderSettings>()
        .map(|s| s.shading_mode == thyllore_effect_core::FlameShadingMode::NoiseRaymarch)
        .unwrap_or(false);
    let baked = world
        .get_component::<crate::ecs::component::FlameBaked>(flame)
        .cloned()
        .unwrap_or_default();
    let temporal_accum = world
        .get_component::<crate::ecs::component::FlameTemporalAccum>(flame)
        .cloned()
        .unwrap_or_default();

    Some(thyllore_effect_core::build_flame_ubo_with_trail(
        &effect,
        &baked,
        &temporal_accum,
        trail.as_deref().map(|t| &t.state),
        is_noise_mode,
    ))
}

fn instance_scissor(
    app: &App,
    extent: vk::Extent2D,
    ubo: &thyllore_effect_core::FlameUBO,
) -> Option<vk::Rect2D> {
    let bend_offset = [
        ubo.wind_bend.wind_direction[0] * ubo.wind_bend.bend_amount,
        ubo.wind_bend.wind_direction[1] * ubo.wind_bend.bend_amount,
    ];
    let support_scale = thyllore_effect_core::flame_shell_support_scale(
        ubo.emitter_params.kind as u32,
        ubo.emitter_params.ring_major_ratio,
        ubo.support_motion.support_margin,
    );
    compute_flame_scissor(
        app,
        extent,
        &ubo.model,
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
    )
}

impl RenderPassNode for FlamePassNode {
    fn name(&self) -> &'static str {
        "flame"
    }

    fn stage(&self) -> PassStage {
        PassStage::Effect
    }

    fn reads(&self, app: &App) -> Vec<TargetUse> {
        flame_frame(app)
            .filter(FlameFrame::has_visible_instance)
            .map(|frame| frame.history_index)
            .map(|history_index| {
                TargetUse::new(
                    TargetRef::Storage(HISTORY_KEYS[1 - history_index]),
                    TargetAccess::Sampled(ShaderStage::Fragment),
                )
            })
            .into_iter()
            .collect()
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        flame_frame(app)
            .filter(FlameFrame::has_visible_instance)
            .map(|frame| frame.history_index)
            .map(|history_index| {
                vec![
                    TargetUse::new(
                        TargetRef::Core(CoreTarget::HdrColor),
                        TargetAccess::Attachment {
                            initial_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                    ),
                    TargetUse::new(
                        TargetRef::Storage(HISTORY_KEYS[history_index]),
                        TargetAccess::Attachment {
                            initial_layout: vk::ImageLayout::UNDEFINED,
                            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                    ),
                ]
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
        record_flame_passes(app, command_buffer, image_index)
    }
}

unsafe fn record_flame_passes(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let Some(frame) = flame_frame(app) else {
        return Ok(());
    };
    let (Some(flame_targets), Some(shading_pipeline), Some(descriptor), Some(flame_ubo)) = (
        app.data
            .ecs_world
            .get_resource::<crate::ecs::resource::FlameRenderTargets>(),
        app.data.raytracing.flame_shading_pipeline.as_ref(),
        app.data.raytracing.flame_descriptor.as_ref(),
        app.data.raytracing.flame_ubo.as_ref(),
    ) else {
        return Ok(());
    };
    let flame_buffer = &flame_targets.buffer;
    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

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

    for (i, (ubo, scissor)) in frame.ubos.iter().zip(&frame.scissors).enumerate() {
        let Some(scissor) = *scissor else {
            continue;
        };
        let ubo_dynamic_offset = flame_ubo.slot_offset(i)? as u32;
        flame_ubo.record_update(
            &ctx.device.device,
            command_buffer,
            i,
            ubo,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        )?;

        thyllore_vulkan_core::renderer::record_flame_shading_pass(
            &ctx,
            flame_buffer,
            shading_pipeline,
            descriptor,
            frame.history_index,
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
