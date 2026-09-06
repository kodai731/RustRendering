use anyhow::Result;
use cgmath::SquareMatrix;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrRayTracingPipelineExtension;

use crate::app::App;
use crate::ecs::resource::{WaterBindingKey, WaterRenderTargets};
use crate::ecs::world::Entity;
use crate::hooks::pass::{
    CoreTarget, PassStage, RenderPassNode, ShaderStage, TargetAccess, TargetRef, TargetUse,
    TransientRequest, TransientSlot,
};
use crate::vulkanr::renderer::deferred::full_extent_scissor;
use thyllore_vulkan_core::resource::RenderTargetKey;

/// Pass nodes in record order. Subscription order inside the effect stage is this order.
pub const WATER_PASS_NODES: &[&dyn RenderPassNode] = &[
    &WaterFrameNode,
    &WaterTraceNode,
    &WaterCausticClearNode,
    &WaterCausticSplatNode,
    &WaterCausticApplyNode,
    &WaterSceneColorCopyNode,
    &WaterHistoryClearNode,
    &WaterShadingNode,
];

const HDR_COLOR: TargetRef = TargetRef::Core(CoreTarget::HdrColor);
const SCENE_COLOR_SLOT: TransientSlot = TransientSlot("water.scene_color");
const TRACE_SLOT: TransientSlot = TransientSlot("water.trace");
const SCENE_COLOR: TargetRef = TargetRef::Transient(SCENE_COLOR_SLOT);
const TRACE: TargetRef = TargetRef::Transient(TRACE_SLOT);
const CAUSTIC_ACCUM: TargetRef = TargetRef::Storage(RenderTargetKey::CausticAccum);
const HISTORY_KEYS: [RenderTargetKey; 2] = [
    RenderTargetKey::EffectHistory(2),
    RenderTargetKey::EffectHistory(3),
];

/// Must match CAUSTIC_GRID_SIZE and local_size in waterCausticSplat.comp
const CAUSTIC_GRID_SIZE: u32 = 512;
const CAUSTIC_WORKGROUP_SIZE: u32 = 16;

const COLOR_SUBRESOURCE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// Everything the water nodes agree on for one frame. `None` means no water pass records this frame.
struct WaterFrame {
    waters: Vec<Entity>,
    history_index: usize,
    settings: crate::ecs::resource::WaterRenderSettings,
}

fn water_frame(app: &App) -> Option<WaterFrame> {
    app.data.ecs_world.get_resource::<WaterRenderTargets>()?;
    app.data.raytracing.water_shading_pipeline.as_ref()?;
    app.data.raytracing.water_descriptor.as_ref()?;
    app.data.raytracing.water_ubo.as_ref()?;
    app.data.viewport.hdr_buffer.as_ref()?;
    water_scene_bindings(app)?;

    let mut waters: Vec<Entity> = app.data.ecs_world.query_waters();
    waters.truncate(thyllore_vulkan_core::resource::MAX_WATER_INSTANCES);
    if waters.is_empty() {
        return None;
    }

    let history_index = first_water_accum(app, &waters)
        .map(|accum| (accum.frame_index & 1) as usize)
        .unwrap_or(0);
    let settings = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::WaterRenderSettings>()
        .map(|settings| *settings)
        .unwrap_or_default();

    Some(WaterFrame {
        waters,
        history_index,
        settings,
    })
}

fn water_scene_bindings(app: &App) -> Option<(vk::AccelerationStructureKHR, vk::Buffer)> {
    if !app.data.raytracing.has_valid_tlas() {
        return None;
    }
    let accel = app.data.raytracing.acceleration_structure.as_ref()?;
    let tlas = accel.tlas.acceleration_structure?;
    let hit_table = accel.hit_shading_table.as_ref()?.buffer;
    Some((tlas, hit_table))
}

fn first_water_accum(
    app: &App,
    waters: &[Entity],
) -> Option<crate::ecs::component::WaterTemporalAccum> {
    waters.first().and_then(|water| {
        app.data
            .ecs_world
            .get_component::<crate::ecs::component::WaterTemporalAccum>(*water)
            .cloned()
    })
}

impl WaterFrame {
    fn is_trace_enabled(&self, app: &App) -> bool {
        self.settings.secondary_rays == thyllore_effect_core::WaterSecondaryRays::RayTracingPipeline
            && app.data.raytracing.water_trace_pipeline.is_some()
            && app.data.raytracing.water_trace_descriptor.is_some()
            && app
                .data
                .ecs_world
                .get_component::<crate::ecs::component::WaterTorusEffect>(self.waters[0])
                .is_some()
    }

    fn is_caustic_enabled(&self, app: &App) -> bool {
        let caustic_strength = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::WaterTorusEffect>(self.waters[0])
            .map(|effect| effect.caustic_strength)
            .unwrap_or(0.0);
        caustic_strength > 0.0
            && app.data.raytracing.water_caustic_splat_pipeline.is_some()
            && app.data.raytracing.water_caustic_apply_pipeline.is_some()
            && app
                .data
                .raytracing
                .water_caustic_descriptor
                .as_ref()
                .is_some_and(|descriptor| {
                    descriptor.splat_descriptor_set != vk::DescriptorSet::null()
                })
    }

    fn is_history_invalidated(&self, app: &App) -> bool {
        first_water_accum(app, &self.waters).is_some_and(|accum| accum.history_invalidated)
    }

    fn written_history(&self) -> TargetRef {
        TargetRef::Storage(HISTORY_KEYS[self.history_index])
    }

    fn read_history(&self) -> TargetRef {
        TargetRef::Storage(HISTORY_KEYS[1 - self.history_index])
    }
}

fn frame_uses(
    app: &App,
    select: impl FnOnce(&WaterFrame, &App) -> Vec<TargetUse>,
) -> Vec<TargetUse> {
    water_frame(app)
        .map(|frame| select(&frame, app))
        .unwrap_or_default()
}

pub struct WaterTraceNode;

impl RenderPassNode for WaterTraceNode {
    fn name(&self) -> &'static str {
        "water_trace"
    }

    fn stage(&self) -> PassStage {
        PassStage::Effect
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        frame_uses(app, |frame, app| {
            if !frame.is_trace_enabled(app) {
                return Vec::new();
            }
            vec![TargetUse::new(
                TRACE,
                TargetAccess::StorageReadWrite(ShaderStage::RayTracing),
            )]
        })
    }

    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        frame_slot: usize,
    ) -> Result<()> {
        let Some(frame) = water_frame(app).filter(|frame| frame.is_trace_enabled(app)) else {
            return Ok(());
        };
        let (Some(trace_pipeline), Some(trace_descriptor), Some(effect)) = (
            app.data.raytracing.water_trace_pipeline.as_ref(),
            app.data.raytracing.water_trace_descriptor.as_ref(),
            app.data
                .ecs_world
                .get_component::<crate::ecs::component::WaterTorusEffect>(frame.waters[0]),
        ) else {
            return Ok(());
        };
        let Some(targets) = app.data.ecs_world.get_resource::<WaterRenderTargets>() else {
            return Ok(());
        };
        let water_buffer = &targets.buffer;
        let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

        let device = &ctx.device.device;
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            trace_pipeline.pipeline,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            trace_pipeline.pipeline_layout,
            0,
            &[trace_descriptor.descriptor_set(frame_slot)?],
            &[],
        );
        let radii = [effect.major_radius, effect.minor_radius];
        let radii_bytes = std::slice::from_raw_parts(radii.as_ptr() as *const u8, 8);
        device.cmd_push_constants(
            command_buffer,
            trace_pipeline.pipeline_layout,
            vk::ShaderStageFlags::INTERSECTION_KHR,
            0,
            radii_bytes,
        );
        let projection = app
            .data
            .ecs_world
            .resource::<crate::ecs::resource::ProjectionData>();
        let inv_view_proj = crate::ecs::systems::water::probe::inverse_view_proj_f64(
            projection.proj,
            projection.view,
        );
        let view_inverse = projection
            .view
            .invert()
            .unwrap_or_else(cgmath::Matrix4::identity);
        let m: &[f32; 16] = inv_view_proj.as_ref();
        let mut frame_data = [0.0f32; 20];
        frame_data[..16].copy_from_slice(m);
        frame_data[16] = view_inverse[3][0];
        frame_data[17] = view_inverse[3][1];
        frame_data[18] = view_inverse[3][2];
        frame_data[19] = 1.0;
        let frame_bytes = std::slice::from_raw_parts(frame_data.as_ptr() as *const u8, 80);
        device.cmd_push_constants(
            command_buffer,
            trace_pipeline.pipeline_layout,
            vk::ShaderStageFlags::RAYGEN_KHR,
            16,
            frame_bytes,
        );
        let light_position = app
            .data
            .ecs_world
            .resource::<crate::ecs::resource::LightState>()
            .light_position;
        let mut light_data = [0.0f32; 8];
        light_data[0] = light_position.x;
        light_data[1] = light_position.y;
        light_data[2] = light_position.z;
        light_data[3] = 1.0;
        light_data[4] = 1.0;
        light_data[5] = 1.0;
        light_data[6] = 1.0;
        light_data[7] = 1.0;
        let light_bytes = std::slice::from_raw_parts(light_data.as_ptr() as *const u8, 32);
        device.cmd_push_constants(
            command_buffer,
            trace_pipeline.pipeline_layout,
            vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            96,
            light_bytes,
        );
        let extent = water_buffer.extent();
        device.cmd_trace_rays_khr(
            command_buffer,
            &trace_pipeline.raygen_region,
            &trace_pipeline.miss_region,
            &trace_pipeline.hit_region,
            &trace_pipeline.callable_region,
            extent.width,
            extent.height,
            1,
        );
        Ok(())
    }
}

/// Borrows the frame's scene color / trace images, rebinds the water descriptors of this frame slot,
/// and uploads the per-instance UBOs before any water pass reads them.
pub struct WaterFrameNode;

impl RenderPassNode for WaterFrameNode {
    fn name(&self) -> &'static str {
        "water_frame"
    }

    fn stage(&self) -> PassStage {
        PassStage::Effect
    }

    fn transients(&self, app: &App) -> Vec<TransientRequest> {
        if water_frame(app).is_none() {
            return Vec::new();
        }
        let Some(targets) = app.data.ecs_world.get_resource::<WaterRenderTargets>() else {
            return Vec::new();
        };
        vec![
            TransientRequest::new(SCENE_COLOR_SLOT, targets.buffer.scene_color_desc()),
            TransientRequest::new(TRACE_SLOT, targets.buffer.trace_desc()),
        ]
    }

    unsafe fn prepare(&self, app: &mut App, frame_slot: usize) -> Result<()> {
        if water_frame(app).is_none() {
            return Ok(());
        }
        let Some((tlas, hit_table)) = water_scene_bindings(app) else {
            return Ok(());
        };
        let scene_color_image = app
            .data
            .viewport
            .transient
            .get(app.data.frame_transients.handle(SCENE_COLOR_SLOT)?)?;
        let trace_image = app
            .data
            .viewport
            .transient
            .get(app.data.frame_transients.handle(TRACE_SLOT)?)?;
        let Some(mut targets) = app.data.ecs_world.get_resource_mut::<WaterRenderTargets>() else {
            return Ok(());
        };

        let history_views = targets.buffer.history_image_views;
        let history_sampler = targets.buffer.history_sampler;
        let key = WaterBindingKey {
            tlas,
            hit_table,
            history_views,
            scene_color_generation: scene_color_image.generation,
            trace_generation: trace_image.generation,
        };
        if targets.is_bound(frame_slot, key) {
            return Ok(());
        }

        let Some(water_ubo) = app.data.raytracing.water_ubo.as_ref() else {
            return Ok(());
        };
        if let Some(descriptor) = app.data.raytracing.water_descriptor.as_ref() {
            descriptor.write_all_at(
                &app.rrdevice,
                frame_slot,
                water_ubo,
                scene_color_image.view,
                history_sampler,
                history_views,
                history_sampler,
                trace_image.view,
                history_sampler,
                tlas,
                hit_table,
            )?;
        }
        if let Some(trace_descriptor) = app.data.raytracing.water_trace_descriptor.as_ref() {
            trace_descriptor.write_all_at(
                &app.rrdevice,
                frame_slot,
                tlas,
                trace_image.view,
                water_ubo,
                hit_table,
            )?;
        }
        targets.mark_bound(frame_slot, key);
        Ok(())
    }

    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        _frame_slot: usize,
    ) -> Result<()> {
        let Some(frame) = water_frame(app) else {
            return Ok(());
        };
        let Some(water_ubo) = app.data.raytracing.water_ubo.as_ref() else {
            return Ok(());
        };
        let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);
        let instance_ubos =
            record_water_ubo_updates(app, &ctx, water_ubo, &frame.waters, command_buffer)?;

        if let Some(mut targets) = app.data.ecs_world.get_resource_mut::<WaterRenderTargets>() {
            targets.frame_instances = instance_ubos;
        }
        Ok(())
    }
}

pub struct WaterCausticClearNode;

impl RenderPassNode for WaterCausticClearNode {
    fn name(&self) -> &'static str {
        "water_caustic_clear"
    }

    fn stage(&self) -> PassStage {
        PassStage::Effect
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        frame_uses(app, |frame, app| {
            if !frame.is_caustic_enabled(app) {
                return Vec::new();
            }
            vec![TargetUse::new(CAUSTIC_ACCUM, TargetAccess::TransferDst)]
        })
    }

    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        _: usize,
        _: usize,
    ) -> Result<()> {
        if water_frame(app).is_none_or(|frame| !frame.is_caustic_enabled(app)) {
            return Ok(());
        }
        let Some(targets) = app.data.ecs_world.get_resource::<WaterRenderTargets>() else {
            return Ok(());
        };

        app.rrdevice.device.cmd_clear_color_image(
            command_buffer,
            targets.buffer.caustic_accum_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue { uint32: [0; 4] },
            &[COLOR_SUBRESOURCE_RANGE],
        );
        Ok(())
    }
}

/// Splats refracted light into the accumulation image.
/// The G-buffer position image stays in GENERAL from the ray query pass and is not declared here.
pub struct WaterCausticSplatNode;

impl RenderPassNode for WaterCausticSplatNode {
    fn name(&self) -> &'static str {
        "water_caustic_splat"
    }

    fn stage(&self) -> PassStage {
        PassStage::Effect
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        frame_uses(app, |frame, app| {
            if !frame.is_caustic_enabled(app) {
                return Vec::new();
            }
            vec![TargetUse::new(
                CAUSTIC_ACCUM,
                TargetAccess::StorageReadWrite(ShaderStage::Compute),
            )]
        })
    }

    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        _: usize,
        _: usize,
    ) -> Result<()> {
        if water_frame(app).is_none_or(|frame| !frame.is_caustic_enabled(app)) {
            return Ok(());
        }
        let (Some(splat_pipeline), Some(descriptor)) = (
            app.data.raytracing.water_caustic_splat_pipeline.as_ref(),
            app.data.raytracing.water_caustic_descriptor.as_ref(),
        ) else {
            return Ok(());
        };

        let device = &app.rrdevice.device;
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            splat_pipeline.pipeline,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            splat_pipeline.pipeline_layout,
            0,
            &[descriptor.splat_descriptor_set],
            &[],
        );
        let splat_group_count = CAUSTIC_GRID_SIZE / CAUSTIC_WORKGROUP_SIZE;
        device.cmd_dispatch(command_buffer, splat_group_count, splat_group_count, 1);
        Ok(())
    }
}

/// Adds the accumulated caustic light to the HDR color.
pub struct WaterCausticApplyNode;

impl RenderPassNode for WaterCausticApplyNode {
    fn name(&self) -> &'static str {
        "water_caustic_apply"
    }

    fn stage(&self) -> PassStage {
        PassStage::Effect
    }

    fn reads(&self, app: &App) -> Vec<TargetUse> {
        frame_uses(app, |frame, app| {
            if !frame.is_caustic_enabled(app) {
                return Vec::new();
            }
            vec![TargetUse::new(
                CAUSTIC_ACCUM,
                TargetAccess::StorageRead(ShaderStage::Compute),
            )]
        })
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        frame_uses(app, |frame, app| {
            if !frame.is_caustic_enabled(app) {
                return Vec::new();
            }
            vec![TargetUse::new(
                HDR_COLOR,
                TargetAccess::StorageReadWrite(ShaderStage::Compute),
            )]
        })
    }

    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        _: usize,
        _: usize,
    ) -> Result<()> {
        if water_frame(app).is_none_or(|frame| !frame.is_caustic_enabled(app)) {
            return Ok(());
        }
        let (Some(apply_pipeline), Some(descriptor), Some(hdr_buffer)) = (
            app.data.raytracing.water_caustic_apply_pipeline.as_ref(),
            app.data.raytracing.water_caustic_descriptor.as_ref(),
            app.data.viewport.hdr_buffer.as_ref(),
        ) else {
            return Ok(());
        };

        let device = &app.rrdevice.device;
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            apply_pipeline.pipeline,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            apply_pipeline.pipeline_layout,
            0,
            &[descriptor.apply_descriptor_set],
            &[],
        );
        device.cmd_dispatch(
            command_buffer,
            hdr_buffer.width.div_ceil(CAUSTIC_WORKGROUP_SIZE),
            hdr_buffer.height.div_ceil(CAUSTIC_WORKGROUP_SIZE),
            1,
        );
        Ok(())
    }
}

pub struct WaterSceneColorCopyNode;

impl RenderPassNode for WaterSceneColorCopyNode {
    fn name(&self) -> &'static str {
        "water_scene_color_copy"
    }

    fn stage(&self) -> PassStage {
        PassStage::Effect
    }

    fn reads(&self, app: &App) -> Vec<TargetUse> {
        frame_uses(app, |_, _| {
            vec![TargetUse::new(HDR_COLOR, TargetAccess::TransferSrc)]
        })
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        frame_uses(app, |frame, _| {
            vec![TargetUse::new(SCENE_COLOR, TargetAccess::TransferDst)]
        })
    }

    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        _: usize,
    ) -> Result<()> {
        if water_frame(app).is_none() {
            return Ok(());
        }
        let (Some(targets), Some(hdr_buffer)) = (
            app.data.ecs_world.get_resource::<WaterRenderTargets>(),
            app.data.viewport.hdr_buffer.as_ref(),
        ) else {
            return Ok(());
        };
        let scene_color_image = app
            .data
            .viewport
            .transient
            .get(app.data.frame_transients.handle(SCENE_COLOR_SLOT)?)?;
        let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

        thyllore_vulkan_core::renderer::record_water_scene_color_copy(
            &ctx,
            hdr_buffer.color_image,
            scene_color_image.image,
            targets.buffer.extent(),
            command_buffer,
        );
        Ok(())
    }
}

pub struct WaterHistoryClearNode;

impl RenderPassNode for WaterHistoryClearNode {
    fn name(&self) -> &'static str {
        "water_history_clear"
    }

    fn stage(&self) -> PassStage {
        PassStage::Effect
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        frame_uses(app, |frame, app| {
            if !frame.is_history_invalidated(app) {
                return Vec::new();
            }
            HISTORY_KEYS
                .iter()
                .map(|key| TargetUse::new(TargetRef::Storage(*key), TargetAccess::TransferDst))
                .collect()
        })
    }

    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        _: usize,
        _: usize,
    ) -> Result<()> {
        if water_frame(app).is_none_or(|frame| !frame.is_history_invalidated(app)) {
            return Ok(());
        }
        let Some(targets) = app.data.ecs_world.get_resource::<WaterRenderTargets>() else {
            return Ok(());
        };

        let black = vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        };
        for &image in &targets.buffer.history_images {
            app.rrdevice.device.cmd_clear_color_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &black,
                &[COLOR_SUBRESOURCE_RANGE],
            );
        }
        Ok(())
    }
}

pub struct WaterShadingNode;

impl RenderPassNode for WaterShadingNode {
    fn name(&self) -> &'static str {
        "water_shading"
    }

    fn stage(&self) -> PassStage {
        PassStage::Effect
    }

    fn reads(&self, app: &App) -> Vec<TargetUse> {
        frame_uses(app, |frame, _| {
            vec![
                TargetUse::new(SCENE_COLOR, TargetAccess::Sampled(ShaderStage::Fragment)),
                TargetUse::new(TRACE, TargetAccess::StorageRead(ShaderStage::Fragment)),
                TargetUse::new(
                    frame.read_history(),
                    TargetAccess::Sampled(ShaderStage::Fragment),
                ),
            ]
        })
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        frame_uses(app, |frame, _| {
            vec![
                TargetUse::new(
                    HDR_COLOR,
                    TargetAccess::Attachment {
                        initial_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    },
                ),
                TargetUse::new(
                    frame.written_history(),
                    TargetAccess::Attachment {
                        initial_layout: vk::ImageLayout::UNDEFINED,
                        final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    },
                ),
            ]
        })
    }

    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        frame_slot: usize,
    ) -> Result<()> {
        let Some(frame) = water_frame(app) else {
            return Ok(());
        };
        let (Some(targets), Some(shading_pipeline), Some(descriptor)) = (
            app.data.ecs_world.get_resource::<WaterRenderTargets>(),
            app.data.raytracing.water_shading_pipeline.as_ref(),
            app.data.raytracing.water_descriptor.as_ref(),
        ) else {
            return Ok(());
        };
        let water_buffer = &targets.buffer;
        let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

        for (i, (ubo, ubo_dynamic_offset)) in targets.frame_instances.iter().enumerate() {
            let effect = app
                .data
                .ecs_world
                .get_component::<crate::ecs::component::WaterTorusEffect>(frame.waters[i])
                .ok_or_else(|| anyhow::anyhow!("Missing WaterTorusEffect for instance {}", i))?;

            let Some(scissor) = compute_water_scissor(
                app,
                water_buffer.extent(),
                &ubo.model,
                effect.major_radius,
                effect.minor_radius,
            ) else {
                continue;
            };

            let push_constants = thyllore_vulkan_core::renderer::WaterPushConstants::new(
                frame.settings.secondary_rays.as_shader_value(),
                frame.settings.debug_view,
            );

            thyllore_vulkan_core::renderer::record_water_shading_pass(
                &ctx,
                water_buffer,
                shading_pipeline,
                descriptor,
                *ubo_dynamic_offset,
                scissor,
                push_constants,
                image_index,
                frame_slot,
                frame.history_index,
                command_buffer,
            )?;
        }
        Ok(())
    }
}

unsafe fn record_water_ubo_updates(
    app: &App,
    ctx: &thyllore_vulkan_core::FrameRenderContext,
    water_ubo: &thyllore_vulkan_core::resource::UniformBuffer<thyllore_effect_core::WaterUBO>,
    waters: &[crate::ecs::world::Entity],
    command_buffer: vk::CommandBuffer,
) -> Result<Vec<(thyllore_effect_core::WaterUBO, u32)>> {
    let projection = app
        .data
        .ecs_world
        .resource::<crate::ecs::resource::ProjectionData>();
    let inv_view_proj =
        crate::ecs::systems::water::probe::inverse_view_proj_f64(projection.proj, projection.view);
    let settings = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::WaterRenderSettings>()
        .map(|settings| *settings)
        .unwrap_or_default();

    let mut instance_ubos = Vec::with_capacity(waters.len());
    for (i, water) in waters.iter().enumerate() {
        let effect = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::WaterTorusEffect>(*water)
            .ok_or_else(|| anyhow::anyhow!("Missing WaterTorusEffect for instance {}", i))?;
        let accum = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::WaterTemporalAccum>(*water)
            .cloned()
            .unwrap_or_default();

        let mut ubo = thyllore_effect_core::build_water_ubo(effect, accum.frame_index as u32);
        ubo.inv_view_proj = inv_view_proj;
        ubo.temporal = [accum.weight, accum.frame_index as f32, 0.0, 0.0];
        ubo.composite[3] = settings.caustic_debug as f32;

        water_ubo.record_update(
            &ctx.device.device,
            command_buffer,
            i,
            &ubo,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
        )?;
        instance_ubos.push((ubo, water_ubo.slot_offset(i)? as u32));
    }
    Ok(instance_ubos)
}

fn compute_water_scissor(
    app: &App,
    extent: vk::Extent2D,
    model: &cgmath::Matrix4<f32>,
    major_radius: f32,
    minor_radius: f32,
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
    for corner in thyllore_math_core::torus_local_bounds_corners(major_radius, minor_radius) {
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
