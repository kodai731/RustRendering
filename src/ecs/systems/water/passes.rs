use anyhow::Result;
use cgmath::SquareMatrix;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrRayTracingPipelineExtension;

use crate::app::App;
use crate::vulkanr::renderer::deferred::full_extent_scissor;

pub unsafe fn record_water_passes(
    app: &App,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    let (Some(water_buffer), Some(shading_pipeline), Some(descriptor)) = (
        app.data.viewport.water_buffer.as_ref(),
        app.data.raytracing.water_shading_pipeline.as_ref(),
        app.data.raytracing.water_descriptor.as_ref(),
    ) else {
        return Ok(());
    };

    if !app.data.raytracing.has_valid_tlas()
        || app
            .data
            .raytracing
            .acceleration_structure
            .as_ref()
            .map(|a| a.hit_shading_table.is_some())
            .unwrap_or(false)
            == false
    {
        return Ok(());
    }

    let ctx = crate::ecs::systems::phases::build_frame_render_context(app, image_index);

    let waters: Vec<_> = app.data.ecs_world.query_waters();

    let instance_count = waters
        .len()
        .min(thyllore_vulkan_core::resource::MAX_WATER_INSTANCES);

    if instance_count == 0 {
        return Ok(());
    }
    // Re-write descriptor with current TLAS and hit table (only if TLAS handle changed)
    {
        let accel = app.data.raytracing.acceleration_structure.as_ref().unwrap();
        let tlas = accel.tlas.acceleration_structure.unwrap();
        if tlas != app.data.raytracing.water_descriptor_tlas.get() {
            let hit_table = accel.hit_shading_table.as_ref().unwrap().buffer;
            let (scene_color_view, scene_color_sampler) = water_buffer.scene_color_binding();
            let water_ubo = app.data.raytracing.water_ubo.as_ref().unwrap();
            descriptor.write_all(
                ctx.device,
                water_ubo,
                scene_color_view,
                scene_color_sampler,
                water_buffer.history_image_views,
                water_buffer.history_sampler,
                water_buffer.trace_image_view,
                water_buffer.history_sampler,
                tlas,
                hit_table,
            )?;

            if let Some(trace_descriptor) = app.data.raytracing.water_trace_descriptor.as_ref() {
                if let Some(water_ubo) = app.data.raytracing.water_ubo.as_ref() {
                    trace_descriptor.write_all(
                        ctx.device,
                        tlas,
                        water_buffer.trace_image_view,
                        water_ubo,
                        hit_table,
                    )?;
                }
            }

            app.data.raytracing.water_descriptor_tlas.set(tlas);
        }
    }

    let settings = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::WaterRenderSettings>()
        .map(|s| *s)
        .unwrap_or_default();
    if settings.secondary_rays == thyllore_effect_core::WaterSecondaryRays::RayTracingPipeline {
        if let (Some(trace_pipeline), Some(trace_descriptor)) = (
            app.data.raytracing.water_trace_pipeline.as_ref(),
            app.data.raytracing.water_trace_descriptor.as_ref(),
        ) {
            if let Some(effect) = app
                .data
                .ecs_world
                .get_component::<crate::ecs::component::WaterTorusEffect>(waters[0])
            {
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
                    &[trace_descriptor.descriptor_set],
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
                let barrier = vk::ImageMemoryBarrier::builder()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(water_buffer.trace_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build();
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[] as &[vk::MemoryBarrier],
                    &[] as &[vk::BufferMemoryBarrier],
                    &[barrier],
                );
            }
        }
    }

    let hdr_buffer = app
        .data
        .viewport
        .hdr_buffer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("HDR buffer not initialized"))?;

    let Some(water_ubo) = app.data.raytracing.water_ubo.as_ref() else {
        return Ok(());
    };
    let instance_ubos = record_water_ubo_updates(
        app,
        &ctx,
        water_ubo,
        &waters[..instance_count],
        command_buffer,
    )?;

    record_water_caustic_pass(app, water_buffer, hdr_buffer, waters[0], command_buffer);

    thyllore_vulkan_core::renderer::record_water_scene_color_copy(
        &ctx,
        hdr_buffer.color_image,
        water_buffer,
        command_buffer,
    );

    let first_water_accum = waters.first().and_then(|water| {
        app.data
            .ecs_world
            .get_component::<crate::ecs::component::WaterTemporalAccum>(*water)
            .cloned()
    });

    let history_index = first_water_accum
        .as_ref()
        .map(|accum| (accum.frame_index & 1) as usize)
        .unwrap_or(0);

    if first_water_accum
        .as_ref()
        .is_some_and(|accum| accum.history_invalidated)
    {
        clear_water_history_images(&ctx.device.device, water_buffer, command_buffer);
    }

    for (i, (ubo, ubo_dynamic_offset)) in instance_ubos.iter().enumerate() {
        let effect = app
            .data
            .ecs_world
            .get_component::<crate::ecs::component::WaterTorusEffect>(waters[i])
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
            settings.secondary_rays.as_shader_value(),
            settings.debug_view,
        );

        // Record shading pass for this instance
        thyllore_vulkan_core::renderer::record_water_shading_pass(
            &ctx,
            water_buffer,
            shading_pipeline,
            descriptor,
            *ubo_dynamic_offset,
            scissor,
            push_constants,
            image_index,
            history_index,
            command_buffer,
        )?;
    }

    Ok(())
}

/// Both history images stay in SHADER_READ_ONLY_OPTIMAL between frames (shading render pass final layout).
unsafe fn clear_water_history_images(
    device: &Device,
    water_buffer: &thyllore_vulkan_core::resource::WaterBuffer,
    command_buffer: vk::CommandBuffer,
) {
    let black = vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 1.0],
    };

    for &image in &water_buffer.history_images {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[build_color_image_barrier(
                image,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::SHADER_READ,
                vk::AccessFlags::TRANSFER_WRITE,
            )],
        );

        device.cmd_clear_color_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &black,
            &[COLOR_SUBRESOURCE_RANGE],
        );

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[build_color_image_barrier(
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            )],
        );
    }
}

/// Splats refracted light into the accumulation image and adds it to the HDR color.
/// The G-buffer position image stays in GENERAL from the ray query pass.
/// Uploads every instance's UBO before the caustic and shading passes read them in this frame.
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

unsafe fn record_water_caustic_pass(
    app: &App,
    water_buffer: &thyllore_vulkan_core::resource::WaterBuffer,
    hdr_buffer: &thyllore_vulkan_core::resource::HdrBuffer,
    water: crate::ecs::world::Entity,
    command_buffer: vk::CommandBuffer,
) {
    let caustic_strength = app
        .data
        .ecs_world
        .get_component::<crate::ecs::component::WaterTorusEffect>(water)
        .map(|effect| effect.caustic_strength)
        .unwrap_or(0.0);
    if caustic_strength <= 0.0 {
        return;
    }

    let (Some(splat_pipeline), Some(apply_pipeline), Some(descriptor)) = (
        app.data.raytracing.water_caustic_splat_pipeline.as_ref(),
        app.data.raytracing.water_caustic_apply_pipeline.as_ref(),
        app.data.raytracing.water_caustic_descriptor.as_ref(),
    ) else {
        return;
    };

    if !app.data.raytracing.has_valid_tlas()
        || descriptor.splat_descriptor_set == vk::DescriptorSet::null()
    {
        return;
    }

    let device = &app.rrdevice.device;
    let accum_image = water_buffer.caustic_accum_image;
    let hdr_image = hdr_buffer.color_image;

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[build_color_image_barrier(
            accum_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
        )],
    );

    device.cmd_clear_color_image(
        command_buffer,
        accum_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &vk::ClearColorValue { uint32: [0; 4] },
        &[COLOR_SUBRESOURCE_RANGE],
    );

    let pre_splat_barriers = [
        build_color_image_barrier(
            accum_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        ),
        build_color_image_barrier(
            hdr_image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        ),
    ];
    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &pre_splat_barriers,
    );

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

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[build_color_image_barrier(
            accum_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ,
        )],
    );

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

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[build_color_image_barrier(
            hdr_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ,
        )],
    );
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

fn build_color_image_barrier(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) -> vk::ImageMemoryBarrier {
    vk::ImageMemoryBarrier::builder()
        .image(image)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(COLOR_SUBRESOURCE_RANGE)
        .build()
}
