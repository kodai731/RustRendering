use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector3};

use crate::app::FrameContext;
use crate::ecs::component::{ConstraintSet, LineMesh};
use crate::ecs::resource::gizmo::BoneSelectionState;
use crate::ecs::resource::gizmo::TransformGizmoData;
use crate::ecs::resource::gizmo::{
    BoneDisplayStyle, BoneGizmoData, ConstraintGizmoData, SpringBoneGizmoData,
};
use crate::ecs::resource::ProjectionData;
use crate::ecs::resource::{Camera, Exposure, GpuPassTimings, GpuTimingsSink, TransformGizmoState};
use crate::ecs::systems::render_data_systems::{
    bone_gizmo_render_data, constraint_gizmo_render_data, gizmo_mesh_render_data,
    gizmo_selectable_render_data, grid_mesh_render_data, spring_bone_gizmo_render_data,
    transform_gizmo_render_data,
};
use crate::ecs::{
    build_bone_line_mesh, build_box_bone_meshes_with_selection, build_constraint_gizmo_mesh,
    build_octahedral_bone_meshes_with_selection, build_sphere_bone_meshes_with_selection,
    build_spring_bone_gizmo_mesh, gizmo_update_vertex_buffer,
};
use crate::render::RenderBackend;
use crate::vulkanr::renderer::scene_renderer::update_object_ubo;

pub unsafe fn run_render_prep_phase(ctx: &mut FrameContext) -> Result<()> {
    let (view, proj, screen_size, aspect) = {
        let proj_data = ctx.world.resource::<ProjectionData>();
        (
            proj_data.view,
            proj_data.proj,
            proj_data.screen_size,
            proj_data.aspect,
        )
    };

    let camera_position = {
        use crate::ecs::systems::camera_systems::compute_camera_position;
        compute_camera_position(&ctx.camera())
    };

    let mut sub: HashMap<String, f32> = HashMap::new();

    let t = Instant::now();
    update_frame_and_scene_uniforms(ctx, view, proj, screen_size, aspect, camera_position)?;
    sub.insert("uniforms".to_string(), t.elapsed().as_secs_f32() * 1000.0);

    let t = Instant::now();
    crate::ecs::systems::batch_run_update_orbit(&mut ctx.world);
    sub.insert("orbit".to_string(), t.elapsed().as_secs_f32() * 1000.0);
    let t = Instant::now();
    crate::ecs::systems::motion_path_sync(ctx);
    sub.insert(
        "motion_path".to_string(),
        t.elapsed().as_secs_f32() * 1000.0,
    );

    let t = Instant::now();
    crate::ecs::systems::flame_bone_attach_sync(ctx);

    let t = Instant::now();
    crate::ecs::systems::flame_time_advance(ctx);
    crate::ecs::systems::field_manifest_sync(ctx);
    sub.insert("flame_time".to_string(), t.elapsed().as_secs_f32() * 1000.0);

    let t = Instant::now();
    crate::ecs::systems::flame_trail_advance(ctx);
    sub.insert(
        "flame_trail".to_string(),
        t.elapsed().as_secs_f32() * 1000.0,
    );

    let timings_write_time = {
        let t = Instant::now();
        gpu_timings_write(&mut ctx.world);
        t.elapsed().as_secs_f32() * 1000.0
    };
    sub.insert("timings_write".to_string(), timings_write_time);

    if let (Some(mut sink), Some(temporal)) = (
        ctx.world
            .get_resource_mut::<crate::ecs::resource::FlameDumpSink>(),
        ctx.world
            .get_resource::<crate::ecs::resource::FlameTemporalState>(),
    ) {
        let t = Instant::now();
        let flame_entities: Vec<_> = ctx.world.query_flames();
        let effects: Vec<crate::ecs::component::FlameEffect> = flame_entities
            .iter()
            .filter_map(|e| {
                ctx.world
                    .get_component::<crate::ecs::component::FlameEffect>(*e)
                    .cloned()
            })
            .collect();
        let trails: Vec<Option<crate::ecs::component::flame_trail::FlameTrail>> = flame_entities
            .iter()
            .map(|e| {
                ctx.world
                    .get_component::<crate::ecs::component::flame_trail::FlameTrail>(*e)
                    .cloned()
            })
            .collect();
        crate::ecs::systems::flame_dump_system(&mut sink, &*temporal, &effects, &trails);
        sub.insert("flame_dump".to_string(), t.elapsed().as_secs_f32() * 1000.0);
    }

    let t = Instant::now();
    crate::ecs::systems::flame_temporal_accumulate(ctx);
    sub.insert(
        "flame_temporal".to_string(),
        t.elapsed().as_secs_f32() * 1000.0,
    );

    let t = Instant::now();
    let render_data_vec = collect_gizmo_render_data(ctx, camera_position);
    sub.insert(
        "collect_gizmo".to_string(),
        t.elapsed().as_secs_f32() * 1000.0,
    );
    let render_data_refs: Vec<_> = render_data_vec.iter().collect();

    let t = Instant::now();
    if let Err(e) = update_object_ubo(
        &render_data_refs,
        ctx.image_index,
        &ctx.graphics.objects,
        ctx.device,
    ) {
        eprintln!("Failed to update object UBOs: {}", e);
    }
    sub.insert("object_ubo".to_string(), t.elapsed().as_secs_f32() * 1000.0);

    let t = Instant::now();
    update_billboard_ubo(ctx, view, proj)?;
    sub.insert(
        "billboard_ubo".to_string(),
        t.elapsed().as_secs_f32() * 1000.0,
    );

    let t = Instant::now();
    update_grid_gizmo_buffers(ctx)?;
    sub.insert("grid".to_string(), t.elapsed().as_secs_f32() * 1000.0);

    let t = Instant::now();
    update_transform_gizmo_mesh(ctx)?;
    sub.insert(
        "transform_gizmo".to_string(),
        t.elapsed().as_secs_f32() * 1000.0,
    );

    let t = Instant::now();
    update_bone_gizmo_mesh(ctx)?;
    sub.insert("bone_gizmo".to_string(), t.elapsed().as_secs_f32() * 1000.0);

    let t = Instant::now();
    update_constraint_gizmo_mesh(ctx)?;
    sub.insert(
        "constraint_gizmo".to_string(),
        t.elapsed().as_secs_f32() * 1000.0,
    );

    let t = Instant::now();
    update_spring_bone_gizmo_mesh(ctx)?;
    sub.insert(
        "spring_gizmo".to_string(),
        t.elapsed().as_secs_f32() * 1000.0,
    );

    let t = Instant::now();
    crate::ecs::systems::gizmo_systems::run_vertical_lines_update(ctx)?;
    sub.insert(
        "vertical_lines".to_string(),
        t.elapsed().as_secs_f32() * 1000.0,
    );

    ctx.world
        .insert_resource(crate::ecs::resource::RenderPrepSubTimings { timings: sub });

    Ok(())
}

fn gpu_timings_write(world: &mut crate::ecs::World) {
    let timings = match world.get_resource::<GpuPassTimings>() {
        Some(t) => t,
        None => return,
    };
    let frame = timings.frame;
    let passes: Vec<(String, f32)> = timings.passes.clone();
    if passes.is_empty() {
        return;
    }
    let mut sink = match world.get_resource_mut::<GpuTimingsSink>() {
        Some(s) => s,
        None => return,
    };
    if frame == sink.last_frame {
        return;
    }
    let passes_map: serde_json::Map<String, serde_json::Value> = passes
        .into_iter()
        .map(|(label, ms)| (label, serde_json::json!(ms)))
        .collect();

    let mut obj: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    obj.insert("frame".to_string(), serde_json::json!(frame));
    obj.insert("passes".to_string(), serde_json::Value::Object(passes_map));

    if let Some(cpu) = world.get_resource::<crate::ecs::resource::CpuFrameTimings>() {
        obj.insert("cpu_dt_ms".to_string(), serde_json::json!(cpu.dt_ms));
        obj.insert("imgui_vtx".to_string(), serde_json::json!(cpu.imgui_vtx));
        obj.insert("imgui_idx".to_string(), serde_json::json!(cpu.imgui_idx));
        let cpu_map: serde_json::Map<String, serde_json::Value> = cpu
            .stages
            .iter()
            .map(|(label, ms)| (label.clone(), serde_json::json!(ms)))
            .collect();
        obj.insert("cpu".to_string(), serde_json::Value::Object(cpu_map));
    }
    if let Some(up) = world.get_resource::<crate::ecs::resource::UpdatePhaseTimings>() {
        let up_map: serde_json::Map<String, serde_json::Value> = up
            .stages
            .iter()
            .map(|(label, ms)| (label.clone(), serde_json::json!(ms)))
            .collect();
        obj.insert(
            "update_phases".to_string(),
            serde_json::Value::Object(up_map),
        );
    }
    if let Some(rps) = world.get_resource::<crate::ecs::resource::RenderPrepSubTimings>() {
        let rps_map: serde_json::Map<String, serde_json::Value> = rps
            .timings
            .iter()
            .map(|(label, ms)| (label.clone(), serde_json::json!(ms)))
            .collect();
        obj.insert(
            "render_prep_sub".to_string(),
            serde_json::Value::Object(rps_map),
        );
    }

    let line = serde_json::Value::Object(obj);
    if let Err(e) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&sink.path)
        .and_then(|mut f| {
            use std::io::Write;
            writeln!(f, "{}", line)
        })
    {
        eprintln!("gpu timings write failed: {}", e);
    }
    sink.last_frame = frame;
}

unsafe fn update_mesh_entity_transforms(ctx: &mut FrameContext) -> Result<()> {
    use crate::ecs::world::{GlobalTransform, MeshRef};
    use crate::render::ObjectUBO;

    let transforms: Vec<(usize, Matrix4<f32>)> = ctx
        .world
        .iter_components::<MeshRef>()
        .map(|(entity, mesh_ref)| {
            let model_matrix = ctx
                .world
                .get_component::<GlobalTransform>(entity)
                .map(|gt| gt.0)
                .unwrap_or_else(Matrix4::identity);
            (mesh_ref.object_index, model_matrix)
        })
        .collect();

    for (object_index, model_matrix) in transforms {
        let ubo = ObjectUBO {
            model: model_matrix,
        };
        ctx.graphics
            .objects
            .update(ctx.device, ctx.image_index, object_index, &ubo)?;
    }

    Ok(())
}

unsafe fn update_frame_and_scene_uniforms(
    ctx: &mut FrameContext,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    screen_size: cgmath::Vector2<f32>,
    aspect: f32,
    camera_position: Vector3<f32>,
) -> Result<()> {
    let light_position = ctx.light_state().light_position;

    {
        let proj_data = ProjectionData {
            view,
            proj,
            screen_size,
            aspect,
        };
        let image_index = ctx.image_index;
        let mut backend = ctx.create_backend();
        backend.update_frame_ubo(
            &proj_data,
            camera_position,
            light_position,
            Vector3::new(1.0, 1.0, 1.0),
            image_index,
        )?;
    }

    update_mesh_entity_transforms(ctx)?;

    let light = ctx.light_state();
    let light_pos = light.light_position;
    let shadow_strength = light.shadow_strength;
    let distance_attenuation = light.distance_attenuation;
    drop(light);

    let debug_mode = ctx.debug_view_state().debug_view_mode.as_int();

    let exposure_value = ctx
        .world
        .get_resource::<Exposure>()
        .map(|e| e.exposure_value)
        .unwrap_or(1.0);

    let mut backend = ctx.create_backend();
    backend.update_scene_uniform(
        view,
        proj,
        light_pos,
        Vector3::new(1.0, 1.0, 1.0),
        debug_mode,
        shadow_strength,
        distance_attenuation,
        exposure_value,
    )?;

    Ok(())
}

fn collect_gizmo_render_data(
    ctx: &FrameContext,
    camera_position: Vector3<f32>,
) -> Vec<crate::ecs::component::RenderData> {
    let mut render_data_vec = vec![
        grid_mesh_render_data(&ctx.grid_mesh()),
        gizmo_mesh_render_data(&ctx.gizmo()),
        gizmo_selectable_render_data(&ctx.light_gizmo(), camera_position),
    ];

    if ctx.world.contains_resource::<BoneGizmoData>() {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        if bone_gizmo.visible {
            render_data_vec.extend(bone_gizmo_render_data(&bone_gizmo));
        }
    }

    if ctx.world.contains_resource::<ConstraintGizmoData>() {
        let cg = ctx.world.resource::<ConstraintGizmoData>();
        if cg.visible {
            render_data_vec.extend(constraint_gizmo_render_data(&cg));
        }
    }

    if ctx.world.contains_resource::<SpringBoneGizmoData>() {
        let sg = ctx.world.resource::<SpringBoneGizmoData>();
        if sg.visible {
            render_data_vec.extend(spring_bone_gizmo_render_data(&sg));
        }
    }

    if ctx.world.contains_resource::<TransformGizmoData>() {
        let tg = ctx.world.resource::<TransformGizmoData>();
        let gizmo_scale = ctx
            .world
            .get_resource::<TransformGizmoState>()
            .map(|s| s.gizmo_scale)
            .unwrap_or(0.08);
        render_data_vec.extend(transform_gizmo_render_data(
            &tg,
            camera_position,
            gizmo_scale,
        ));
    }

    render_data_vec
}

unsafe fn update_billboard_ubo(
    ctx: &mut FrameContext,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
) -> Result<()> {
    let model_matrix = {
        let billboard = ctx.billboard();
        billboard
            .transform
            .as_ref()
            .map(|t| t.model_matrix)
            .unwrap_or(Matrix4::identity())
    };

    let image_index = ctx.image_index;
    ctx.update_billboard_ubo_internal(model_matrix, view, proj, image_index)?;

    Ok(())
}

unsafe fn update_grid_gizmo_buffers(ctx: &mut FrameContext) -> Result<()> {
    let mesh = ctx.gizmo().mesh.clone();
    let backend = ctx.create_backend();
    gizmo_update_vertex_buffer(&mesh, &backend)?;

    Ok(())
}

unsafe fn update_transform_gizmo_mesh(ctx: &mut FrameContext) -> Result<()> {
    if !ctx.world.contains_resource::<TransformGizmoData>() {
        return Ok(());
    }

    let visible = {
        let tg = ctx.world.resource::<TransformGizmoData>();
        tg.visible
    };

    if !visible {
        return Ok(());
    }

    let (mode, active_handle) = {
        let tg = ctx.world.resource::<TransformGizmoData>();
        let state = ctx
            .world
            .resource::<crate::ecs::resource::TransformGizmoState>();
        (state.mode, tg.active_handle)
    };

    let camera_dir = ctx.camera_direction();

    let (mut line_mesh_clone, mut solid_mesh_clone) = {
        let tg = ctx.world.resource::<TransformGizmoData>();
        let mut line_mesh_clone = tg.line_mesh.clone();
        let mut solid_mesh_clone = tg.solid_mesh.clone();
        drop(tg);
        (line_mesh_clone, solid_mesh_clone)
    };

    match mode {
        crate::ecs::resource::TransformGizmoMode::Translate => {
            crate::ecs::systems::transform_gizmo_systems::build_translate_gizmo_meshes(
                active_handle,
                &mut line_mesh_clone,
                &mut solid_mesh_clone,
            );
        }
        crate::ecs::resource::TransformGizmoMode::Rotate => {
            crate::ecs::systems::transform_gizmo_systems::build_rotate_gizmo_meshes(
                active_handle,
                camera_dir,
                &mut line_mesh_clone,
                &mut solid_mesh_clone,
            );
        }
        crate::ecs::resource::TransformGizmoMode::Scale => {
            crate::ecs::systems::transform_gizmo_systems::build_scale_gizmo_meshes(
                active_handle,
                &mut line_mesh_clone,
                &mut solid_mesh_clone,
            );
        }
    }

    {
        let frame_slot = ctx.frame_slot;
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut line_mesh_clone, frame_slot)?;
        backend.update_or_create_line_buffers(&mut solid_mesh_clone, frame_slot)?;
    }

    {
        let mut tg = ctx.world.resource_mut::<TransformGizmoData>();
        tg.line_mesh = line_mesh_clone;
        tg.solid_mesh = solid_mesh_clone;
    }

    Ok(())
}

unsafe fn update_bone_gizmo_mesh(ctx: &mut FrameContext) -> Result<()> {
    if !ctx.world.contains_resource::<BoneGizmoData>() {
        return Ok(());
    }

    let (
        visible,
        display_style,
        skeleton_id,
        transforms,
        offsets,
        distance_scaling_enabled,
        distance_scaling_factor,
        mesh_scale,
    ) = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        (
            bone_gizmo.visible,
            bone_gizmo.display_style,
            bone_gizmo.cached_skeleton_id,
            bone_gizmo.cached_global_transforms.clone(),
            bone_gizmo.bone_local_offsets.clone(),
            bone_gizmo.distance_scaling_enabled,
            bone_gizmo.distance_scaling_factor,
            bone_gizmo.mesh_scale,
        )
    };

    if !visible {
        return Ok(());
    }

    let Some(skel_id) = skeleton_id else {
        return Ok(());
    };

    let Some(skeleton) = ctx.assets.get_skeleton_by_skeleton_id(skel_id) else {
        return Ok(());
    };
    let skeleton = skeleton.clone();

    let visual_scale = compute_visual_scale(
        ctx,
        &transforms,
        distance_scaling_enabled,
        distance_scaling_factor,
    );

    match display_style {
        BoneDisplayStyle::Stick => {
            update_stick_bone_mesh(ctx, &skeleton, &transforms, &offsets, mesh_scale)?;
        }
        BoneDisplayStyle::Octahedral => {
            update_octahedral_bone_mesh(
                ctx,
                &skeleton,
                &transforms,
                &offsets,
                visual_scale,
                mesh_scale,
            )?;
        }
        BoneDisplayStyle::Box => {
            update_box_bone_mesh(
                ctx,
                &skeleton,
                &transforms,
                &offsets,
                visual_scale,
                mesh_scale,
            )?;
        }
        BoneDisplayStyle::Sphere => {
            update_sphere_bone_mesh(
                ctx,
                &skeleton,
                &transforms,
                &offsets,
                visual_scale,
                mesh_scale,
            )?;
        }
    }

    Ok(())
}

fn compute_visual_scale(
    ctx: &FrameContext,
    transforms: &[Matrix4<f32>],
    distance_scaling_enabled: bool,
    distance_scaling_factor: f32,
) -> f32 {
    if !distance_scaling_enabled || transforms.is_empty() {
        return 1.0;
    }

    let camera_pos = {
        use crate::ecs::systems::camera_systems::compute_camera_position;
        compute_camera_position(&ctx.world.resource::<Camera>())
    };

    let mut center = Vector3::new(0.0f32, 0.0, 0.0);
    for t in transforms.iter() {
        center.x += t[3][0];
        center.y += t[3][1];
        center.z += t[3][2];
    }
    let count = transforms.len() as f32;
    center /= count;

    let distance = (center - camera_pos).magnitude();
    (distance * distance_scaling_factor).max(0.1)
}

unsafe fn update_stick_bone_mesh(
    ctx: &mut FrameContext,
    skeleton: &crate::animation::Skeleton,
    transforms: &[Matrix4<f32>],
    offsets: &[[f32; 3]],
    mesh_scale: f32,
) -> Result<()> {
    {
        let mut bone_gizmo = ctx.world.resource_mut::<BoneGizmoData>();
        build_bone_line_mesh(
            skeleton,
            transforms,
            offsets,
            mesh_scale,
            None,
            &mut bone_gizmo.stick_mesh,
        );
    }

    let mut mesh_clone = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        bone_gizmo.stick_mesh.clone()
    };

    {
        let frame_slot = ctx.frame_slot;
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut mesh_clone, frame_slot)?;
    }

    {
        let mut bone_gizmo = ctx.world.resource_mut::<BoneGizmoData>();
        bone_gizmo.stick_mesh.vertex_buffer_handles = mesh_clone.vertex_buffer_handles;
        bone_gizmo.stick_mesh.index_buffer_handles = mesh_clone.index_buffer_handles;
        bone_gizmo.stick_mesh.last_written_slot = mesh_clone.last_written_slot;
    }

    Ok(())
}

unsafe fn update_octahedral_bone_mesh(
    ctx: &mut FrameContext,
    skeleton: &crate::animation::Skeleton,
    transforms: &[Matrix4<f32>],
    offsets: &[[f32; 3]],
    visual_scale: f32,
    mesh_scale: f32,
) -> Result<()> {
    let selection = ctx
        .world
        .get_resource::<BoneSelectionState>()
        .map(|s| (*s).clone())
        .unwrap_or_default();

    let (mut solid_mesh, mut wire_mesh) = {
        let bg = ctx.world.resource::<BoneGizmoData>();
        let mut solid_mesh = bg.solid_mesh.clone();
        let mut wire_mesh = bg.wire_mesh.clone();
        drop(bg);
        (solid_mesh, wire_mesh)
    };
    build_octahedral_bone_meshes_with_selection(
        skeleton,
        transforms,
        offsets,
        &selection,
        visual_scale,
        mesh_scale,
        None,
        &mut solid_mesh,
        &mut wire_mesh,
    );

    {
        let frame_slot = ctx.frame_slot;
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut solid_mesh, frame_slot)?;
        backend.update_or_create_line_buffers(&mut wire_mesh, frame_slot)?;
    }

    {
        let mut bone_gizmo = ctx.world.resource_mut::<BoneGizmoData>();
        bone_gizmo.solid_mesh = solid_mesh;
        bone_gizmo.wire_mesh = wire_mesh;
    }

    Ok(())
}

unsafe fn update_box_bone_mesh(
    ctx: &mut FrameContext,
    skeleton: &crate::animation::Skeleton,
    transforms: &[Matrix4<f32>],
    offsets: &[[f32; 3]],
    visual_scale: f32,
    mesh_scale: f32,
) -> Result<()> {
    let selection = ctx
        .world
        .get_resource::<BoneSelectionState>()
        .map(|s| (*s).clone())
        .unwrap_or_default();

    let (mut solid_mesh, mut wire_mesh) = {
        let bg = ctx.world.resource::<BoneGizmoData>();
        let mut solid_mesh = bg.solid_mesh.clone();
        let mut wire_mesh = bg.wire_mesh.clone();
        drop(bg);
        (solid_mesh, wire_mesh)
    };
    build_box_bone_meshes_with_selection(
        skeleton,
        transforms,
        offsets,
        &selection,
        visual_scale,
        mesh_scale,
        None,
        &mut solid_mesh,
        &mut wire_mesh,
    );

    {
        let frame_slot = ctx.frame_slot;
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut solid_mesh, frame_slot)?;
        backend.update_or_create_line_buffers(&mut wire_mesh, frame_slot)?;
    }

    {
        let mut bone_gizmo = ctx.world.resource_mut::<BoneGizmoData>();
        bone_gizmo.solid_mesh = solid_mesh;
        bone_gizmo.wire_mesh = wire_mesh;
    }

    Ok(())
}

unsafe fn update_sphere_bone_mesh(
    ctx: &mut FrameContext,
    skeleton: &crate::animation::Skeleton,
    transforms: &[Matrix4<f32>],
    offsets: &[[f32; 3]],
    visual_scale: f32,
    mesh_scale: f32,
) -> Result<()> {
    let selection = ctx
        .world
        .get_resource::<BoneSelectionState>()
        .map(|s| (*s).clone())
        .unwrap_or_default();

    let mut solid_mesh = LineMesh::default();
    let mut wire_mesh = LineMesh::default();

    {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        solid_mesh.vertex_buffer_handles[0] = bone_gizmo.solid_mesh.vertex_buffer_handles[0];
        solid_mesh.index_buffer_handles[0] = bone_gizmo.solid_mesh.index_buffer_handles[0];
        wire_mesh.vertex_buffer_handles[0] = bone_gizmo.wire_mesh.vertex_buffer_handles[0];
        wire_mesh.index_buffer_handles[0] = bone_gizmo.wire_mesh.index_buffer_handles[0];
    }

    build_sphere_bone_meshes_with_selection(
        skeleton,
        transforms,
        offsets,
        &selection,
        visual_scale,
        mesh_scale,
        None,
        &mut solid_mesh,
        &mut wire_mesh,
    );

    {
        let frame_slot = ctx.frame_slot;
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut solid_mesh, frame_slot)?;
        backend.update_or_create_line_buffers(&mut wire_mesh, frame_slot)?;
    }

    {
        let mut bone_gizmo = ctx.world.resource_mut::<BoneGizmoData>();
        bone_gizmo.solid_mesh = solid_mesh;
        bone_gizmo.wire_mesh = wire_mesh;
    }

    Ok(())
}

unsafe fn update_constraint_gizmo_mesh(ctx: &mut FrameContext) -> Result<()> {
    if !ctx.world.contains_resource::<ConstraintGizmoData>() {
        return Ok(());
    }
    if !ctx.world.contains_resource::<BoneGizmoData>() {
        return Ok(());
    }

    let visible = {
        let cg = ctx.world.resource::<ConstraintGizmoData>();
        cg.visible
    };
    if !visible {
        return Ok(());
    }

    let (skeleton_id, transforms, offsets, constraint_mesh_scale) = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        (
            bone_gizmo.cached_skeleton_id,
            bone_gizmo.cached_global_transforms.clone(),
            bone_gizmo.bone_local_offsets.clone(),
            bone_gizmo.mesh_scale,
        )
    };

    let Some(skel_id) = skeleton_id else {
        return Ok(());
    };
    let Some(skeleton) = ctx.assets.get_skeleton_by_skeleton_id(skel_id) else {
        return Ok(());
    };
    let skeleton = skeleton.clone();

    let constraint_set = ctx
        .world
        .iter_constrained_entities()
        .next()
        .map(|(_, cs)| cs.clone());

    let Some(constraint_set) = constraint_set else {
        return Ok(());
    };

    let mut wire_mesh = LineMesh::default();

    {
        let cg = ctx.world.resource::<ConstraintGizmoData>();
        wire_mesh.vertex_buffer_handles[0] = cg.wire_mesh.vertex_buffer_handles[0];
        wire_mesh.index_buffer_handles[0] = cg.wire_mesh.index_buffer_handles[0];
    }

    build_constraint_gizmo_mesh(
        &constraint_set,
        &skeleton,
        &transforms,
        &offsets,
        constraint_mesh_scale,
        &mut wire_mesh,
    );

    {
        let frame_slot = ctx.frame_slot;
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut wire_mesh, frame_slot)?;
    }

    {
        let mut cg = ctx.world.resource_mut::<ConstraintGizmoData>();
        cg.wire_mesh = wire_mesh;
    }

    Ok(())
}

unsafe fn update_spring_bone_gizmo_mesh(ctx: &mut FrameContext) -> Result<()> {
    if !ctx.world.contains_resource::<SpringBoneGizmoData>() {
        return Ok(());
    }
    if !ctx.world.contains_resource::<BoneGizmoData>() {
        return Ok(());
    }

    let visible = {
        let sg = ctx.world.resource::<SpringBoneGizmoData>();
        sg.visible
    };
    if !visible {
        return Ok(());
    }

    let (transforms, offsets, mesh_scale) = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        (
            bone_gizmo.cached_global_transforms.clone(),
            bone_gizmo.bone_local_offsets.clone(),
            bone_gizmo.mesh_scale,
        )
    };

    use crate::ecs::component::{SpringBoneSetup, WithSpringBone};
    let setup = ctx
        .world
        .iter_components::<WithSpringBone>()
        .next()
        .and_then(|(e, _)| ctx.world.get_component::<SpringBoneSetup>(e).cloned());

    let Some(setup) = setup else {
        return Ok(());
    };

    let mut wire_mesh = LineMesh::default();

    {
        let sg = ctx.world.resource::<SpringBoneGizmoData>();
        wire_mesh.vertex_buffer_handles[0] = sg.wire_mesh.vertex_buffer_handles[0];
        wire_mesh.index_buffer_handles[0] = sg.wire_mesh.index_buffer_handles[0];
    }

    build_spring_bone_gizmo_mesh(&setup, &transforms, &offsets, mesh_scale, &mut wire_mesh);

    {
        let frame_slot = ctx.frame_slot;
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut wire_mesh, frame_slot)?;
    }

    {
        let mut sg = ctx.world.resource_mut::<SpringBoneGizmoData>();
        sg.wire_mesh = wire_mesh;
    }

    Ok(())
}
